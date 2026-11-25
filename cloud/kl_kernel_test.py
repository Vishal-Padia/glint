import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

# Linux torch wheels bundle Triton.
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch").add_local_dir(ROOT / "glint", "/root/glint")
app = modal.App("glint-kl-kernel-test", image=image)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


# Any of these is fine for a kernel check; Modal takes the first one free.
@app.function(gpu=["H100", "A100-80GB", "A100"], timeout=10 * 60)
def check():
    import torch
    import triton

    from glint.kl_kernel import kl_fwd_kernel

    log(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}, triton {triton.__version__}")

    N, V_PADDED, V, BLOCK_V = 8, 151_936, 151_665, 4096
    torch.manual_seed(0)
    s = (torch.randn(N, V_PADDED, device="cuda") * 5).bfloat16()
    t = (torch.randn(N, V_PADDED, device="cuda") * 5).bfloat16()
    # Huge logits in the padding columns: if the mask leaks, lse jumps to ~1000.
    s[:, V:] = 1000.0
    t[:, V:] = 1000.0

    kl = torch.empty(N, device="cuda", dtype=torch.float32)
    lse_s = torch.empty_like(kl)
    lse_t = torch.empty_like(kl)

    log(f"pass 1: N={N}, V={V} (row stride {V_PADDED}), BLOCK_V={BLOCK_V}, {triton.cdiv(V, BLOCK_V)} tiles per row")
    kl_fwd_kernel[(N,)](s, t, kl, lse_s, lse_t, s.stride(0), t.stride(0), V, BLOCK_V=BLOCK_V)
    torch.cuda.synchronize()

    ref_s = torch.logsumexp(s[:, :V].double(), dim=-1)
    ref_t = torch.logsumexp(t[:, :V].double(), dim=-1)
    err_s = (lse_s.double() - ref_s).abs().max().item()
    err_t = (lse_t.double() - ref_t).abs().max().item()
    log(f"lse_s kernel {lse_s[:3].tolist()}")
    log(f"lse_s ref    {ref_s[:3].tolist()}")
    log(f"max |lse_s - ref| = {err_s:.2e}, max |lse_t - ref| = {err_t:.2e}")

    assert err_s < 1e-4 and err_t < 1e-4, "lse mismatch"
    log("PASS 1 OK")

    # Pass 2: KL against the float64 reference, in three regimes.
    from glint.kl_reference import reverse_kl

    noise = torch.randn(N, V_PADDED, device="cuda")
    cases = {
        "random (large KL)": (s, t),
        "identical (KL = 0)": (s, s.clone()),
        "close (small KL)": (s, (s.float() + 0.1 * noise).bfloat16()),  # what training looks like near convergence
    }
    ok = True
    for name, (cs, ct) in cases.items():
        kl_fwd_kernel[(N,)](cs, ct, kl, lse_s, lse_t, cs.stride(0), ct.stride(0), V, BLOCK_V=BLOCK_V)
        ref = reverse_kl(cs[:, :V], ct[:, :V])
        err = (kl.double() - ref).abs().max().item()
        log(f"{name:20s} kl {kl[0].item():.6e}  ref {ref[0].item():.6e}  max abs err {err:.2e}")
        ok &= err < 1e-4
    assert ok, "kl mismatch"
    log("PASS 2 OK")

    # Wrapper: real Qwen2.5 shapes (student 1.5B, teacher 7B), N not a multiple of the chunk size.
    from glint.kl_kernel import reverse_kl_chunked

    N, D_S, D_T, V_S, V_T, CHUNK = 3000, 1536, 3584, 151_936, 152_064, 1024
    h_s = torch.randn(N, D_S, device="cuda").bfloat16()
    h_t = torch.randn(N, D_T, device="cuda").bfloat16()
    w_s = (torch.randn(V_S, D_S, device="cuda") * 3 / D_S**0.5).bfloat16()  # logit std ~3
    w_t = (torch.randn(V_T, D_T, device="cuda") * 3 / D_T**0.5).bfloat16()

    log(f"wrapper: N={N} in chunks of {CHUNK}, student [{V_S}, {D_S}], teacher [{V_T}, {D_T}]")
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    kl_w, _, _ = reverse_kl_chunked(h_s, w_s, h_t, w_t, V=V, chunk_rows=CHUNK)
    torch.cuda.synchronize()
    log(f"wrapper peak extra memory {(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB")

    # Reference from the full bf16 logits. Chunked and full matmuls may round a few logits
    # differently, so the bar is relative 1e-3: broken plumbing (strides, tail chunk) is O(1) off.
    ref = reverse_kl((h_s @ w_s.T)[:, :V], (h_t @ w_t.T)[:, :V])
    rel = ((kl_w.double() - ref).abs() / ref.abs()).max().item()
    log(f"wrapper kl[0] {kl_w[0].item():.6f}  ref {ref[0].item():.6f}  kl[-1] {kl_w[-1].item():.6f}  ref {ref[-1].item():.6f}")
    log(f"wrapper max rel err {rel:.2e}")
    assert rel < 1e-3, "wrapper mismatch"
    log("WRAPPER OK")

    # Backward, loss = sum_i g_i * K_i. Three comparisons, to separate bugs from bf16 rounding.
    from glint.kl_kernel import kl_bwd_kernel, reverse_kl_bwd_chunked

    def rel_err(a, b):  # ||a - b|| / ||b||
        return ((a.double() - b).norm() / b.norm()).item()

    N_B, CHUNK_B = 300, 128  # small N keeps the float64 reference cheap; 3 chunks, last one partial
    hs, ht = h_s[:N_B].contiguous(), h_t[:N_B].contiguous()
    g = torch.rand(N_B, device="cuda")
    s_b, t_b = hs @ w_s.T, ht @ w_t.T  # bf16 logits, shared by the kernel and the reference below
    kl_b, lse_s_b, lse_t_b = (torch.empty(N_B, device="cuda") for _ in range(3))
    kl_fwd_kernel[(N_B,)](s_b, t_b, kl_b, lse_s_b, lse_t_b, s_b.stride(0), t_b.stride(0), V, BLOCK_V=BLOCK_V)

    # Reference dL/ds: float64 autograd starting from the same bf16 logits.
    s64 = s_b.double().requires_grad_()
    (reverse_kl(s64[:, :V], t_b.double()[:, :V]) * g.double()).sum().backward()

    # 1. Kernel alone: only error source is storing grad_s in bf16.
    grad_s = torch.zeros_like(s_b)
    kl_bwd_kernel[(N_B,)](s_b, t_b, lse_s_b, lse_t_b, kl_b, g, grad_s,
                          s_b.stride(0), t_b.stride(0), grad_s.stride(0), V, BLOCK_V=BLOCK_V)
    e_s, pad_s = rel_err(grad_s, s64.grad), grad_s[:, V:].abs().max().item()
    log(f"backward kernel   grad_s   rel err {e_s:.2e}, max |padding cols| {pad_s:.1e}")

    # 2. Wrapper vs the same-logits reference pushed through the matmuls in float64.
    grad_h, grad_w = reverse_kl_bwd_chunked(hs, w_s, ht, w_t, lse_s_b, lse_t_b, kl_b, g, V=V, chunk_rows=CHUNK_B)
    e_h = rel_err(grad_h, s64.grad @ w_s.double())
    e_w = rel_err(grad_w, s64.grad.T @ hs.double())
    pad_w = grad_w[V:].abs().max().item()
    log(f"backward wrapper  grad_h_s rel err {e_h:.2e}, grad_w_s rel err {e_w:.2e}, max |padding rows| {pad_w:.1e}")

    # 3. Info only: vs logits computed in float64 from h and w, i.e. what bf16 logits cost.
    hs64, ws64 = hs.double().requires_grad_(), w_s.double().requires_grad_()
    (reverse_kl((hs64 @ ws64.T)[:, :V], (ht.double() @ w_t.double().T)[:, :V]) * g.double()).sum().backward()
    log(f"vs float64 logits grad_h_s rel err {rel_err(grad_h, hs64.grad):.2e}, grad_w_s rel err {rel_err(grad_w, ws64.grad):.2e}")

    assert e_s < 1e-2 and e_h < 1e-2 and e_w < 1e-2 and pad_s == 0 and pad_w == 0, "backward mismatch"
    log("BACKWARD OK")

    # Autograd: the Function must reproduce the wrapper's grads, including when g comes from
    # kl.mean() (autograd then passes a stride-0 expanded tensor, not N real values).
    from glint.kl_kernel import reverse_kl_fused

    def fused_grads(loss_fn):
        h = hs.clone().requires_grad_()
        w = w_s.clone().requires_grad_()
        loss_fn(reverse_kl_fused(h, w, ht, w_t, V=V, chunk_rows=CHUNK_B)).backward()
        return h.grad, w.grad

    gh, gw = fused_grads(lambda k: (k * g).sum())
    # Same path by hand: the forward wrapper (not the direct kernel call above, whose launch
    # settings differ and so round the reductions differently), then the backward wrapper.
    kl_r, lse_s_r, lse_t_r = reverse_kl_chunked(hs, w_s, ht, w_t, V=V, chunk_rows=CHUNK_B)
    gh_r, gw_r = reverse_kl_bwd_chunked(hs, w_s, ht, w_t, lse_s_r, lse_t_r, kl_r, g, V=V, chunk_rows=CHUNK_B)
    same = torch.equal(gh, gh_r) and torch.equal(gw, gw_r)
    log(f"autograd (kl * g).sum(): grads identical to the wrapper's: {same}")

    s64m = s_b.double().requires_grad_()
    reverse_kl(s64m[:, :V], t_b.double()[:, :V]).mean().backward()
    gh_m, gw_m = fused_grads(lambda k: k.mean())
    e_hm = rel_err(gh_m, s64m.grad @ w_s.double())
    e_wm = rel_err(gw_m, s64m.grad.T @ hs.double())
    log(f"autograd kl.mean(): grad_h_s rel err {e_hm:.2e}, grad_w_s rel err {e_wm:.2e}")
    assert same and e_hm < 1e-2 and e_wm < 1e-2, "autograd mismatch"
    log("AUTOGRAD OK")


@app.local_entrypoint()
def main():
    log("launching check() on a Modal H100 or A100 (image build first if anything changed)")
    check.remote()
    log("done")

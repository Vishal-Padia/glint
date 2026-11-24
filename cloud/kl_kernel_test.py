"""Check the reverse-KL Triton kernel on a Modal GPU (Triton has no macOS wheels).

Run from the glint repo root, in the capstan workspace:
    MODAL_PROFILE=capstan uv run modal run cloud/kl_kernel_test.py > logs/kl_kernel_fwd.log 2>&1
"""

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

    from glint.kl_kernel import kl_kernel

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
    kl_kernel[(N,)](s, t, kl, lse_s, lse_t, s.stride(0), t.stride(0), V, BLOCK_V=BLOCK_V)
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
        kl_kernel[(N,)](cs, ct, kl, lse_s, lse_t, cs.stride(0), ct.stride(0), V, BLOCK_V=BLOCK_V)
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
    kl_w = reverse_kl_chunked(h_s, w_s, h_t, w_t, V=V, chunk_rows=CHUNK)
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


@app.local_entrypoint()
def main():
    log("launching check() on a Modal H100 or A100 (image build first if anything changed)")
    check.remote()
    log("done")

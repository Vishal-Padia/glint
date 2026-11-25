import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch").add_local_dir(ROOT / "glint", "/root/glint")
app = modal.App("glint-kl-bench", image=image)

V, D_S, D_T, V_S, V_T = 151_665, 1536, 3584, 151_936, 152_064
ROWS = [1024, 4096, 16_384]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


@app.function(gpu="H100", timeout=20 * 60)
def bench():
    import torch
    import torch.nn.functional as F

    from glint.kl_kernel import reverse_kl_fused

    log(f"GPU: {torch.cuda.get_device_name()}, torch {torch.__version__}")
    MiB = 2**20

    def torch_kl(h_s, w_s, h_t, w_t):
        s = (h_s @ w_s.T)[:, :V].float()
        with torch.no_grad():  # frozen teacher, same as ours
            t = (h_t @ w_t.T)[:, :V].float()
        log_p, log_q = F.log_softmax(s, -1), F.log_softmax(t, -1)
        return (log_p.exp() * (log_p - log_q)).sum(-1)

    contenders = {"pytorch": torch_kl, "fused": reverse_kl_fused}

    torch.manual_seed(0)
    w_s = (torch.randn(V_S, D_S, device="cuda") * 3 / D_S**0.5).bfloat16().requires_grad_()
    w_t = (torch.randn(V_T, D_T, device="cuda") * 3 / D_T**0.5).bfloat16()

    def matmul_tflop(n):  # forward: student + teacher projections; backward: both again + grad_h_s + grad_w_s
        return 2 * n * (2 * D_S * V_S + 2 * D_T * V_T + 2 * D_S * V_S) / 1e12

    for n in ROWS:
        h_s = torch.randn(n, D_S, device="cuda").bfloat16().requires_grad_()
        h_t = torch.randn(n, D_T, device="cuda").bfloat16()
        kls = {}
        for name, fn in contenders.items():

            def step():
                h_s.grad = w_s.grad = None
                kl = fn(h_s, w_s, h_t, w_t)
                kl.mean().backward()
                return kl.detach()

            try:
                h_s.grad = w_s.grad = None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                base = torch.cuda.memory_allocated()
                kls[name] = step()
                torch.cuda.synchronize()
                peak = (torch.cuda.max_memory_allocated() - base) / MiB
                for _ in range(3):
                    step()
                times = []
                for _ in range(10):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    step()
                    b.record()
                    torch.cuda.synchronize()
                    times.append(a.elapsed_time(b))
                ms = sorted(times)[len(times) // 2]
                extra = f"  ({matmul_tflop(n) / (ms / 1e3):.0f} TFLOP/s of fused-path matmul work)" if name == "fused" else ""
                log(f"N={n:6d} {name:8s} fwd+bwd {ms:8.1f} ms   peak extra {peak:8.0f} MiB{extra}")
            except torch.OutOfMemoryError:
                log(f"N={n:6d} {name:8s} OOM")
            h_s.grad = w_s.grad = None
            torch.cuda.empty_cache()
        if len(kls) == 2:
            rel = ((kls["fused"] - kls["pytorch"]).abs() / kls["pytorch"].abs()).max().item()
            log(f"N={n:6d} KL agreement: max rel diff {rel:.1e}")
        del h_s, h_t, kls

    # Where the fused path spends GPU time, at the largest N (Q6).
    n = ROWS[-1]
    h_s = torch.randn(n, D_S, device="cuda").bfloat16().requires_grad_()
    h_t = torch.randn(n, D_T, device="cuda").bfloat16()
    for _ in range(2):
        h_s.grad = w_s.grad = None
        reverse_kl_fused(h_s, w_s, h_t, w_t).mean().backward()
    torch.cuda.synchronize()
    h_s.grad = w_s.grad = None
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        reverse_kl_fused(h_s, w_s, h_t, w_t).mean().backward()
        torch.cuda.synchronize()

    buckets, others = {"matmul": 0.0, "triton fwd": 0.0, "triton bwd": 0.0, "other": 0.0}, []
    for e in prof.key_averages():
        us = getattr(e, "self_device_time_total", 0) or 0
        if us == 0:
            continue
        name = e.key
        if "kl_fwd_kernel" in name:
            buckets["triton fwd"] += us
        elif "kl_bwd_kernel" in name:
            buckets["triton bwd"] += us
        elif any(k in name.lower() for k in ("gemm", "nvjet", "cutlass", "xmma")):
            buckets["matmul"] += us
        else:
            buckets["other"] += us
            others.append((us, name))
    total = sum(buckets.values())
    log(f"N={n} fused GPU time breakdown, total {total / 1e3:.1f} ms:")
    for k, us in buckets.items():
        log(f"  {k:10s} {us / 1e3:7.1f} ms  {100 * us / total:5.1f}%")
    log("  biggest 'other' kernels:")
    for us, name in sorted(others, reverse=True)[:6]:
        log(f"    {us / 1e3:6.1f} ms  {name[:90]}")
    log("BENCH DONE")


@app.local_entrypoint()
def main():
    log("launching bench() on a Modal H100")
    bench.remote()
    log("done")

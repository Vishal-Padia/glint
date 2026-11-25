import itertools
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch").add_local_dir(ROOT / "glint", "/root/glint")
app = modal.App("glint-kl-tune", image=image)

N, V, V_S, V_T = 1024, 151_665, 151_936, 152_064  # one chunk, real widths
BLOCKS = [1024, 2048, 4096, 8192, 16384]
WARPS = [4, 8, 16]
STAGES = [1, 2, 3]
H100_TBPS = 3.35


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


@app.function(gpu="H100", timeout=30 * 60)
def tune():
    import torch
    from triton.testing import do_bench

    from glint.kl_kernel import kl_bwd_kernel, kl_fwd_kernel

    log(f"GPU: {torch.cuda.get_device_name()}")
    torch.manual_seed(0)
    s = (torch.randn(N, V_S, device="cuda") * 3).bfloat16()
    t = (torch.randn(N, V_T, device="cuda") * 3).bfloat16()
    kl, lse_s, lse_t = (torch.empty(N, device="cuda") for _ in range(3))
    g = torch.full((N,), 1 / N, device="cuda")
    grad_s = torch.zeros_like(s)

    def fwd(block, warps, stages):
        kl_fwd_kernel[(N,)](s, t, kl, lse_s, lse_t, s.stride(0), t.stride(0), V,
                            BLOCK_V=block, num_warps=warps, num_stages=stages)

    def bwd(block, warps, stages):
        kl_bwd_kernel[(N,)](s, t, lse_s, lse_t, kl, g, grad_s, s.stride(0), t.stride(0), grad_s.stride(0), V,
                            BLOCK_V=block, num_warps=warps, num_stages=stages)

    # Minimum bytes each kernel must move. fwd: one pass over s and t. bwd: read s and t, write grad_s.
    kernels = {"fwd": (fwd, 2 * N * V * 2), "bwd": (bwd, 3 * N * V * 2)}
    fwd(4096, 4, 2)  # lse/kl inputs for bwd

    for name, (fn, nbytes) in kernels.items():
        floor_ms = nbytes / (H100_TBPS * 1e12) * 1e3
        log(f"=== {name}: {nbytes / 1e9:.2f} GB per 1024-row chunk, bandwidth floor {floor_ms:.3f} ms")
        results = []
        for block, warps, stages in itertools.product(BLOCKS, WARPS, STAGES):
            try:
                ms = do_bench(lambda: fn(block, warps, stages), warmup=10, rep=50)
            except Exception as e:  # e.g. out of shared memory / registers
                log(f"BLOCK_V={block:5d} warps={warps:2d} stages={stages}  failed: {type(e).__name__}")
                continue
            results.append((ms, block, warps, stages))
            log(f"BLOCK_V={block:5d} warps={warps:2d} stages={stages}  {ms:.3f} ms  {nbytes / ms / 1e6:6.0f} GB/s  "
                f"{100 * floor_ms / ms:4.0f}% of peak")
        results.sort()
        base = next(ms for ms, b, w, st in results if (b, w, st) == (4096, 4, 2)) if any(
            (b, w, st) == (4096, 4, 2) for _, b, w, st in results) else None
        log(f"--- {name} best 5 (current default BLOCK_V=4096, warps=4, stages=default):")
        for ms, block, warps, stages in results[:5]:
            log(f"    BLOCK_V={block:5d} warps={warps:2d} stages={stages}  {ms:.3f} ms  {100 * floor_ms / ms:4.0f}% of peak")
        if base:
            log(f"    default-ish (4096, 4, 2): {base:.3f} ms, {100 * floor_ms / base:.0f}% of peak")
    log("TUNE DONE")


@app.local_entrypoint()
def main():
    log("launching tune() on a Modal H100")
    tune.remote()
    log("done")

"""Check the prime-rl Qwen2 patch on a Modal GPU.

Run from the glint repo root, in the capstan workspace:
    MODAL_PROFILE=capstan uv run modal run cloud/qwen2_parity.py > logs/qwen2_parity.log 2>&1
"""

import os
import subprocess
import time

import modal
from images import PY, ROOT, prime_rl_image

image = (
    prime_rl_image.add_local_python_source("images")
    .add_local_dir(ROOT / "glint", "/glint/glint")
    .add_local_file(ROOT / "cloud" / "qwen2_real_weights.py", "/glint/qwen2_real_weights.py")
)

app = modal.App("glint-qwen2-parity", image=image)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@app.function(gpu="A10", timeout=30 * 60)
def check():
    log("=== container started; GPU:")
    subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"], check=True)

    log("=== step 1/2: pytest tests/unit/train/models/test_qwen2.py (tiny random Qwen2, HF vs prime-rl)")
    t = time.time()
    tests = subprocess.run([PY, "-m", "pytest", "-v", "tests/unit/train/models/test_qwen2.py"], cwd="/prime-rl")
    log(f"=== step 1/2 done in {time.time() - t:.0f}s, exit code {tests.returncode}")

    log("=== step 2/2: real Qwen2.5-0.5B-Instruct weights, HF vs prime-rl")
    t = time.time()
    real = subprocess.run([PY, "/glint/qwen2_real_weights.py"], cwd="/glint", env={**os.environ, "PYTHONPATH": "/glint"})
    log(f"=== step 2/2 done in {time.time() - t:.0f}s, exit code {real.returncode}")

    assert tests.returncode == 0, "pytest failed"
    assert real.returncode == 0, "real-weights check failed"
    log("=== ALL CHECKS PASSED")


@app.local_entrypoint()
def main():
    log("launching check() on Modal (image build happens first if anything changed)")
    check.remote()
    log("done")

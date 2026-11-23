"""Check the prime-rl Qwen2 patch on a Modal GPU.

Run from the glint repo root, in the capstan workspace:
    MODAL_PROFILE=capstan uv run modal run cloud/qwen2_parity.py > logs/qwen2_parity.log 2>&1
"""

import os
import subprocess
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PRIME_RL = ROOT / "third_party" / "prime-rl"

image = (
    modal.Image.from_registry("nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git")
    .pip_install("uv")
    .env(
        {
            # The image has no .git, so hatch-vcs can't derive these. Values from `hatch version` locally.
            # ponytail: hard-coded, refresh when the deps/ submodules move.
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_RENDERERS": "0.2.1.dev13",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VERIFIERS": "0.4.2.dev65",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PRIME_PYDANTIC_CONFIG": "0.4.3",
            "UV_LINK_MODE": "copy",
        }
    )
    .add_local_dir(PRIME_RL, "/prime-rl", copy=True, ignore=["**/.git", "**/.venv", "**/__pycache__", "**/.DS_Store"])
    # --frozen installs straight from uv.lock. --locked re-validates the lock against every index,
    # including flashinfer.ai, whose intermittent 503s killed two builds.
    .run_commands("cd /prime-rl && uv sync --frozen --extra gpu --extra flash-attn --group dev")
    .add_local_dir(ROOT / "glint", "/glint/glint")
    .add_local_file(ROOT / "cloud" / "qwen2_real_weights.py", "/glint/qwen2_real_weights.py")
)

app = modal.App("glint-qwen2-parity", image=image)
PY = "/prime-rl/.venv/bin/python"


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

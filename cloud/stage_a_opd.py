import os
import subprocess
import time
import urllib.request

import modal
from images import ROOT, prime_rl_image

CONFIG = "configs/stage_a_opd.toml"
TEACHER = "Qwen/Qwen2.5-7B-Instruct"
VENV = "/prime-rl/.venv/bin"

image = (
    prime_rl_image.env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_python_source("images")
    .add_local_file(ROOT / CONFIG, f"/glint/{CONFIG}")
)
hf_cache = modal.Volume.from_name("glint-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("glint-runs", create_if_missing=True)
app = modal.App("glint-stage-a-opd", image=image)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())} UTC] {msg}", flush=True)


def tail(path, n=40):
    if os.path.exists(path):
        log(f"--- last {n} lines of {path}")
        print("".join(open(path).readlines()[-n:]), flush=True)


def wait_for_teacher(proc, log_path, timeout=20 * 60):
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            tail(log_path, 80)
            raise RuntimeError(f"teacher exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen("http://localhost:8001/v1/models", timeout=5)
            log(f"teacher is up after {time.time() - start:.0f}s")
            return
        except OSError:
            log(f"waiting for teacher ({time.time() - start:.0f}s)")
            time.sleep(15)
    raise TimeoutError("teacher did not come up")


@app.function(
    # 2 GPUs queue faster than 3. GPU 0 holds the teacher and the student's vLLM side by side.
    # Modal takes the first type that's free. The trainer is pinned to FlashAttention-2 below,
    # since on Hopper prime-rl would auto-pick FA3, which this image doesn't install.
    gpu=["H100:2", "A100-80GB:2", "L40S:2"],
    timeout=2 * 60 * 60,
    volumes={"/cache": hf_cache, "/runs": runs},
    secrets=[modal.Secret.from_dotenv(ROOT)],  # WANDB_API_KEY
)
def train():
    run_dir = f"/runs/stage_a_opd/{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(run_dir, exist_ok=True)
    log(f"run dir: {run_dir}  (Modal volume glint-runs)")
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], check=True)
    # prime-rl spawns helpers (vllm-router, torchrun) by bare name, as if under `uv run`.
    venv_env = {**os.environ, "PATH": f"{VENV}:{os.environ['PATH']}", "VIRTUAL_ENV": "/prime-rl/.venv"}

    log(f"=== step 1/3: starting frozen teacher {TEACHER} on GPU 0 (45% of memory), port 8001 (log: {run_dir}/teacher.log)")
    teacher_log = f"{run_dir}/teacher.log"
    teacher = subprocess.Popen(
        [f"{VENV}/inference", "--vllm.model", TEACHER, "--server.port", "8001",
         "--vllm.gpu-memory-utilization", "0.45", "--vllm.enforce-eager"],
        cwd="/prime-rl",
        env={**venv_env, "CUDA_VISIBLE_DEVICES": "0"},
        stdout=open(teacher_log, "w"),
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_teacher(teacher, teacher_log)
        hf_cache.commit()

        log("=== step 2/3: rl (student inference on GPU 0 next to the teacher, 40% of memory; trainer on GPU 1)")
        log("prime-rl writes per-process logs to <run>/logs/{inference,trainer,orchestrator}.log")
        t = time.time()
        rl = subprocess.run(
            [f"{VENV}/rl", "@", f"/glint/{CONFIG}", "--output-dir", run_dir,
             "--inference.vllm.gpu-memory-utilization", "0.4",
             "--trainer.model.attn", "flash_attention_2",
             "--no-trainer.monitors.prime", "--no-orchestrator.monitors.prime"],
            cwd="/prime-rl",
            env={**venv_env, "CUDA_VISIBLE_DEVICES": "0,1"},
        )
        log(f"=== step 2/3 done in {(time.time() - t) / 60:.1f} min, exit code {rl.returncode}")
    finally:
        teacher.terminate()
        log("=== step 3/3: collecting logs")
        for name in ("orchestrator", "trainer", "inference"):
            for root, _, files in os.walk(run_dir):
                if f"{name}.log" in files:
                    tail(os.path.join(root, f"{name}.log"))
        tail(teacher_log, 20)
        runs.commit()
        hf_cache.commit()
    assert rl.returncode == 0, "rl failed"
    log("STAGE A: DONE")


@app.local_entrypoint()
def main():
    log("launching Stage A on Modal (2 GPUs: H100, A100-80GB or L40S, whichever is free)")
    train.remote()
    log("finished")

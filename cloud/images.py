"""Modal image with the prime-rl fork installed (shared by the cloud/ jobs)."""

from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
PRIME_RL = ROOT / "third_party" / "prime-rl"

prime_rl_image = (
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
    .run_commands("cd /prime-rl && uv sync --frozen --extra gpu --extra flash-attn --extra disagg --group dev --package prime-rl --package reverse-text")
)

PY = "/prime-rl/.venv/bin/python"

# glint

A glint is a small beam.

Glint is a small-scale study of **multi-teacher on-policy distillation**: merging two specialist teachers into one smaller student, where the student writes its own rollouts and the teachers score every token. It runs on one or two GPUs so anyone can rerun it.

- Teachers: Qwen2.5-Coder-7B-Instruct (code), Qwen2.5-7B-Instruct (general)
- Student: Qwen2.5-1.5B-Instruct
- Loss: reverse KL over the full vocabulary, computed with a memory-efficient kernel

This is a set of experiments and a write-up, not a library. Work in progress.

## Layout

```
glint/          reference implementations (float64 reverse KL)
cloud/          Modal jobs
third_party/    prime-rl fork (adds Qwen2 / Qwen2.5 support)
```

## Setup

```bash
git clone --recurse-submodules https://github.com/Vishal-Padia/glint.git
cd glint
uv sync
uv run python -m glint.kl_reference   # self-checks for the reference KL
```

GPU jobs run on [Modal](https://modal.com):

```bash
uv run modal run cloud/qwen2_parity.py
```

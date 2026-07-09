# Installation Guide (First-Time Setup)

This guide is for a fresh clone of Hematopathology LLM.

## Prerequisites

- Linux/macOS shell
- Python 3.12 or later
- `uv` installed and available on PATH

If `uv` is missing, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1) Clone and enter the repo

```bash
git clone https://github.com/cooperlab/heme-llm.git
cd heme-llm
```

## 2) Bootstrap all backend environments

Run:

```bash
make bootstrap
```

What this does:

- Regenerates backend lock files under `requirements/<backend>/lock.txt`
- Creates backend virtual environments under `.venvs/`
- Installs the project in editable mode into each backend environment

Expected environments:

- `.venvs/heme-llm-hf`
- `.venvs/heme-llm-vllm`
- `.venvs/heme-llm-sglang`
- `.venvs/heme-llm-ollama`

## 3) Run a quick HF pipeline smoke test

Use a single GPU on a multi-GPU server:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/heme-llm-hf/bin/python scripts/run_pipeline.py --config configs/run_pipeline.yaml
```

## Useful Make Targets

- `make help` - list available targets
- `make lockfiles` - regenerate all backend lock files
- `make envs` - recreate all backend environments
- `make env-hf` - recreate only HF environment
- `make clean-envs` - remove `.venvs`

## Troubleshooting

If you see import errors like `No module named pathology_llm`, recreate the target environment:

```bash
make env-hf
```

If CUDA/Torch issues occur in HF, rerun:

```bash
make env-hf
```

The HF target pins the CUDA 12.4-compatible torch stack by default.

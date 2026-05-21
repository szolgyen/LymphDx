# Installation Guide (First-Time Setup)

This guide is for a fresh clone of ReportLLM.

## Prerequisites

- Linux/macOS shell
- Python 3.11
- `uv` installed and available on PATH

If `uv` is missing, install it first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## 1) Clone and enter the repo

```bash
git clone <your-repo-url>
cd ReportLLM
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

- `.venvs/report-llm-hf`
- `.venvs/report-llm-vllm`
- `.venvs/report-llm-sglang`
- `.venvs/report-llm-ollama`

## 3) Run a quick HF pipeline smoke test

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/report-llm-hf/bin/python scripts/run_pipeline.py \
  --backend hf \
  --model google/medgemma-4b-it \
  --decoder guidance \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
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

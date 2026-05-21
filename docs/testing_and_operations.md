# Testing and Operations

## Test Suite

Run all tests:

```bash
uv run python -m pytest -q
```

Current coverage includes:

- Schema validation behavior.
- HF adapter parsing and decoder behavior.
- Decoder factory placeholder routing.

## Common Runtime Commands

Dummy backend:

```bash
uv run python scripts/run_pipeline.py \
  --backend dummy \
  --decoder none \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
```

HF + Guidance:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/report-llm-hf/bin/python scripts/run_pipeline.py \
  --backend hf \
  --model google/medgemma-4b-it \
  --decoder guidance \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
```

HF + Outlines (single GPU):

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/report-llm-hf/bin/python scripts/run_pipeline.py \
  --backend hf \
  --model google/medgemma-4b-it \
  --decoder outlines \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
```

## Observed HF Model/Decoder Matrix

Observed from local runs in this repository (May 2026):

| Model | `guidance` | `outlines` | Notes |
|---|---|---|---|
| `google/medgemma-4b-it` | Works | Works | Successful end-to-end extraction run observed. |
| `google/medgemma-1.5-4b-it` | Works | Works | `decoder=none` failed (non-JSON outputs), constrained decoders worked. |
| `aaditya/Llama3-OpenBioLLM-8B` | Works | Works | Successful constrained extraction observed for both decoders. |

Notes:

- This is an observed compatibility snapshot, not a strict support matrix.
- Results can vary by GPU topology, CUDA driver, and dependency versions.
- For HF backend runs, pinning to a single GPU (`CUDA_VISIBLE_DEVICES=0`) is the default documented path.

## Logging

- Console + file logging are enabled.
- Per-run log files: `outputs/logs/run_pipeline_YYYYMMDD_HHMMSS.log`
- Guidance mode logs start/finish timing per report.

## Environment Reproducibility

Tracked dependency sources:

- `pyproject.toml`
- `requirements/base.txt`
- `requirements/hf/requirements.txt`
- `requirements/vllm/requirements.txt`
- `requirements/sglang/requirements.txt`
- `requirements/ollama/requirements.txt`
- `uv.lock`
- `requirements/hf/lock.txt`
- `requirements/vllm/lock.txt`
- `requirements/sglang/lock.txt`
- `requirements/ollama/lock.txt`

Locking model:

- `uv.lock` is the single lock file for dependencies declared in `pyproject.toml`.
- Backend-specific environments use backend requirements files plus backend lock files.
- Regenerate backend locks with `uv pip compile requirements/<backend>/requirements.txt -o requirements/<backend>/lock.txt`.

Notes:

- CUDA-compatible Torch is pinned for the current environment constraints.
- `gpustat` is tracked for optional Guidance GPU monitoring metrics.

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
uv run python scripts/run_pipeline.py \
  --backend hf \
  --model google/medgemma-4b-it \
  --decoder guidance \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
```

## Logging

- Console + file logging are enabled.
- Per-run log files: `outputs/logs/run_pipeline_YYYYMMDD_HHMMSS.log`
- Guidance mode logs start/finish timing per report.

## Environment Reproducibility

Tracked dependency sources:

- `pyproject.toml`
- `requirements/base.txt`
- `requirements/hf.txt`
- `uv.lock`

Notes:

- CUDA-compatible Torch is pinned for the current environment constraints.
- `gpustat` is tracked for optional Guidance GPU monitoring metrics.

# ReportLLM

Structured pathology report extraction with modular LLM backends.

## Quick Start

For first-time setup from a fresh clone, use the installation guide:

- [docs/installation.md](docs/installation.md)

Shortest path:

```bash
make bootstrap
```

Then run the HF pipeline:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/report-llm-hf/bin/python scripts/run_pipeline.py \
  --backend hf \
  --model google/medgemma-4b-it \
  --decoder guidance \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
```
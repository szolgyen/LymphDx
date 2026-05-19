# ReportLLM Documentation

This folder contains structured documentation for the current framework.

## Documents

- `architecture_overview.md`
  - High-level architecture, module boundaries, and design principles.
- `pipeline_and_cli.md`
  - End-to-end execution flow from CLI to output files.
- `inference_stack.md`
  - Adapter/decoder architecture, backend support, and constrained decoding behavior.
- `testing_and_operations.md`
  - Test strategy, runtime commands, logging, and environment reproducibility.
- `status_and_roadmap.md`
  - Current implementation status and prioritized next steps.

## Fast Start

Run the current HF guidance path:

```bash
uv run python scripts/run_pipeline.py \
  --backend hf \
  --model google/medgemma-4b-it \
  --decoder guidance \
  --input-file configs/extraction/sample_reports.txt \
  --diagnosis-terms-file configs/extraction/diagnosis_terms_v1.txt \
  --log-level INFO
```

Core outputs:

- `outputs/predictions/`
- `outputs/logs/run_pipeline_*.log`

## Current State (Short)

- End-to-end pipeline is working.
- HF backend is implemented and supports:
  - `decoder=none` (best-effort JSON prompting)
  - `decoder=guidance` (strict schema-guided generation path)
- Decoder placeholders exist for `outlines` and `sglang`.

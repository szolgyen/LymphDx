# Hematopathology LLM Documentation

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

Edit `configs/pipeline/run_pipeline.yaml` with your settings, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/heme-llm-hf/bin/python scripts/run_pipeline.py --config configs/pipeline/run_pipeline.yaml
```

Core outputs:

- `outputs/YYYYMMDD_HHMMSS/`
- `outputs/predictions_YYYYMMDD_HHMMSS.jsonl`
- `outputs/predictions_broken_YYYYMMDD_HHMMSS.jsonl`
- `outputs/YYYYMMDD_HHMMSS/run_pipeline_YYYYMMDD_HHMMSS.log`

## Current State (Short)

- End-to-end pipeline is working.
- HF backend is implemented and supports:
  - `decoder=none` (best-effort JSON prompting)
  - `decoder=guidance` (strict schema-guided generation path)
  - `decoder=outlines` (strict constrained decoding path)
- Decoder status:
  - `none`, `guidance`, and `outlines` are implemented.
  - `sglang` decoder remains a placeholder.
- Backend status:
  - `hf` is fully implemented.
  - `vllm`, `sglang`, and `ollama` adapters are placeholders in the current codebase.

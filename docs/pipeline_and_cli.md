# Pipeline and CLI Flow

## Configuration

Execution is configured via YAML files in `configs/`.

Default config: `configs/run_pipeline.yaml`

Main configuration keys:

- `backend`: ✅ `hf` | ❌ `vllm` | ❌ `sglang` | ❌ `ollama` (required)
- `model`: 
   - google/medgemma-4b-it
   - google/medgemma-1.5-4b-it
   - aaditya/Llama3-OpenBioLLM-8B
- `decoder`: `none | guidance | outlines` (required)
- `schema`: schema version (e.g., `v5`)
- `input_file`: reports file (.xlsx)
- `diagnosis_terms_file`: allowed diagnosis terms file
   - default: `inputs/LN_Dx_dictionary_codes_20260824.xlsx`
- `output_dir`: JSON output directory
- `log_level`: `DEBUG` | `INFO` | `WARNING` | `ERROR` | `CRITICAL`

## CLI Entrypoint

`scripts/run_pipeline.py` loads config from a YAML file.

Run the pipeline:

```sh
.venvs/heme-llm-hf/bin/python scripts/run_pipeline.py
```

See [installation.md](installation.md) for first-time setup.

> In the current codebase, non-HF adapters (`vllm`, `sglang`, `ollama`) are placeholders and not production runtime paths yet.

## Runtime Sequence

1. Configure logging and output log file.
2. Load diagnosis terms and reports.
3. Create adapter via `create_adapter`.
4. Build `ExtractionPipeline`.
5. For each report:
   - Build prompt with diagnosis constraints.
   - Run adapter extraction.
   - Parse and validate schema.
6. Write outputs to `outputs/predictions`.
7. Fail if all reports fail extraction.

## Error Behavior

- Per-report extraction errors are logged and pipeline continues.
- If all reports fail, CLI exits non-zero.
- Validation failures raise explicit schema/constraint exceptions.

## Output Artifacts

- Timestamped output directory: `outputs/YYYYMMDD_HHMMSS/`
- Timestamped predictions: `outputs/YYYYMMDD_HHMMSS/predictions_YYYYMMDD_HHMMSS.jsonl`
- Broken extraction records: `outputs/YYYYMMDD_HHMMSS/predictions_broken_YYYYMMDD_HHMMSS.jsonl`
- Execution logs: `outputs/YYYYMMDD_HHMMSS/run_pipeline_YYYYMMDD_HHMMSS.log`
- Individual reports: `outputs/YYYYMMDD_HHMMSS/reports`

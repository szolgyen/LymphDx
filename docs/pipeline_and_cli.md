# Pipeline and CLI Flow

## Configuration

Execution is configured via YAML files in `configs/pipeline/`.

Default config: `configs/pipeline/run_pipeline.yaml`

Main configuration keys:

- `backend`: `hf | vllm | sglang | ollama` (required)
- `model`: model identifier for selected backend (required)
- `decoder`: `none | guidance | outlines` (required)
- `schema`: schema version (e.g., `v5`)
- `input_file`: reports file (.txt one-per-line or .xlsx)
- `diagnosis_terms_file`: allowed diagnosis terms file
- `output_dir`: JSON output directory
- `log_level`: `DEBUG | INFO | WARNING | ERROR | CRITICAL`

## CLI Entrypoint

`scripts/run_pipeline.py` loads config from a YAML file.

Run the pipeline:

```sh
.venvs/heme-llm-hf/bin/python scripts/run_pipeline.py --config configs/pipeline/run_pipeline.yaml
```

Optional override (uses YAML default for unspecified options):

```sh
.venvs/heme-llm-hf/bin/python scripts/run_pipeline.py --config configs/pipeline/run_pipeline.yaml --log-level DEBUG
```

See [installation.md](installation.md) for first-time setup.

Note:

- In the current codebase, non-HF adapters (`vllm`, `sglang`, `ollama`) are placeholders and not production runtime paths yet.

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

- Timestamped output directory: `outputs/predictions/YYYYMMDD_HHMMSS/`
- Timestamped predictions: `predictions_YYYYMMDD_HHMMSS.jsonl`
- Broken extraction records: `predictions_broken_YYYYMMDD_HHMMSS.jsonl`
- Execution logs: `outputs/logs/run_pipeline_YYYYMMDD_HHMMSS.log`

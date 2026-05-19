# Pipeline and CLI Flow

## CLI Entrypoint

`scripts/run_pipeline.py` controls execution.

Main arguments:

- `--backend`: `dummy | hf | vllm | sglang | ollama`
- `--model`: model identifier for selected backend
- `--decoder`: `auto | none | sglang | guidance | outlines`
- `--input-file`: reports text file (one report per line)
- `--prompt-template`: prompt template path
- `--diagnosis-terms-file`: allowed diagnosis terms
- `--output-dir`: JSON output directory
- `--log-level`: `DEBUG | INFO | WARNING | ERROR | CRITICAL`

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

- Per-case JSON files under `outputs/predictions/`
- Aggregate `predictions.jsonl`
- Execution logs under `outputs/logs/`

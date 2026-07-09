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


## Fast Start

Edit `configs/run_pipeline.yaml` with your settings, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/heme-llm-hf/bin/python scripts/run_pipeline.py
```

Core outputs:

- `outputs/YYYYMMDD_HHMMSS/`
- `outputs/YYYYMMDD_HHMMSS/reports/case_*.json`
- `outputs/YYYYMMDD_HHMMSS/reports/case_*.txt`
- `outputs/predictions_YYYYMMDD_HHMMSS.jsonl`
- `outputs/predictions_broken_YYYYMMDD_HHMMSS.jsonl`
- `outputs/YYYYMMDD_HHMMSS/run_pipeline_YYYYMMDD_HHMMSS.log`

## Ontology mapping

In `decoder=none` mode, the LLM extracts unconstrained diagnoses.

Edit `configs/ontology.yaml` with the paths of the LLM output and the dictionary of the ontology terms, then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/heme-llm-hf/bin/python scripts/ontology.py
```

This generates an ontology mapped JSON file with the valid diagnostic terms.

## Export results in an Excel file

Edit `configs/excel_merge.yaml` with the paths of the ground truth file (input_excel) and the ontology mapped JSON file, then run:

```bash
.venvs/heme-llm-hf/bin/python scripts/merge_predictions_excel.py
```

## Evaluate performance

Edit `configs/evaluation.yaml` with your settings, then run:

```bash
.venvs/heme-llm-hf/bin/python scripts/evaluate.py
```

This generates metrics about the LLM's performance by comparing predictions to the ground truth. The evaluation generates the folowing files:

- `accuracy_coverage_plot.png`
- `threshold_accuracy_coverage.png`
- `evaluation_summary.json`
- `error_analysis.json`
- `hard_case_metrics.csv`
- `rare_diagnosis_metrics.csv`
- `report_level_metrics.csv`
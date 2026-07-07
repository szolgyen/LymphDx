# Validation Template Merge

This document describes the validation template merge functionality that fills an Excel template with prediction data from JSONL files.

## Overview

The validation template merge process takes prediction data from a JSONL file (guidance predictions) and populates an Excel template with case-level and container-level predictions. The template structure supports:

- **Case-level data**: case_id, primary_diagnosis, prediction flags (is_definitive, has_differential_diagnosis, etc.)
- **Container-level data**: container labels, is_lymph_node status, container diagnosis
- **Validation matching**: GT (ground truth) vs Predicted columns with match highlighting and helper columns

## Template Structure

The validation template has a Results sheet with 27 columns:

1. **Id** - Case ID from JSONL
2. **Container Duplicate** - Boolean: False for first container per case, True for subsequent containers
3. **Report** - Empty (reserved for ground truth data)
4. **GT Report Diagnosis** - Empty (reserved for ground truth)
5. **Predicted Report Diagnosis** - Filled from JSONL `primary_diagnosis`
6. **Predicted Report Diagnosis Match** - Helper column, populated with "literal_match" if GT matches Predicted
7. **GT Container** - Empty (reserved for ground truth)
8. **Predicted Container** - Container label from JSONL
9. **Predicted Container Match** - Helper column for GT/Predicted matching
10. **GT Has Differential Diagnosis** - Empty
11. **Predicted Has Differential Diagnosis** - From JSONL `has_differential_diagnosis`
12. **Predicted Has Differential Diagnosis Match** - Helper column
13. **GT Is Lymph Node** - Empty
14. **Predicted Is Lymph Node** - From container `is_lymph_node` field
15. **Predicted Is Lymph Node Match** - Helper column
16. **GT Is Definitive** - Empty
17. **Predicted Is Definitive** - From JSONL `is_definitive`
18. **Predicted Is Definitive Match** - Helper column
19. **GT Has Prior Malignancy** - Empty
20. **Predicted Has Prior Malignancy** - From JSONL `has_prior_malignancy`
21. **Predicted Has Prior Malignancy Match** - Helper column
22. **GT Has Concurrent Malignancy** - Empty
23. **Predicted Has Concurrent Malignancy** - From JSONL `has_concurrent_malignancy`
24. **Predicted Has Concurrent Malignancy Match** - Helper column
25. **GT Container Diagnosis** - Empty
26. **Predicted Container Diagnosis** - From container `diagnosis` field

## Usage

### Command Line

Run the merge script with default parameters:
```bash
python scripts/merge_predictions_validation_template.py
```

Or specify custom paths:
```bash
python scripts/merge_predictions_validation_template.py \
  --template-excel <path_to_template.xlsx> \
  --predictions-jsonl <path_to_predictions.jsonl> \
  --output-excel <path_to_output.xlsx>
```


## Data Mapping

### Row Generation

One row is generated per container within each case:
- If a case has 4 containers, 4 rows will be created
- Container Duplicate is False for the first container, True for subsequent ones

### Field Mapping

| Excel Column | JSONL Source | Notes |
|---|---|---|
| Id | case_id | Normalized using normalize_case_id() |
| Container Duplicate | Container index | False for idx=0, True for idx>0 |
| Predicted Report Diagnosis | primary_diagnosis | Case-level field |
| Predicted Container | containers[].label | Container label (A, B, C, etc.) |
| Predicted Has Differential Diagnosis | has_differential_diagnosis | Case-level boolean |
| Predicted Is Lymph Node | containers[].is_lymph_node | Container-level boolean |
| Predicted Is Definitive | is_definitive | Case-level boolean |
| Predicted Has Prior Malignancy | has_prior_malignancy | Case-level boolean |
| Predicted Has Concurrent Malignancy | has_concurrent_malignancy | Case-level boolean |
| Predicted Container Diagnosis | containers[].diagnosis | Container-level text field |

## Match Highlighting

When GT columns are populated with validation data:

1. Cell highlighting (light green `#C6EFCE`) is applied to Predicted cells where the value matches the corresponding GT cell (case-insensitive string comparison)
2. Match helper columns are populated with "literal_match" string where highlighting is applied
3. This allows quick visual scanning of correct vs incorrect predictions

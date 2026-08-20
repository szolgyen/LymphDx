"""DataFrame builders for constructing evaluation-ready datasets.

Combines ground-truth and prediction data into structured DataFrames for analysis.
"""

import logging
from typing import Any

import pandas as pd

from . import prediction_extractors as extractors

logger = logging.getLogger(__name__)


def build_report_level_dataframe(
    ground_truth_df: pd.DataFrame,
    prediction_map: dict[int, dict[str, Any]],
    top_k_values: list[int],
) -> pd.DataFrame:
    """Build a report-level DataFrame combining ground-truth and prediction data.

    Creates one row per case by joining ground-truth annotations with model predictions,
    extracting top-k diagnosis information and accuracy metrics.

    Args:
        ground_truth_df: DataFrame with ground-truth annotations.
        prediction_map: Lookup map from case_id to prediction record.
        top_k_values: List of k values to evaluate (e.g., [1, 3, 5]).

    Returns:
        Report-level DataFrame with combined ground-truth and prediction columns.
    """
    rows: list[dict[str, Any]] = []

    for case_id in ground_truth_df["Id"].dropna().unique():
        record = prediction_map.get(int(case_id))
        if record is None:
            continue

        gt_row = ground_truth_df.loc[ground_truth_df["Id"] == case_id].iloc[0]

        # Extract top-1, top-3, and top-5 predictions
        top_1 = extractors.extract_top_1_primary_diagnosis(record)
        if top_1 is None:
            continue

        top_3_groups = _extract_top_k_groups_dict(record, k=3)
        if top_3_groups is None:
            continue

        top_5_groups = _extract_top_k_groups_dict(record, k=5)
        if top_5_groups is None:
            continue

        row = _build_report_row(
            case_id=case_id,
            gt_row=gt_row,
            record=record,
            top_1=top_1,
            top_3_groups=top_3_groups,
            top_5_groups=top_5_groups,
            top_k_values=top_k_values,
        )
        rows.append(row)

    return pd.DataFrame(rows)


def _extract_top_k_groups_dict(record: dict[str, Any], k: int) -> dict[str, Any] | None:
    """Helper to extract all group categories for top-k predictions.

    Args:
        record: Prediction record.
        k: Number of top predictions.

    Returns:
        Dictionary with keys like "valid_primary_diagnosis_group_1", or None if any group is missing.
    """
    groups = {}
    for group_idx in range(1, 4):
        group_key = f"group_{group_idx}"
        group_values = extractors.extract_top_k_primary_diagnosis_groups(
            record, k, group_key
        )
        if group_values is None:
            return None
        groups[f"valid_primary_diagnosis_group_{group_idx}"] = group_values
    return groups


def _build_report_row(
    case_id: Any,
    gt_row: pd.Series,
    record: dict[str, Any],
    top_1: dict[str, Any],
    top_3_groups: dict[str, Any],
    top_5_groups: dict[str, Any],
    top_k_values: list[int],
) -> dict[str, Any]:
    """Build a single report row combining ground-truth and prediction data.

    Args:
        case_id: Case identifier.
        gt_row: Ground-truth row from Excel.
        record: Prediction record.
        top_1: Top-1 diagnosis dictionary.
        top_3_groups: Top-3 group dictionaries.
        top_5_groups: Top-5 group dictionaries.
        top_k_values: List of k values to evaluate.

    Returns:
        Dictionary representing a single report-level row.
    """
    # Extract ground-truth fields
    gt_code = gt_row["GT Code"]
    gt_group1 = gt_row["GT Report Diagnosis Group 1"]
    gt_group2 = gt_row["GT Report Diagnosis Group 2"]
    gt_group3 = gt_row["GT Report Diagnosis Group 3"]

    # Extract top-k diagnosis codes for accuracy checks
    topk_codes = {
        k: extractors.extract_top_k_primary_diagnosis_codes(record, k)
        for k in range(1, max(top_k_values) + 1)
    }

    # Extract top-1 predictions
    pred_group1 = top_1.get("valid_primary_diagnosis_group_1")
    pred_group2 = top_1.get("valid_primary_diagnosis_group_2")
    pred_group3 = top_1.get("valid_primary_diagnosis_group_3")

    # Extract top-3 and top-5 group predictions
    top3_group1 = top_3_groups.get("valid_primary_diagnosis_group_1")
    top3_group2 = top_3_groups.get("valid_primary_diagnosis_group_2")
    top3_group3 = top_3_groups.get("valid_primary_diagnosis_group_3")

    top5_group1 = top_5_groups.get("valid_primary_diagnosis_group_1")
    top5_group2 = top_5_groups.get("valid_primary_diagnosis_group_2")
    top5_group3 = top_5_groups.get("valid_primary_diagnosis_group_3")

    # Extract boolean condition fields
    gt_has_differential = extractors.extract_boolean_field(
        gt_row["GT Has Differential Diagnosis"]
    )
    gt_is_definitive = extractors.extract_boolean_field(gt_row["GT Is Definitive"])
    gt_has_prior_malignancy = extractors.extract_boolean_field(
        gt_row["GT Has Prior Malignancy"]
    )
    gt_has_concurrent_malignancy = extractors.extract_boolean_field(
        gt_row["GT Has Concurrent Malignancy"]
    )

    pred_has_differential = extractors.extract_boolean_field(
        record.get("has_differential_diagnosis")
    )
    pred_is_definitive = extractors.extract_boolean_field(record.get("is_definitive"))
    pred_has_prior_malignancy = extractors.extract_boolean_field(
        record.get("has_prior_malignancy")
    )
    pred_has_concurrent_malignancy = extractors.extract_boolean_field(
        record.get("has_concurrent_malignancy")
    )

    return {
        # Case and code information
        "case_id": case_id,
        "gt_code": gt_code,
        "pred_code": top_1.get("valid_primary_diagnosis_code"),
        "pred_score": top_1.get("valid_primary_diagnosis_score"),
        # Ground-truth group classifications
        "gt_group1": gt_group1,
        "gt_group2": gt_group2,
        "gt_group3": gt_group3,
        # Predicted group classifications
        "pred_group1": pred_group1,
        "pred_group2": pred_group2,
        "pred_group3": pred_group3,
        # Ground-truth boolean conditions
        "gt_has_differential": gt_has_differential,
        "gt_is_definitive": gt_is_definitive,
        "gt_has_prior_malignancy": gt_has_prior_malignancy,
        "gt_has_concurrent_malignancy": gt_has_concurrent_malignancy,
        # Predicted boolean conditions
        "pred_has_differential": pred_has_differential,
        "pred_is_definitive": pred_is_definitive,
        "pred_has_prior_malignancy": pred_has_prior_malignancy,
        "pred_has_concurrent_malignancy": pred_has_concurrent_malignancy,
        # Top-k accuracy flags
        "top1_correct": gt_code in topk_codes[1],
        "top3_correct": gt_code in topk_codes[3],
        "top5_correct": gt_code in topk_codes[5],
        # Top-1 group accuracy flags
        "top1_group1_correct": gt_group1 == pred_group1,
        "top1_group2_correct": gt_group2 == pred_group2,
        "top1_group3_correct": gt_group3 == pred_group3,
        # Top-3 group accuracy flags
        "top3_group1_correct": gt_group1 in top3_group1 if top3_group1 else False,
        "top3_group2_correct": gt_group2 in top3_group2 if top3_group2 else False,
        "top3_group3_correct": gt_group3 in top3_group3 if top3_group3 else False,
        # Top-5 group accuracy flags
        "top5_group1_correct": gt_group1 in top5_group1 if top5_group1 else False,
        "top5_group2_correct": gt_group2 in top5_group2 if top5_group2 else False,
        "top5_group3_correct": gt_group3 in top5_group3 if top5_group3 else False,
        # Boolean condition accuracy flags
        "has_differential_correct": gt_has_differential == pred_has_differential,
        "is_definitive_correct": gt_is_definitive == pred_is_definitive,
        "has_prior_malignancy_correct": gt_has_prior_malignancy
        == pred_has_prior_malignancy,
        "has_concurrent_malignancy_correct": gt_has_concurrent_malignancy
        == pred_has_concurrent_malignancy,
    }

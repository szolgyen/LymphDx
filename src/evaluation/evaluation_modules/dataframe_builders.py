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

        # Extract container groups (may be None if no container predictions)
        top_3_container_groups = _extract_top_k_container_groups_dict(record, k=3)
        top_5_container_groups = _extract_top_k_container_groups_dict(record, k=5)

        row = _build_report_row(
            case_id=case_id,
            gt_row=gt_row,
            record=record,
            top_1=top_1,
            top_3_groups=top_3_groups,
            top_5_groups=top_5_groups,
            top_3_container_groups=top_3_container_groups,
            top_5_container_groups=top_5_container_groups,
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


def _extract_top_k_container_groups_dict(record: dict[str, Any], k: int) -> dict[str, Any] | None:
    """Helper to extract all group categories for top-k container predictions.

    Args:
        record: Prediction record.
        k: Number of top predictions.

    Returns:
        Dictionary with keys like "valid_diagnosis_group_1", or None if container has no predictions.
    """
    groups = {}
    for group_idx in range(1, 4):
        group_key = f"group_{group_idx}"
        group_values = extractors.extract_top_k_container_diagnosis_groups(
            record, k, group_key
        )
        if group_values is None:
            return None
        groups[f"valid_diagnosis_group_{group_idx}"] = group_values
    return groups


def _build_report_row(
    case_id: Any,
    gt_row: pd.Series,
    record: dict[str, Any],
    top_1: dict[str, Any],
    top_3_groups: dict[str, Any],
    top_5_groups: dict[str, Any],
    top_3_container_groups: dict[str, Any] | None,
    top_5_container_groups: dict[str, Any] | None,
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
        top_3_container_groups: Top-3 container group dictionaries (may be None).
        top_5_container_groups: Top-5 container group dictionaries (may be None).
        top_k_values: List of k values to evaluate.

    Returns:
        Dictionary representing a single report-level row.
    """
    # Extract ground-truth fields
    gt_code = gt_row["GT Report Diagnosis Code"]
    gt_name = gt_row["GT Report Diagnosis"].strip()
    gt_group1 = gt_row["GT Report WHO-like Subcategories"]
    gt_group2 = gt_row["GT Report WHO-like Categories"]
    gt_group3 = gt_row["GT Report WHO-like Major Sections/Lineages"]
    gt_group4 = gt_row["GT Report Diagnosis Group 4"]

    # Extract container ground-truth fields (may be NaN)
    gt_container_code = gt_row.get("GT Container Diagnosis Code")
    if pd.isna(gt_container_code):
        gt_container_code = None
    else:
        gt_container_code = int(gt_container_code)

    # Extract container ground-truth group fields (may be NaN)
    gt_container_group1 = gt_row.get("GT Container WHO-like Subcategories")
    gt_container_group2 = gt_row.get("GT Container WHO-like Categories")
    gt_container_group3 = gt_row.get("GT Container WHO-like Major Sections/Lineages")
    gt_container_group4 = gt_row.get("GT Container Diagnosis Group 4")
    if pd.isna(gt_container_group1):
        gt_container_group1 = None
    if pd.isna(gt_container_group2):
        gt_container_group2 = None
    if pd.isna(gt_container_group3):
        gt_container_group3 = None
    if pd.isna(gt_container_group4):
        gt_container_group4 = None

    # Extract top-k diagnosis codes for accuracy checks
    topk_codes = {
        k: extractors.extract_top_k_primary_diagnosis_codes(record, k)
        for k in range(1, max(top_k_values) + 1)
    }

    top_1_container = extractors.extract_top_1_container_diagnosis(record)
    pred_container_code = (
        top_1_container.get("valid_diagnosis_code")
        if top_1_container is not None
        else None
    )
    if pred_container_code is not None and not pd.isna(pred_container_code):
        pred_container_code = int(pred_container_code)
    else:
        pred_container_code = None

    # Extract container group predictions from top-1
    pred_container_group1 = (
        top_1_container.get("valid_diagnosis_group_1")
        if top_1_container is not None
        else None
    )
    pred_container_group2 = (
        top_1_container.get("valid_diagnosis_group_2")
        if top_1_container is not None
        else None
    )
    pred_container_group3 = (
        top_1_container.get("valid_diagnosis_group_3")
        if top_1_container is not None
        else None
    )
    pred_container_group4 = (
        top_1_container.get("valid_diagnosis_group_4")
        if top_1_container is not None
        else None
    )

    # Extract top-k container diagnosis codes for accuracy checks
    topk_container_codes = (
        {
            k: extractors.extract_top_k_container_diagnosis_codes(record, k)
            for k in range(1, max(top_k_values) + 1)
        }
        if gt_container_code is not None
        else {}
    )

    # Extract top-3 and top-5 container group predictions
    top3_container_group1 = (
        top_3_container_groups.get("valid_diagnosis_group_1")
        if top_3_container_groups
        else None
    )
    top3_container_group2 = (
        top_3_container_groups.get("valid_diagnosis_group_2")
        if top_3_container_groups
        else None
    )
    top3_container_group3 = (
        top_3_container_groups.get("valid_diagnosis_group_3")
        if top_3_container_groups
        else None
    )
    top3_container_group4 = (
        top_3_container_groups.get("valid_diagnosis_group_4")
        if top_3_container_groups
        else None
    )

    top5_container_group1 = (
        top_5_container_groups.get("valid_diagnosis_group_1")
        if top_5_container_groups
        else None
    )
    top5_container_group2 = (
        top_5_container_groups.get("valid_diagnosis_group_2")
        if top_5_container_groups
        else None
    )
    top5_container_group3 = (
        top_5_container_groups.get("valid_diagnosis_group_3")
        if top_5_container_groups
        else None
    )
    top5_container_group4 = (
        top_5_container_groups.get("valid_diagnosis_group_4")
        if top_5_container_groups
        else None
    )

    # Extract top-1 predictions
    pred_group1 = top_1.get("valid_primary_diagnosis_group_1")
    pred_group2 = top_1.get("valid_primary_diagnosis_group_2")
    pred_group3 = top_1.get("valid_primary_diagnosis_group_3")
    pred_group4 = top_1.get("valid_primary_diagnosis_group_4")

    # Extract top-3 and top-5 group predictions
    top3_group1 = top_3_groups.get("valid_primary_diagnosis_group_1")
    top3_group2 = top_3_groups.get("valid_primary_diagnosis_group_2")
    top3_group3 = top_3_groups.get("valid_primary_diagnosis_group_3")
    top3_group4 = top_3_groups.get("valid_primary_diagnosis_group_4")

    top5_group1 = top_5_groups.get("valid_primary_diagnosis_group_1")
    top5_group2 = top_5_groups.get("valid_primary_diagnosis_group_2")
    top5_group3 = top_5_groups.get("valid_primary_diagnosis_group_3")
    top5_group4 = top_5_groups.get("valid_primary_diagnosis_group_4")

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
        "gt_name": gt_name,
        "pred_code": top_1.get("valid_primary_diagnosis_code"),
        "pred_name": top_1.get("valid_primary_diagnosis_name"),
        "pred_container_code": pred_container_code,
        "pred_score": top_1.get("valid_primary_diagnosis_score"),
        # Ground-truth group classifications
        "gt_group0": gt_name,
        "gt_group1": gt_group1,
        "gt_group2": gt_group2,
        "gt_group3": gt_group3,
        "gt_group4": gt_group4,
        # Predicted group classifications
        "pred_group0": top_1.get("valid_primary_diagnosis_name"),
        "pred_group1": pred_group1,
        "pred_group2": pred_group2,
        "pred_group3": pred_group3,
        "pred_group4": pred_group4,
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
        "top1_group0_correct": gt_code in topk_codes[1],
        "top1_group1_correct": gt_group1 == pred_group1,
        "top1_group2_correct": gt_group2 == pred_group2,
        "top1_group3_correct": gt_group3 == pred_group3,
        "top1_group4_correct": gt_group4 == pred_group4,
        # Top-3 group accuracy flags
        "top3_group0_correct": gt_code in topk_codes[3],
        "top3_group1_correct": gt_group1 in top3_group1 if top3_group1 else False,
        "top3_group2_correct": gt_group2 in top3_group2 if top3_group2 else False,
        "top3_group3_correct": gt_group3 in top3_group3 if top3_group3 else False,
        "top3_group4_correct": gt_group4 in top3_group4 if top3_group4 else False,
        # Top-5 group accuracy flags
        "top5_group0_correct": gt_code in topk_codes[5],
        "top5_group1_correct": gt_group1 in top5_group1 if top5_group1 else False,
        "top5_group2_correct": gt_group2 in top5_group2 if top5_group2 else False,
        "top5_group3_correct": gt_group3 in top5_group3 if top5_group3 else False,
        "top5_group4_correct": gt_group4 in top5_group4 if top5_group4 else False,
        # Container ground-truth group classifications
        "gt_container_code": gt_container_code,
        "gt_container_group1": gt_container_group1,
        "gt_container_group2": gt_container_group2,
        "gt_container_group3": gt_container_group3,
        "gt_container_group4": gt_container_group4,
        # Container predicted group classifications
        "pred_container_group1": pred_container_group1,
        "pred_container_group2": pred_container_group2,
        "pred_container_group3": pred_container_group3,
        "pred_container_group4": pred_container_group4,
        # Container diagnosis accuracy flags (None if no GT container code)
        "container_top1_correct": gt_container_code in topk_container_codes.get(1, [])
        if gt_container_code is not None
        else None,
        "container_top3_correct": gt_container_code in topk_container_codes.get(3, [])
        if gt_container_code is not None
        else None,
        "container_top5_correct": gt_container_code in topk_container_codes.get(5, [])
        if gt_container_code is not None
        else None,
        # Boolean condition accuracy flags
        "has_differential_correct": gt_has_differential == pred_has_differential,
        "is_definitive_correct": gt_is_definitive == pred_is_definitive,
        "has_prior_malignancy_correct": gt_has_prior_malignancy
        == pred_has_prior_malignancy,
        "has_concurrent_malignancy_correct": gt_has_concurrent_malignancy
        == pred_has_concurrent_malignancy,
    }


def build_container_level_dataframe(
    ground_truth_df: pd.DataFrame,
    prediction_map: dict[int, dict[str, Any]],
    top_k_values: list[int],
) -> pd.DataFrame:
    """Build a container-level DataFrame from ground-truth data.

    Creates one row per container by iterating through all rows in ground_truth_df
    (which may have multiple rows per report ID) and joining with predictions.
    This captures all containers, not just the first one per report.

    Args:
        ground_truth_df: DataFrame with ground-truth annotations (one row per container).
        prediction_map: Lookup map from case_id to prediction record.
        top_k_values: List of k values to evaluate (e.g., [1, 3, 5]).

    Returns:
        Container-level DataFrame with one row per container from ground truth.
    """
    rows: list[dict[str, Any]] = []

    # Iterate through ALL rows in ground_truth_df, not just unique IDs
    # This ensures we capture all containers per report
    for idx, gt_row in ground_truth_df.iterrows():
        case_id = gt_row.get("Id")
        if pd.isna(case_id):
            continue

        # Check if this row has container ground-truth information
        gt_container_code = gt_row.get("GT Container Diagnosis Code")
        if pd.isna(gt_container_code):
            # Skip rows without container GT data
            continue

        gt_container_code_int = int(gt_container_code)

        record = prediction_map.get(int(case_id))
        if record is None:
            continue

        # Extract top-1 container predictions
        top_1_container = extractors.extract_top_1_container_diagnosis(record)
        
        # For containers, we need predictions. If top_1_container is None, skip
        if top_1_container is None:
            continue

        # Extract container predictions
        pred_container_code = (
            top_1_container.get("valid_diagnosis_code")
            if top_1_container is not None
            else None
        )
        if pred_container_code is not None and not pd.isna(pred_container_code):
            pred_container_code = int(pred_container_code)
        else:
            pred_container_code = None

        # Extract container diagnosis name from prediction
        pred_container_name = (
            top_1_container.get("valid_diagnosis_name")
            if top_1_container is not None
            else None
        )

        # Extract GT container diagnosis name from ground truth
        gt_container_name = gt_row.get("GT Container Diagnosis")
        if pd.isna(gt_container_name) or gt_container_name is None:
            # Fall back to code if name is not available
            gt_container_name = str(gt_container_code_int)

        # Extract container group predictions from top-1
        pred_container_group1 = (
            top_1_container.get("valid_diagnosis_group_1")
            if top_1_container is not None
            else None
        )
        pred_container_group2 = (
            top_1_container.get("valid_diagnosis_group_2")
            if top_1_container is not None
            else None
        )
        pred_container_group3 = (
            top_1_container.get("valid_diagnosis_group_3")
            if top_1_container is not None
            else None
        )
        pred_container_group4 = (
            top_1_container.get("valid_diagnosis_group_4")
            if top_1_container is not None
            else None
        )

        # Extract top-k container diagnosis codes for accuracy checks
        topk_container_codes = {
            k: extractors.extract_top_k_container_diagnosis_codes(record, k)
            for k in range(1, max(top_k_values) + 1)
        }

        # Extract top-k container group predictions
        top_3_container_groups = _extract_top_k_container_groups_dict(record, k=3)
        top_5_container_groups = _extract_top_k_container_groups_dict(record, k=5)

        top3_container_group1 = (
            top_3_container_groups.get("valid_diagnosis_group_1")
            if top_3_container_groups
            else None
        )
        top3_container_group2 = (
            top_3_container_groups.get("valid_diagnosis_group_2")
            if top_3_container_groups
            else None
        )
        top3_container_group3 = (
            top_3_container_groups.get("valid_diagnosis_group_3")
            if top_3_container_groups
            else None
        )
        top3_container_group4 = (
            top_3_container_groups.get("valid_diagnosis_group_4")
            if top_3_container_groups
            else None
        )

        top5_container_group1 = (
            top_5_container_groups.get("valid_diagnosis_group_1")
            if top_5_container_groups
            else None
        )
        top5_container_group2 = (
            top_5_container_groups.get("valid_diagnosis_group_2")
            if top_5_container_groups
            else None
        )
        top5_container_group3 = (
            top_5_container_groups.get("valid_diagnosis_group_3")
            if top_5_container_groups
            else None
        )
        top5_container_group4 = (
            top_5_container_groups.get("valid_diagnosis_group_4")
            if top_5_container_groups
            else None
        )

        # Extract ground-truth container fields
        gt_container_group1 = gt_row.get("GT Container WHO-like Subcategories")
        gt_container_group2 = gt_row.get("GT Container WHO-like Categories")
        gt_container_group3 = gt_row.get("GT Container WHO-like Major Sections/Lineages")
        gt_container_group4 = gt_row.get("GT Container Diagnosis Group 4")
        
        if pd.isna(gt_container_group1):
            gt_container_group1 = None
        if pd.isna(gt_container_group2):
            gt_container_group2 = None
        if pd.isna(gt_container_group3):
            gt_container_group3 = None
        if pd.isna(gt_container_group4):
            gt_container_group4 = None

        # Create container row (container-specific columns only, no report-level booleans)
        container_row = {
            "case_id": int(case_id),
            "gt_container_code": gt_container_code_int,
            "gt_container_group1": gt_container_group1,
            "gt_container_group2": gt_container_group2,
            "gt_container_group3": gt_container_group3,
            "gt_container_group4": gt_container_group4,
            "pred_container_code": pred_container_code,
            "pred_container_group1": pred_container_group1,
            "pred_container_group2": pred_container_group2,
            "pred_container_group3": pred_container_group3,
            "pred_container_group4": pred_container_group4,
            # Group 0 is the diagnosis code itself, but display the diagnosis name
            "gt_group0": gt_container_name,
            "pred_group0": pred_container_name if pred_container_name is not None else str(pred_container_code),
            # Top-k accuracy flags
            "container_top1_correct": gt_container_code_int in topk_container_codes.get(1, []),
            "container_top3_correct": gt_container_code_int in topk_container_codes.get(3, []),
            "container_top5_correct": gt_container_code_int in topk_container_codes.get(5, []),
            # Top-0 group accuracy flags (primary diagnosis code)
            "top1_group0_correct": gt_container_code_int in topk_container_codes.get(1, []),
            "top3_group0_correct": gt_container_code_int in topk_container_codes.get(3, []),
            "top5_group0_correct": gt_container_code_int in topk_container_codes.get(5, []),
            # Top-1 group accuracy flags
            "top1_group1_correct": gt_container_group1 == pred_container_group1,
            "top1_group2_correct": gt_container_group2 == pred_container_group2,
            "top1_group3_correct": gt_container_group3 == pred_container_group3,
            "top1_group4_correct": gt_container_group4 == pred_container_group4,
            # Top-3 group accuracy flags
            "top3_group1_correct": gt_container_group1 in top3_container_group1 if top3_container_group1 else False,
            "top3_group2_correct": gt_container_group2 in top3_container_group2 if top3_container_group2 else False,
            "top3_group3_correct": gt_container_group3 in top3_container_group3 if top3_container_group3 else False,
            "top3_group4_correct": gt_container_group4 in top3_container_group4 if top3_container_group4 else False,
            # Top-5 group accuracy flags
            "top5_group1_correct": gt_container_group1 in top5_container_group1 if top5_container_group1 else False,
            "top5_group2_correct": gt_container_group2 in top5_container_group2 if top5_container_group2 else False,
            "top5_group3_correct": gt_container_group3 in top5_container_group3 if top5_container_group3 else False,
            "top5_group4_correct": gt_container_group4 in top5_container_group4 if top5_container_group4 else False,
        }
        
        rows.append(container_row)

    container_df = pd.DataFrame(rows)

    # Remap columns for consistency with metric calculations
    if len(container_df) > 0:
        container_df["top1_correct"] = container_df["container_top1_correct"]
        container_df["top3_correct"] = container_df["container_top3_correct"]
        container_df["top5_correct"] = container_df["container_top5_correct"]

        # Rename ground-truth group columns
        container_df["gt_group1"] = container_df["gt_container_group1"]
        container_df["gt_group2"] = container_df["gt_container_group2"]
        container_df["gt_group3"] = container_df["gt_container_group3"]
        container_df["gt_group4"] = container_df["gt_container_group4"]
        # gt_group0 already set in container_row

        # Rename predicted group columns
        container_df["pred_group1"] = container_df["pred_container_group1"]
        container_df["pred_group2"] = container_df["pred_container_group2"]
        container_df["pred_group3"] = container_df["pred_container_group3"]
        container_df["pred_group4"] = container_df["pred_container_group4"]
        # pred_group0 already set in container_row

    return container_df

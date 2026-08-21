"""Analysis functions for detailed evaluation breakdowns.

Performs stratified analysis including hard cases, error attribution, rare diagnoses,
and prediction score threshold analysis.
"""

import logging

import numpy as np
import pandas as pd

from . import metrics_calculators as metrics

logger = logging.getLogger(__name__)


def compute_hard_case_metrics(
    report_df: pd.DataFrame,
    group_terminology: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute metrics for "hard" cases based on ground-truth conditions.

    Hard cases are defined as:
    - Cases with a differential diagnosis
    - Cases that are not definitive
    - Cases with a prior malignancy
    - Cases with a concurrent malignancy

    Args:
        report_df: Report-level DataFrame.
        group_terminology: Mapping of group keys to display names.

    Returns:
        DataFrame with one row per hard case category and associated metrics.
    """
    # Define hard case categories based on ground-truth conditions
    case_categories = {
        "has_differential": report_df["gt_has_differential"],
        "non_definitive": ~report_df["gt_is_definitive"],
        "has_prior_malignancy": report_df["gt_has_prior_malignancy"],
        "has_concurrent_malignancy": report_df["gt_has_concurrent_malignancy"],
    }

    rows = []
    for category_name, category_mask in case_categories.items():
        subset = report_df[category_mask]
        rows.append(
            {
                "category": category_name,
                "n": len(subset),
                **metrics.compute_summary_accuracy_metrics(subset, group_terminology),
            }
        )

    return pd.DataFrame(rows)


def compute_error_case_breakdown(report_df: pd.DataFrame) -> dict[str, float]:
    """Compute fractions of error cases by attribute combination.

    Analyzes all incorrect top-1 predictions and categorizes them by combinations of:
    - D: has differential diagnosis
    - N: NOT definitive (is_definitive = False)
    - P: has prior malignancy
    - C: has concurrent malignancy

    Returns mutually exclusive and exhaustive categories:
    - none: 0 attributes (0 cases)
    - one_*: exactly 1 attribute
    - two_**: exactly 2 attributes (all 6 pairs)
    - three_***: exactly 3 attributes (all 4 triples)
    - all_four: all 4 attributes

    Args:
        report_df: Report-level DataFrame.

    Returns:
        Dictionary mapping category names to fractions of total errors.
    """
    errors = report_df[~report_df["top1_correct"]].copy()

    if len(errors) == 0:
        return {}

    # Create boolean columns for each attribute
    errors["has_D"] = errors["gt_has_differential"]
    errors["has_N"] = ~errors["gt_is_definitive"]
    errors["has_P"] = errors["gt_has_prior_malignancy"]
    errors["has_C"] = errors["gt_has_concurrent_malignancy"]

    # Count number of attributes per error
    errors["num_attrs"] = (
        errors["has_D"].astype(int)
        + errors["has_N"].astype(int)
        + errors["has_P"].astype(int)
        + errors["has_C"].astype(int)
    )

    total = len(errors)
    result = {}

    # Errors with no attributes
    result["fraction_none"] = (errors["num_attrs"] == 0).sum() / total

    # Errors with exactly one attribute
    one_attr = errors[errors["num_attrs"] == 1]
    result["fraction_one_D"] = one_attr["has_D"].sum() / total
    result["fraction_one_N"] = one_attr["has_N"].sum() / total
    result["fraction_one_P"] = one_attr["has_P"].sum() / total
    result["fraction_one_C"] = one_attr["has_C"].sum() / total

    # Errors with exactly two attributes (6 pairs)
    two_attr = errors[errors["num_attrs"] == 2]
    result["fraction_two_DN"] = (two_attr["has_D"] & two_attr["has_N"]).sum() / total
    result["fraction_two_DP"] = (two_attr["has_D"] & two_attr["has_P"]).sum() / total
    result["fraction_two_DC"] = (two_attr["has_D"] & two_attr["has_C"]).sum() / total
    result["fraction_two_NP"] = (two_attr["has_N"] & two_attr["has_P"]).sum() / total
    result["fraction_two_NC"] = (two_attr["has_N"] & two_attr["has_C"]).sum() / total
    result["fraction_two_PC"] = (two_attr["has_P"] & two_attr["has_C"]).sum() / total

    # Errors with exactly three attributes (4 triples)
    three_attr = errors[errors["num_attrs"] == 3]
    result["fraction_three_DNP"] = (
        three_attr["has_D"] & three_attr["has_N"] & three_attr["has_P"]
    ).sum() / total
    result["fraction_three_DNC"] = (
        three_attr["has_D"] & three_attr["has_N"] & three_attr["has_C"]
    ).sum() / total
    result["fraction_three_DPC"] = (
        three_attr["has_D"] & three_attr["has_P"] & three_attr["has_C"]
    ).sum() / total
    result["fraction_three_NPC"] = (
        three_attr["has_N"] & three_attr["has_P"] & three_attr["has_C"]
    ).sum() / total

    # Errors with all four attributes
    result["fraction_all_four"] = (errors["num_attrs"] == 4).sum() / total

    return result


def compute_rare_diagnosis_metrics(
    report_df: pd.DataFrame,
    frequency_threshold: int,
    group_terminology: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute metrics stratified by diagnosis frequency.

    Separates cases into rare (< threshold occurrences) and common (>= threshold)
    diagnoses in the ground-truth dataset.

    Args:
        report_df: Report-level DataFrame.
        frequency_threshold: Minimum number of ground-truth occurrences for non-rare status.
        group_terminology: Mapping of group keys to display names.

    Returns:
        DataFrame with one row per rarity category and associated metrics.
    """
    report_df = report_df.copy()

    frequencies = report_df["gt_code"].value_counts()
    rare_codes = set(frequencies[frequencies < frequency_threshold].index)

    report_df["is_rare"] = report_df["gt_code"].isin(rare_codes)

    rows = []
    for is_rare in [True, False]:
        subset = report_df[report_df["is_rare"] == is_rare]
        rows.append(
            {
                "rare": is_rare,
                "n": len(subset),
                **metrics.compute_summary_accuracy_metrics(subset, group_terminology),
            }
        )

    return pd.DataFrame(rows)


def compute_threshold_sweep_metrics(
    report_df: pd.DataFrame,
    num_threshold_steps: int,
) -> pd.DataFrame:
    """Compute accuracy and coverage metrics across prediction score thresholds.

    Sweeps through linearly-spaced thresholds from min to max prediction score
    and computes the fraction of cases above each threshold (coverage) and
    accuracy metrics for those cases.

    Args:
        report_df: Report-level DataFrame with pred_score column.
        num_threshold_steps: Number of threshold points to evaluate.

    Returns:
        DataFrame with threshold, coverage, and accuracy metrics for each step.
    """
    thresholds = np.linspace(
        report_df["pred_score"].min(),
        report_df["pred_score"].max(),
        num_threshold_steps,
    )

    rows = []
    for threshold in thresholds:
        subset = report_df[report_df["pred_score"] >= threshold]

        rows.append(
            {
                "threshold": threshold,
                "coverage": len(subset) / len(report_df),
                "accuracy_top1": subset["top1_correct"].mean()
                if len(subset)
                else np.nan,
                "accuracy_top3": subset["top3_correct"].mean()
                if len(subset)
                else np.nan,
                "accuracy_top5": subset["top5_correct"].mean()
                if len(subset)
                else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_accuracy_by_group(
    report_df: pd.DataFrame,
    group_column: str,
    group_terminology: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute accuracy metrics stratified by diagnosis group.

    Breaks down metrics for each unique value in the specified group column
    (e.g., gt_group1, gt_group2, gt_group3).

    Args:
        report_df: Report-level DataFrame.
        group_column: Column name to group by (e.g., "gt_group1").
        group_terminology: Mapping of group keys to display names.

    Returns:
        DataFrame with one row per group value and associated metrics.
    """
    rows = []

    for group_value in report_df[group_column].dropna().unique():
        subset = report_df[report_df[group_column] == group_value]
        rows.append(
            {
                "group": group_value,
                "n": len(subset),
                **metrics.compute_summary_accuracy_metrics(subset, group_terminology),
            }
        )

    return pd.DataFrame(rows)

"""Metric calculators for quantitative evaluation analysis.

Computes accuracy, Matthews correlation coefficient, and other performance metrics.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)


def calculate_accuracy(correctness_series: pd.Series) -> float:
    """Calculate accuracy from a series of correctness flags.

    Args:
        correctness_series: Boolean Series indicating correctness (True/False).

    Returns:
        Accuracy as float between 0 and 1, or NaN if series is empty.
    """
    if len(correctness_series) == 0:
        return np.nan
    return float(correctness_series.mean())


def calculate_matthews_correlation_coefficient(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> float:
    """Calculate Matthews Correlation Coefficient (MCC).

    Provides a balanced metric for binary classification that works even with imbalanced classes.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        MCC score between -1 and 1, or NaN if series is empty.
    """
    if len(y_true) == 0:
        return np.nan
    return float(matthews_corrcoef(y_true, y_pred))


def normalize_group_key(group_key: str) -> str:
    """Normalize group key for DataFrame column access.

    Converts "group_1" to "group1" by removing underscores.

    Args:
        group_key: Group key like "group_1", "group_2", etc.

    Returns:
        Normalized group key without underscores.
    """
    return group_key.replace("_", "")


def make_metric_column_name(prefix: str, group_name: str, suffix: str) -> str:
    """Build a metric column name from components.

    Replaces spaces and special characters with underscores and converts to lowercase.

    Args:
        prefix: Prefix for the metric name (e.g., "").
        group_name: Group display name (e.g., "Malignant/neoplastic").
        suffix: Suffix for the metric name (e.g., "top1_accuracy").

    Returns:
        Formatted metric column name.
    """
    # Filter out empty prefix
    parts = [p for p in [prefix, group_name, suffix] if p]
    return (
        "_".join(parts).lower().replace(" ", "_").replace("/", "_").replace("\\", "_")
    )


def sanitize_for_filename(text: str) -> str:
    """Convert text to a filename-safe string.

    Converts to lowercase and replaces spaces and special characters with underscores.

    Args:
        text: Text to sanitize.

    Returns:
        Filename-safe string.
    """
    return text.lower().replace(" ", "_").replace("/", "_").replace("\\", "_")


def compute_summary_accuracy_metrics(
    report_df: pd.DataFrame,
    group_terminology: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute accuracy metrics for primary diagnoses and condition attributes.

    Generates metrics for:
    - Top-1, Top-3, Top-5 primary diagnosis accuracy
    - Condition detection (differential, definitive, prior/concurrent malignancy)
    - Group-level accuracy for each diagnosis group

    Args:
        report_df: Report-level DataFrame with accuracy flags.
        group_terminology: Mapping of group keys to display names.

    Returns:
        Dictionary of computed metric names and values.

    Raises:
        ValueError: If group_terminology is not provided.
    """
    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    result = {
        "primary_top1_accuracy": calculate_accuracy(report_df["top1_correct"]),
        "primary_top3_accuracy": calculate_accuracy(report_df["top3_correct"]),
        "primary_top5_accuracy": calculate_accuracy(report_df["top5_correct"]),
        "differential_mentioned_accuracy": calculate_accuracy(
            report_df["has_differential_correct"]
        ),
        "is_definitive_accuracy": calculate_accuracy(
            report_df["is_definitive_correct"]
        ),
        "prior_malignancy_accuracy": calculate_accuracy(
            report_df["has_prior_malignancy_correct"]
        ),
        "concurrent_malignancy_accuracy": calculate_accuracy(
            report_df["has_concurrent_malignancy_correct"]
        ),
        "differential_mentioned_mcc": calculate_matthews_correlation_coefficient(
            report_df["gt_has_differential"], report_df["pred_has_differential"]
        ),
        "is_definitive_mcc": calculate_matthews_correlation_coefficient(
            report_df["gt_is_definitive"], report_df["pred_is_definitive"]
        ),
        "prior_malignancy_mcc": calculate_matthews_correlation_coefficient(
            report_df["gt_has_prior_malignancy"], report_df["pred_has_prior_malignancy"]
        ),
        "concurrent_malignancy_mcc": calculate_matthews_correlation_coefficient(
            report_df["gt_has_concurrent_malignancy"],
            report_df["pred_has_concurrent_malignancy"],
        ),
    }

    # Add dynamic group-based metrics
    for group_key, group_name in group_terminology.items():
        group_normalized = normalize_group_key(group_key)
        result[make_metric_column_name("", group_name, "top1_accuracy")] = (
            calculate_accuracy(report_df[f"top1_{group_normalized}_correct"])
        )
        result[make_metric_column_name("", group_name, "top3_accuracy")] = (
            calculate_accuracy(report_df[f"top3_{group_normalized}_correct"])
        )
        result[make_metric_column_name("", group_name, "top5_accuracy")] = (
            calculate_accuracy(report_df[f"top5_{group_normalized}_correct"])
        )

    return result


def compute_conditional_group_accuracy_metrics(
    report_df: pd.DataFrame,
    group_terminology: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute top-1 accuracy given correct group classification.

    For each diagnosis group, calculates top-1 accuracy but only among cases where
    the group-level top-1 prediction was already correct.

    Args:
        report_df: Report-level DataFrame with accuracy flags.
        group_terminology: Mapping of group keys to display names.

    Returns:
        Dictionary of computed metric names and values.

    Raises:
        ValueError: If group_terminology is not provided.
    """
    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    result = {}
    for group_key, group_name in group_terminology.items():
        group_normalized = normalize_group_key(group_key)
        # Subset to cases where top-1 group classification was correct
        subset = report_df.loc[
            report_df[f"top1_{group_normalized}_correct"], "top1_correct"
        ]
        result[
            make_metric_column_name("", group_name, "top1_accuracy_given_correct")
        ] = calculate_accuracy(subset)

    return result


def compute_report_level_metrics(
    report_df: pd.DataFrame,
    group_terminology: dict[str, str] | None = None,
) -> dict[str, float]:
    """Compute all report-level metrics.

    Combines summary accuracy metrics with conditional group accuracy metrics.

    Args:
        report_df: Report-level DataFrame with accuracy flags.
        group_terminology: Mapping of group keys to display names.

    Returns:
        Dictionary of all computed metrics including case count.
    """
    return {
        "n_cases": len(report_df),
        **compute_summary_accuracy_metrics(report_df, group_terminology),
        **compute_conditional_group_accuracy_metrics(report_df, group_terminology),
    }

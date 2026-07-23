import yaml
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
import seaborn as sns

logger = logging.getLogger(__name__)

short_names = {
    "Chronic lymphocytic leukemia/Small lymphocytic lymphoma": "CLL/SLL",
    "Immunodeficiency-associated lymphoproliferative disorders": "IAL",
    "Monoclonal Immunoglobulin deposition": "MIDD",
    "Tumor-like lesions with T cell predominance": "TLT",
    "Tumor-like lesions with B cell predominance": "TLB",
    "Histiocytic/dendritic cell neoplasms": "HDCN",
    "Hodgkin lymphoma or Mature B cell?": "HL/MBC",
}


###############################################################################
# IO
###############################################################################


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error("Configuration file not found: %s", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records."""
    records: list[dict[str, Any]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def load_gt(path: Path, gt_sheet_name: str) -> pd.DataFrame:
    """Load ground-truth annotations."""
    return pd.read_excel(path, sheet_name=gt_sheet_name)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with stable indentation."""
    with path.open("w") as f:
        json.dump(data, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predictions against ground-truth annotations."
    )

    parser.add_argument(
        "--guidance",
        action="store_true",
        help="Skip score-based evaluation for guidance mode.",
    )

    parser.add_argument(
        "--config",
        default="configs/evaluation.yaml",
        help="Path to evaluation configuration file",
    )

    return parser.parse_args()


###############################################################################
# PREDICTION ACCESSORS
###############################################################################


def get_valid_primary_diagnoses(record: dict[str, Any]) -> dict[str, Any]:
    """Return the dictionary of top-k valid primary diagnoses from the record."""
    return record.get("valid_primary_diagnoses", {})


def get_primary_top1(record: dict[str, Any]) -> dict[str, Any] | None:
    """Return the dictionary for the top-1 valid primary diagnosis from the record.
    {
        "valid_primary_diagnosis_code": str,
        "valid_primary_diagnosis_name": str,
        "valid_primary_diagnosis_score": float,
        "valid_primary_diagnosis_group_1": str,
        "valid_primary_diagnosis_group_2": str,
        "valid_primary_diagnosis_group_3": str
    }"""
    return get_valid_primary_diagnoses(record).get("top_1")


def get_primary_top3(record: dict[str, Any], group: str) -> list[str] | None:
    """Return a list of valid primary diagnosis groups for the top-3 predictions."""
    if group == "group_1":
        return [
            record.get("valid_primary_diagnoses", {})
            .get(f"top_{k}", {})
            .get("valid_primary_diagnosis_group_1", {})
            for k in range(1, 4)
        ]

    if group == "group_2":
        return [
            record.get("valid_primary_diagnoses", {})
            .get(f"top_{i}", {})
            .get("valid_primary_diagnosis_group_2", {})
            for i in range(1, 4)
        ]

    if group == "group_3":
        return [
            record.get("valid_primary_diagnoses", {})
            .get(f"top_{i}", {})
            .get("valid_primary_diagnosis_group_3", {})
            for i in range(1, 4)
        ]

    return None


def get_primary_top5(record: dict[str, Any], group: str) -> list[str] | None:
    """Return a list of valid primary diagnosis groups for the top-5 predictions."""
    if group == "group_1":
        return [
            record.get("valid_primary_diagnoses", {})
            .get(f"top_{i}", {})
            .get("valid_primary_diagnosis_group_1", {})
            for i in range(1, 6)
        ]

    if group == "group_2":
        return [
            record.get("valid_primary_diagnoses", {})
            .get(f"top_{i}", {})
            .get("valid_primary_diagnosis_group_2", {})
            for i in range(1, 6)
        ]

    if group == "group_3":
        return [
            record.get("valid_primary_diagnoses", {})
            .get(f"top_{i}", {})
            .get("valid_primary_diagnosis_group_3", {})
            for i in range(1, 6)
        ]

    return None


def get_primary_top_k_codes(record: dict[str, Any], k: int) -> list[int]:
    """Return a list of valid primary diagnosis codes for the top-k predictions."""
    preds = get_valid_primary_diagnoses(record)
    codes: list[int] = []

    for rank in range(1, k + 1):
        item = preds.get(f"top_{rank}")

        if item:
            codes.append(item.get("valid_primary_diagnosis_code"))

    return codes


def build_prediction_map(records: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Build a mapping from case_id to prediction record from the list of records."""
    return {int(record["case_id"]): record for record in records}


###############################################################################
# REPORT-LEVEL DATASET
###############################################################################


def build_report_level_df(
    gt: pd.DataFrame, prediction_map: dict[int, dict[str, Any]], top_k_values: list[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    """ Build a report-level DataFrame by combining ground-truth and prediction data."""

    for case_id in gt["Id"].dropna().unique():
        record = prediction_map.get(int(case_id))

        if record is None:
            continue

        gt_row = gt.loc[gt["Id"] == case_id].iloc[0]
        top1 = get_primary_top1(record)
        top3 = {
            "valid_primary_diagnosis_group_1": get_primary_top3(record, "group_1"),
            "valid_primary_diagnosis_group_2": get_primary_top3(record, "group_2"),
            "valid_primary_diagnosis_group_3": get_primary_top3(record, "group_3"),
        }
        top5 = {
            "valid_primary_diagnosis_group_1": get_primary_top5(record, "group_1"),
            "valid_primary_diagnosis_group_2": get_primary_top5(record, "group_2"),
            "valid_primary_diagnosis_group_3": get_primary_top5(record, "group_3"),
        }
        if top1 is None:
            continue

        if top3 is None:
            continue

        if top5 is None:
            continue

        rows.append(
            build_report_row(case_id, gt_row, record, top1, top3, top5, top_k_values)
        )

    return pd.DataFrame(rows)


def build_report_row(
    case_id: Any,
    gt_row: pd.Series,
    record: dict[str, Any],
    top1: dict[str, Any],
    top3: dict[str, Any],
    top5: dict[str, Any],
    top_k_values: list[int],
) -> dict[str, Any]:
    """Build a single row for the report-level DataFrame."""

    gt_code = gt_row["GT Code"]
    topk_codes = {
        k: get_primary_top_k_codes(record, k) for k in range(1, max(top_k_values) + 1)
    }

    gt_group1 = gt_row["GT Report Diagnosis Group 1"]
    gt_group2 = gt_row["GT Report Diagnosis Group 2"]
    gt_group3 = gt_row["GT Report Diagnosis Group 3"]

    pred_group1 = top1.get("valid_primary_diagnosis_group_1")
    pred_group2 = top1.get("valid_primary_diagnosis_group_2")
    pred_group3 = top1.get("valid_primary_diagnosis_group_3")

    top3_group1 = top3.get("valid_primary_diagnosis_group_1")
    top3_group2 = top3.get("valid_primary_diagnosis_group_2")
    top3_group3 = top3.get("valid_primary_diagnosis_group_3")

    top5_group1 = top5.get("valid_primary_diagnosis_group_1")
    top5_group2 = top5.get("valid_primary_diagnosis_group_2")
    top5_group3 = top5.get("valid_primary_diagnosis_group_3")

    gt_has_differential = str_to_bool(gt_row["GT Has Differential Diagnosis"])
    gt_is_definitive = str_to_bool(gt_row["GT Is Definitive"])
    gt_has_prior_malignancy = str_to_bool(gt_row["GT Has Prior Malignancy"])
    gt_has_concurrent_malignancy = str_to_bool(gt_row["GT Has Concurrent Malignancy"])

    pred_has_differential = str_to_bool(record.get("has_differential_diagnosis"))
    pred_is_definitive = str_to_bool(record.get("is_definitive"))
    pred_has_prior_malignancy = str_to_bool(record.get("has_prior_malignancy"))
    pred_has_concurrent_malignancy = str_to_bool(
        record.get("has_concurrent_malignancy")
    )

    return {
        "case_id": case_id,
        "gt_code": gt_code,
        "pred_code": top1.get("valid_primary_diagnosis_code"),
        "pred_score": top1.get("valid_primary_diagnosis_score"),
        "gt_group1": gt_group1,
        "gt_group2": gt_group2,
        "gt_group3": gt_group3,
        "pred_group1": pred_group1,
        "pred_group2": pred_group2,
        "pred_group3": pred_group3,
        "gt_has_differential": gt_has_differential,
        "gt_is_definitive": gt_is_definitive,
        "gt_has_prior_malignancy": gt_has_prior_malignancy,
        "gt_has_concurrent_malignancy": gt_has_concurrent_malignancy,
        "pred_has_differential": pred_has_differential,
        "pred_is_definitive": pred_is_definitive,
        "pred_has_prior_malignancy": pred_has_prior_malignancy,
        "pred_has_concurrent_malignancy": pred_has_concurrent_malignancy,
        "top1_correct": gt_code in topk_codes[1],
        "top3_correct": gt_code in topk_codes[3],
        "top5_correct": gt_code in topk_codes[5],
        "top1_group1_correct": gt_group1 == pred_group1,
        "top1_group2_correct": gt_group2 == pred_group2,
        "top1_group3_correct": gt_group3 == pred_group3,
        "top3_group1_correct": gt_group1 in top3_group1,
        "top3_group2_correct": gt_group2 in top3_group2,
        "top3_group3_correct": gt_group3 in top3_group3,
        "top5_group1_correct": gt_group1 in top5_group1,
        "top5_group2_correct": gt_group2 in top5_group2,
        "top5_group3_correct": gt_group3 in top5_group3,
        "has_differential_correct": gt_has_differential == pred_has_differential,
        "is_definitive_correct": gt_is_definitive == pred_is_definitive,
        "has_prior_malignancy_correct": gt_has_prior_malignancy
        == pred_has_prior_malignancy,
        "has_concurrent_malignancy_correct": gt_has_concurrent_malignancy
        == pred_has_concurrent_malignancy,
    }


def str_to_bool(value: str) -> bool:
    """Convert a string to a boolean value."""

    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    if not isinstance(value, str):
        logger.error("Expected a string or boolean, got %s", type(value))
        raise ValueError(f"Expected a string or boolean, got {type(value)}")

    value_lower = value.strip().lower()

    if value_lower in {"true", "1", "yes"}:
        return True
    elif value_lower in {"false", "0", "no"}:
        return False
    else:
        logger.error("Cannot convert string to boolean: %s", value)
        raise ValueError(f"Cannot convert string to boolean: {value}")


###############################################################################
# METRIC HELPERS
###############################################################################


def accuracy(series: pd.Series) -> float:
    if len(series) == 0:
        return np.nan

    return float(series.mean())


def mcc(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(matthews_corrcoef(y_true, y_pred))


def summarize_accuracy(df: pd.DataFrame) -> dict[str, float]:
    return {
        "primary_top1_accuracy": accuracy(df["top1_correct"]),
        "primary_top3_accuracy": accuracy(df["top3_correct"]),
        "primary_top5_accuracy": accuracy(df["top5_correct"]),
        "i-CLASSi_1_top1_accuracy": accuracy(df["top1_group1_correct"]),
        "i-CLASSi_2_top1_accuracy": accuracy(df["top1_group2_correct"]),
        "i-CLASSi_3_top1_accuracy": accuracy(df["top1_group3_correct"]),
        "i-CLASSi_1_top3_accuracy": accuracy(df["top3_group1_correct"]),
        "i-CLASSi_2_top3_accuracy": accuracy(df["top3_group2_correct"]),
        "i-CLASSi_3_top3_accuracy": accuracy(df["top3_group3_correct"]),
        "i-CLASSi_1_top5_accuracy": accuracy(df["top5_group1_correct"]),
        "i-CLASSi_2_top5_accuracy": accuracy(df["top5_group2_correct"]),
        "i-CLASSi_3_top5_accuracy": accuracy(df["top5_group3_correct"]),
        "differential_mentioned_accuracy": accuracy(df["has_differential_correct"]),
        "is_definitive_accuracy": accuracy(df["is_definitive_correct"]),
        "prior_malignancy_accuracy": accuracy(df["has_prior_malignancy_correct"]),
        "concurrent_malignancy_accuracy": accuracy(
            df["has_concurrent_malignancy_correct"]
        ),
        "differential_mentioned_mcc": mcc(
            df["gt_has_differential"], df["pred_has_differential"]
        ),
        "is_definitive_mcc": mcc(df["gt_is_definitive"], df["pred_is_definitive"]),
        "prior_malignancy_mcc": mcc(
            df["gt_has_prior_malignancy"], df["pred_has_prior_malignancy"]
        ),
        "concurrent_malignancy_mcc": mcc(
            df["gt_has_concurrent_malignancy"], df["pred_has_concurrent_malignancy"]
        ),
    }


def compute_report_metrics(df: pd.DataFrame) -> dict[str, float]:
    return {
        "n_cases": len(df),
        **summarize_accuracy(df),
        **stratified_top1_accuracy(df),
    }


def stratified_top1_accuracy(df: pd.DataFrame) -> dict[str, float]:
    return {
        "i-CLASSi_1_top1_accuracy_given_correct": accuracy(
            df.loc[df["top1_group1_correct"], "top1_correct"]
        ),
        "i-CLASSi_2_top1_accuracy_given_correct": accuracy(
            df.loc[df["top1_group2_correct"], "top1_correct"]
        ),
        "i-CLASSi_3_top1_accuracy_given_correct": accuracy(
            df.loc[df["top1_group3_correct"], "top1_correct"]
        ),
    }


###############################################################################
# HARD CASE ANALYSIS
###############################################################################


def hard_case_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute metrics for "hard" cases based on ground-truth conditions:
    - differential: cases with a differential diagnosis
    - non_definitive: cases that are not definitive
    - prior_malignancy: cases with a prior malignancy
    - concurrent_malignancy: cases with a concurrent malignancy
    """
    definitions = {
        "has_differential": df["gt_has_differential"],
        "non_definitive": ~df["gt_is_definitive"],
        "has_prior_malignancy": df["gt_has_prior_malignancy"],
        "has_concurrent_malignancy": df["gt_has_concurrent_malignancy"],
    }

    rows = []

    for category, mask in definitions.items():
        subset = df[mask]
        rows.append(
            {
                "category": category,
                "n": len(subset),
                **summarize_accuracy(subset),
            }
        )

    return pd.DataFrame(rows)


###############################################################################
# ACCURACY IN DIFFERENT ICLASSI GROUPS
###############################################################################
def accuracy_in_groups(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Compute accuracy metrics for different ICLASSI groups."""
    rows = []

    for group in df[group_col].dropna().unique():
        subset = df[df[group_col] == group]
        rows.append(
            {
                "group": group,
                "n": len(subset),
                **summarize_accuracy(subset),
            }
        )

    return pd.DataFrame(rows)


###############################################################################
# ERROR ANALYSIS
###############################################################################


def error_analysis(df: pd.DataFrame) -> dict[str, float]:
    """Compute metrics for error cases (where top-1 prediction is incorrect).

    Returns mutually exclusive and exhaustive categories based on combinations of:
    - gt_has_differential (D)
    - gt_is_definitive (N = NOT definitive)
    - gt_has_prior_malignancy (P)
    - gt_has_concurrent_malignancy (C)

    Categories:
    - none: 0 attributes
    - one_*: exactly 1 attribute
    - two_**: exactly 2 attributes (all pairs)
    - three_***: exactly 3 attributes (all triples)
    - all_four: all 4 attributes
    """
    errors = df[~df["top1_correct"]].copy()

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

    # None (0 attributes)
    result["fraction_none"] = (errors["num_attrs"] == 0).sum() / total

    # Exactly one attribute
    one_attr = errors[errors["num_attrs"] == 1]
    result["fraction_one_D"] = one_attr["has_D"].sum() / total
    result["fraction_one_N"] = one_attr["has_N"].sum() / total
    result["fraction_one_P"] = one_attr["has_P"].sum() / total
    result["fraction_one_C"] = one_attr["has_C"].sum() / total

    # Exactly two attributes (all 6 pairs)
    two_attr = errors[errors["num_attrs"] == 2]
    result["fraction_two_DN"] = (two_attr["has_D"] & two_attr["has_N"]).sum() / total
    result["fraction_two_DP"] = (two_attr["has_D"] & two_attr["has_P"]).sum() / total
    result["fraction_two_DC"] = (two_attr["has_D"] & two_attr["has_C"]).sum() / total
    result["fraction_two_NP"] = (two_attr["has_N"] & two_attr["has_P"]).sum() / total
    result["fraction_two_NC"] = (two_attr["has_N"] & two_attr["has_C"]).sum() / total
    result["fraction_two_PC"] = (two_attr["has_P"] & two_attr["has_C"]).sum() / total

    # Exactly three attributes (all 4 triples)
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

    # All four attributes
    result["fraction_all_four"] = (errors["num_attrs"] == 4).sum() / total

    return result


###############################################################################
# RARE DIAGNOSIS ANALYSIS
###############################################################################


def rare_diagnosis_metrics(
    df: pd.DataFrame,
    threshold: int,
) -> pd.DataFrame:
    """Compute metrics for rare diagnoses based on a frequency threshold i.e.
    diagnoses that appear fewer than `threshold` times in the GT dataset."""
    df = df.copy()

    frequencies = df["gt_code"].value_counts()
    rare_codes = set(frequencies[frequencies < threshold].index)

    df["rare"] = df["gt_code"].isin(rare_codes)

    rows = []

    for is_rare in [True, False]:
        subset = df[df["rare"] == is_rare]

        rows.append(
            {
                "rare": is_rare,
                "n": len(subset),
                **summarize_accuracy(subset),
            }
        )

    return pd.DataFrame(rows)


###############################################################################
# THRESHOLD ANALYSIS
###############################################################################


def threshold_analysis(
    df: pd.DataFrame,
    n_steps: int,
) -> pd.DataFrame:
    """Compute accuracy and coverage metrics for prediction score threshold steps."""
    thresholds = np.linspace(
        df["pred_score"].min(),
        df["pred_score"].max(),
        n_steps,
    )

    rows = []

    for threshold in thresholds:
        subset = df[df["pred_score"] >= threshold]

        rows.append(
            {
                "threshold": threshold,
                "coverage": len(subset) / len(df),
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


###############################################################################
# PLOTTING
###############################################################################


def make_threshold_plot(threshold_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax_accuracy = plt.subplots(figsize=(8, 6))

    ax_accuracy.plot(
        threshold_df["threshold"],
        threshold_df["accuracy_top1"],
        color="blue",
        label="Top-1 Accuracy",
    )
    ax_accuracy.plot(
        threshold_df["threshold"],
        threshold_df["accuracy_top3"],
        linestyle="--",
        color="blue",
        label="Top-3 Accuracy",
    )
    ax_accuracy.plot(
        threshold_df["threshold"],
        threshold_df["accuracy_top5"],
        linestyle=":",
        color="blue",
        label="Top-5 Accuracy",
    )
    ax_accuracy.set_ylabel("Accuracy", color="blue")
    ax_accuracy.tick_params(axis="y", labelcolor="blue")
    ax_accuracy.set_xlabel("Prediction Score Threshold")
    ax_accuracy.set_xlim(
        threshold_df["threshold"].min() - 0.01, threshold_df["threshold"].max() + 0.01
    )
    ax_accuracy.set_ylim(-0.05, 1.05)

    ax_accuracy.legend(loc="lower left")

    ax_coverage = ax_accuracy.twinx()
    ax_coverage.plot(
        threshold_df["threshold"],
        threshold_df["coverage"],
        color="red",
        label="Coverage",
    )
    ax_coverage.set_ylabel("Coverage", color="red")
    ax_coverage.tick_params(axis="y", labelcolor="red")
    ax_coverage.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def make_coverage_accuracy_plot(threshold_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        threshold_df["accuracy_top1"],
        threshold_df["coverage"],
        color="blue",
        label="Top-1 Accuracy",
    )
    ax.plot(
        threshold_df["accuracy_top3"],
        threshold_df["coverage"],
        linestyle="--",
        color="blue",
        label="Top-3 Accuracy",
    )
    ax.plot(
        threshold_df["accuracy_top5"],
        threshold_df["coverage"],
        linestyle=":",
        color="blue",
        label="Top-5 Accuracy",
    )
    ax.set_xlabel("Accuracy")
    ax.set_ylabel("Coverage")
    ax.set_xlim(0.7, 1.01)
    ax.set_ylim(0.0, 1.01)

    ax.legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close(fig)


def make_group_confusion_matrix_plot(
    df: pd.DataFrame, group: str, output_path: Path
) -> None:
    """Make a confusion matrix plot for the specified diagnosis group."""

    y_true = ["nan" if pd.isna(x) else x for x in df[f"gt_{group}"]]
    y_true = [
        short_names.get(x, x) if x in short_names else x for x in df[f"gt_{group}"]
    ]
    y_pred = ["nan" if pd.isna(x) else x for x in df[f"pred_{group}"]]
    y_pred = [
        short_names.get(x, x) if x in short_names else x for x in df[f"pred_{group}"]
    ]

    _normalize = True
    if _normalize:
        cm = confusion_matrix(
            y_true, y_pred, labels=np.unique(y_true), normalize="pred"
        )
        cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        annot = cm_df.map(lambda x: "0" if x == 0 else f"{x:.2f}")
    else:
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
        cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        annot = True

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_df, annot=annot, fmt="", cmap="Blues")
    plt.title(f"Confusion Matrix for {group}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


###############################################################################
# EVALUATION PIPELINE
###############################################################################


def run_evaluation(
    gt_excel: Path,
    predictions_jsonl: Path,
    output_dir: Path,
    params: dict[str, Any],
    output_file_names: dict[str, str],
    exclude_failed: bool = True,
    guidance: bool = False,
) -> None:
    logger.info("Starting evaluation")
    logger.info("Loading ground-truth from %s", gt_excel)
    logger.info("Loading predictions from %s", predictions_jsonl)

    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_gt(gt_excel, params.get("GT_SHEET_NAME"))
    records = load_jsonl(predictions_jsonl)
    prediction_map = build_prediction_map(records)

    logger.info("Building report-level DataFrame")
    # Build a DataFrame with one row per report.
    report_df = build_report_level_df(gt, prediction_map, params.get("TOP_K_VALUES"))

    if exclude_failed and not guidance:
        report_df = report_df.dropna(subset=["pred_score"])

    # Compute aggregate metrics based on the report-level DataFrame.
    logger.info("Computing metrics")
    report_metrics = compute_report_metrics(report_df)
    # Compute aggregate metrics for "hard" cases only.
    hard_df = hard_case_metrics(report_df)
    # Compute aggregate metrics for error cases only.
    errors_metrics = error_analysis(report_df)
    # Compute accuracy metrics for different ICLASSI groups.
    group1_accuracy_breakdown_df = accuracy_in_groups(report_df, "gt_group1")
    group2_accuracy_breakdown_df = accuracy_in_groups(report_df, "gt_group2")
    group3_accuracy_breakdown_df = accuracy_in_groups(report_df, "gt_group3")

    # Compare aggregate metrics for rare vs. common diagnoses.
    rare_df = rare_diagnosis_metrics(report_df, params.get("RARE_DIAGNOSIS_THRESHOLD"))
    # Compute accuracy and coverage metrics vs. prediction score threshold.
    if guidance:
        threshold_df = None
    else:
        threshold_df = threshold_analysis(
            report_df,
            params.get("THRESHOLD_STEPS"),
        )

        for k in params.get("TOP_K_VALUES"):
            report_metrics[f"threshold_auc_top{k}"] = float(
                auc(
                    threshold_df["coverage"],
                    threshold_df[f"accuracy_top{k}"],
                )
            )

    logger.info("Writing evaluation results to %s", output_dir)
    write_outputs(
        output_dir=output_dir,
        report_metrics=report_metrics,
        report_df=report_df,
        hard_df=hard_df,
        errors_metrics=errors_metrics,
        rare_df=rare_df,
        threshold_df=threshold_df,
        group1_accuracy_breakdown_df=group1_accuracy_breakdown_df,
        group2_accuracy_breakdown_df=group2_accuracy_breakdown_df,
        group3_accuracy_breakdown_df=group3_accuracy_breakdown_df,
        output_file_names=output_file_names,
    )
    logger.info("Evaluation completed successfully")


def write_outputs(
    output_dir: Path,
    report_metrics: dict[str, float],
    report_df: pd.DataFrame,
    hard_df: pd.DataFrame,
    errors_metrics: dict[str, float],
    rare_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    group1_accuracy_breakdown_df: pd.DataFrame,
    group2_accuracy_breakdown_df: pd.DataFrame,
    group3_accuracy_breakdown_df: pd.DataFrame,
    output_file_names: dict[str, str],
) -> None:
    logger.info(
        "Writing report-level metrics to %s",
        output_dir / output_file_names.get("OUTPUT_REPORT_LEVEL"),
    )
    report_df.to_csv(
        output_dir / output_file_names.get("OUTPUT_REPORT_LEVEL"), index=False
    )
    hard_df.to_csv(output_dir / output_file_names.get("OUTPUT_HARD_CASES"), index=False)
    rare_df.to_csv(
        output_dir / output_file_names.get("OUTPUT_RARE_DIAGNOSIS"), index=False
    )

    group1_accuracy_breakdown_df.to_csv(
        output_dir / output_file_names.get("OUTPUT_GROUP1_ACCURACY"), index=False
    )
    group2_accuracy_breakdown_df.to_csv(
        output_dir / output_file_names.get("OUTPUT_GROUP2_ACCURACY"), index=False
    )
    group3_accuracy_breakdown_df.to_csv(
        output_dir / output_file_names.get("OUTPUT_GROUP3_ACCURACY"), index=False
    )

    write_json(report_metrics, output_dir / output_file_names.get("OUTPUT_SUMMARY"))
    write_json(
        errors_metrics, output_dir / output_file_names.get("OUTPUT_ERROR_ANALYSIS")
    )

    if threshold_df is not None:
        make_threshold_plot(
            threshold_df,
            output_dir / output_file_names.get("OUTPUT_THRESHOLD_PLOT"),
        )

        make_coverage_accuracy_plot(
            threshold_df,
            output_dir / output_file_names.get("OUTPUT_COVERAGE_ACCURACY_PLOT"),
        )

    if "OUTPUT_CONFUSION_MATRIX_GROUP1" in output_file_names:
        make_group_confusion_matrix_plot(
            report_df,
            "group1",
            output_dir / output_file_names.get("OUTPUT_CONFUSION_MATRIX_GROUP1"),
        )

    if "OUTPUT_CONFUSION_MATRIX_GROUP2" in output_file_names:
        make_group_confusion_matrix_plot(
            report_df,
            "group2",
            output_dir / output_file_names.get("OUTPUT_CONFUSION_MATRIX_GROUP2"),
        )

    if "OUTPUT_CONFUSION_MATRIX_GROUP3" in output_file_names:
        make_group_confusion_matrix_plot(
            report_df,
            "group3",
            output_dir / output_file_names.get("OUTPUT_CONFUSION_MATRIX_GROUP3"),
        )


###############################################################################
# MAIN
###############################################################################


def main() -> None:

    args = parse_args()

    logger.info("Loading evaluation configuration from %s", args.config)
    config = load_config(args.config)

    params = config.get("parameters", {})
    output_file_names = config.get("output_files", {})

    run_evaluation(
        gt_excel=Path(config["input_excel_file"]),
        predictions_jsonl=Path(config["input_jsonl_file"]),
        output_dir=Path(config["output_folder"]),
        params=params,
        output_file_names=output_file_names,
        exclude_failed=params.get("EXCLUDE_FAILED", True),
        guidance=args.guidance,
    )

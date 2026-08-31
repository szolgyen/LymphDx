"""Evaluation pipeline orchestrator.

Main entry point that coordinates loading data, building analysis datasets,
computing metrics, performing stratified analyses, and generating outputs.
"""

import logging
import pandas as pd
from pathlib import Path
from typing import Any
from sklearn.metrics import auc

from .evaluation_modules import io_utils
from .evaluation_modules import prediction_extractors
from .evaluation_modules import dataframe_builders
from .evaluation_modules import metrics_calculators
from .evaluation_modules import analysis_functions
from .evaluation_modules import plotting_generators

logger = logging.getLogger(__name__)

###############################################################################
# EVALUATION PIPELINE
###############################################################################


def run_evaluation(
    gt_excel: Path,
    predictions_jsonl: Path,
    output_dir: Path,
    params: dict[str, Any],
    output_files_enabled: dict[str, bool],
    group_terminology: dict[str, str] | None = None,
    exclude_failed: bool = True,
    ontology_matching: bool = False,
    dictionary_excel_path: Path | str | None = None,
) -> None:
    """Run complete evaluation pipeline.

    Loads data, builds analysis DataFrames, computes metrics, performs stratified analyses,
    and generates output reports and visualizations.

    Args:
        gt_excel: Path to Excel file with ground-truth annotations.
        predictions_jsonl: Path to JSONL file with model predictions.
        output_dir: Directory for output files.
        params: Configuration parameters (TOP_K_VALUES, RARE_DIAGNOSIS_THRESHOLD, etc.).
        output_files_enabled: Mapping of output types to boolean flags indicating whether to generate them.
        group_terminology: Mapping of group keys to display names.
        exclude_failed: Whether to exclude failed predictions (NaN scores).
        ontology_matching: Whether to run in ontology matching mode (skips score-based analysis).
        dictionary_excel_path: Path to Excel file containing diagnosis group mappings.

    Raises:
        ValueError: If group_terminology is not provided.
    """
    logger.info("Starting evaluation")
    logger.info("Loading ground-truth from %s", gt_excel)
    logger.info("Loading predictions from %s", predictions_jsonl)

    output_dir.mkdir(parents=True, exist_ok=True)

    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    # Load input data
    gt_df = io_utils.load_ground_truth_excel(gt_excel, params.get("GT_SHEET_NAME"))
    records = io_utils.load_jsonl(predictions_jsonl)
    prediction_map = prediction_extractors.build_prediction_lookup_map(records)

    # Build report-level DataFrame
    logger.info("Building report-level DataFrame")
    report_df = dataframe_builders.build_report_level_dataframe(
        gt_df, prediction_map, params.get("TOP_K_VALUES")
    )

    if exclude_failed and ontology_matching:
        report_df = report_df.dropna(subset=["pred_score"])

    # Compute metrics
    logger.info("Computing metrics")
    report_metrics = metrics_calculators.compute_report_level_metrics(
        report_df, group_terminology
    )

    # Perform stratified analyses
    hard_cases_df = analysis_functions.compute_hard_case_metrics(
        report_df, group_terminology
    )
    error_breakdown = analysis_functions.compute_error_case_breakdown(report_df)
    group1_accuracy_df = analysis_functions.compute_accuracy_by_group(
        report_df, "gt_group1", group_terminology
    )
    group2_accuracy_df = analysis_functions.compute_accuracy_by_group(
        report_df, "gt_group2", group_terminology
    )
    group3_accuracy_df = analysis_functions.compute_accuracy_by_group(
        report_df, "gt_group3", group_terminology
    )

    # Build container-level DataFrame
    logger.info("Building container-level DataFrame")
    container_df = dataframe_builders.build_container_level_dataframe(report_df)

    # Compute container-level group accuracy breakdowns
    if len(container_df) > 0:
        logger.info("Computing container-level accuracy metrics")
        container_group1_accuracy_df = analysis_functions.compute_accuracy_by_group(
            container_df, "gt_container_group1", group_terminology
        )
        container_group2_accuracy_df = analysis_functions.compute_accuracy_by_group(
            container_df, "gt_container_group2", group_terminology
        )
        container_group3_accuracy_df = analysis_functions.compute_accuracy_by_group(
            container_df, "gt_container_group3", group_terminology
        )
    else:
        logger.warning("No container data available; skipping container-level analysis")
        container_df = pd.DataFrame()
        container_group1_accuracy_df = pd.DataFrame()
        container_group2_accuracy_df = pd.DataFrame()
        container_group3_accuracy_df = pd.DataFrame()

    rare_diagnosis_df = analysis_functions.compute_rare_diagnosis_metrics(
        report_df, params.get("RARE_DIAGNOSIS_THRESHOLD"), group_terminology
    )

    # Threshold analysis (skipped in ontology matching mode)
    if ontology_matching:
        threshold_df = analysis_functions.compute_threshold_sweep_metrics(
            report_df, params.get("THRESHOLD_STEPS")
        )

        # Compute AUC for accuracy-coverage tradeoff
        for k in params.get("TOP_K_VALUES"):
            report_metrics[f"threshold_auc_top{k}"] = float(
                auc(
                    threshold_df["coverage"],
                    threshold_df[f"accuracy_top{k}"],
                )
            )
    else:
        threshold_df = None

    # Write all outputs
    logger.info("Writing evaluation results to %s", output_dir)
    _write_evaluation_outputs(
        output_dir=output_dir,
        report_metrics=report_metrics,
        report_df=report_df,
        hard_cases_df=hard_cases_df,
        error_breakdown=error_breakdown,
        rare_diagnosis_df=rare_diagnosis_df,
        threshold_df=threshold_df,
        group1_accuracy_df=group1_accuracy_df,
        group2_accuracy_df=group2_accuracy_df,
        group3_accuracy_df=group3_accuracy_df,
        container_df=container_df,
        container_group1_accuracy_df=container_group1_accuracy_df,
        container_group2_accuracy_df=container_group2_accuracy_df,
        container_group3_accuracy_df=container_group3_accuracy_df,
        output_files_enabled=output_files_enabled,
        group_terminology=group_terminology,
        dictionary_excel_path=dictionary_excel_path,
    )
    logger.info("Evaluation completed successfully")


def _write_evaluation_outputs(
    output_dir: Path,
    report_metrics: dict[str, float],
    report_df: pd.DataFrame,
    hard_cases_df: pd.DataFrame,
    error_breakdown: dict[str, float],
    rare_diagnosis_df: pd.DataFrame,
    threshold_df: pd.DataFrame,
    group1_accuracy_df: pd.DataFrame,
    group2_accuracy_df: pd.DataFrame,
    group3_accuracy_df: pd.DataFrame,
    container_df: pd.DataFrame,
    container_group1_accuracy_df: pd.DataFrame,
    container_group2_accuracy_df: pd.DataFrame,
    container_group3_accuracy_df: pd.DataFrame,
    output_files_enabled: dict[str, bool],
    group_terminology: dict[str, str] | None = None,
    dictionary_excel_path: Path | str | None = None,
) -> None:
    """Write all evaluation outputs to files and generate visualizations.

    Args:
        output_dir: Output directory path.
        report_metrics: Dictionary of computed metrics.
        report_df: Report-level DataFrame.
        hard_cases_df: DataFrame with hard case metrics.
        error_breakdown: Dictionary of error category fractions.
        rare_diagnosis_df: DataFrame with rare diagnosis metrics.
        threshold_df: DataFrame with threshold sweep results.
        group1_accuracy_df: DataFrame with group1 accuracy breakdown.
        group2_accuracy_df: DataFrame with group2 accuracy breakdown.
        group3_accuracy_df: DataFrame with group3 accuracy breakdown.
        container_df: Container-level DataFrame.
        container_group1_accuracy_df: DataFrame with container group1 accuracy breakdown.
        container_group2_accuracy_df: DataFrame with container group2 accuracy breakdown.
        container_group3_accuracy_df: DataFrame with container group3 accuracy breakdown.
        output_files_enabled: Mapping of output types to boolean flags indicating whether to generate them.
        group_terminology: Mapping of group keys to display names.
        dictionary_excel_path: Path to Excel file containing diagnosis group mappings.

    Raises:
        ValueError: If group_terminology is not provided.
    """
    # Define hardcoded output filenames
    OUTPUT_FILENAMES = {
        "OUTPUT_SUMMARY": "evaluation_summary.json",
        "OUTPUT_REPORT_LEVEL": "report_level_metrics.csv",
        "OUTPUT_HARD_CASES": "hard_case_metrics.csv",
        "OUTPUT_ERROR_ANALYSIS": "error_analysis.json",
        "OUTPUT_RARE_DIAGNOSIS": "rare_diagnosis_metrics.csv",
        "OUTPUT_THRESHOLDS": "threshold_metrics.csv",
        "OUTPUT_THRESHOLD_PLOT": "threshold_accuracy_coverage.png",
        "OUTPUT_COVERAGE_ACCURACY_PLOT": "accuracy_coverage_plot.png",
        "OUTPUT_ACCURACY_BREAKDOWN_GROUP_1": "report_accuracy_breakdown_group_1.csv",
        "OUTPUT_ACCURACY_BREAKDOWN_GROUP_2": "report_accuracy_breakdown_group_2.csv",
        "OUTPUT_ACCURACY_BREAKDOWN_GROUP_3": "report_accuracy_breakdown_group_3.csv",
        "OUTPUT_CONFUSION_MATRIX_GROUP_1": "report_confusion_matrix_group_1.png",
        "OUTPUT_CONFUSION_MATRIX_GROUP_2": "report_confusion_matrix_group_2.png",
        "OUTPUT_CONFUSION_MATRIX_GROUP_3": "report_confusion_matrix_group_3.png",
        "OUTPUT_CONTAINER_ACCURACY_BREAKDOWN_GROUP_1": "container_accuracy_breakdown_group_1.csv",
        "OUTPUT_CONTAINER_ACCURACY_BREAKDOWN_GROUP_2": "container_accuracy_breakdown_group_2.csv",
        "OUTPUT_CONTAINER_ACCURACY_BREAKDOWN_GROUP_3": "container_accuracy_breakdown_group_3.csv",
    }
    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    # Write CSV outputs
    if output_files_enabled.get("OUTPUT_REPORT_LEVEL", False):
        output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_REPORT_LEVEL"]
        logger.info("Writing report-level data to %s", output_path)
        report_df.to_csv(output_path, index=False)

    if output_files_enabled.get("OUTPUT_HARD_CASES", False):
        output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_HARD_CASES"]
        hard_cases_df.to_csv(output_path, index=False)

    if output_files_enabled.get("OUTPUT_RARE_DIAGNOSIS", False):
        output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_RARE_DIAGNOSIS"]
        rare_diagnosis_df.to_csv(output_path, index=False)

    # Write group accuracy breakdown files
    accuracy_dfs = {
        "group_1": group1_accuracy_df,
        "group_2": group2_accuracy_df,
        "group_3": group3_accuracy_df,
    }

    for group_key, group_name in group_terminology.items():
        # Map group_key to the corresponding flag name
        flag_name = f"OUTPUT_ACCURACY_BREAKDOWN_{group_key.upper()}"

        # Check if this accuracy breakdown is enabled in the config
        if output_files_enabled.get(flag_name, False):
            sanitized_name = metrics_calculators.sanitize_for_filename(group_name)
            filename = output_dir / f"report_accuracy_breakdown_{sanitized_name}.csv"
            accuracy_dfs[group_key].to_csv(filename, index=False)

    # Write container group accuracy breakdown files (if data is available)
    if len(container_df) > 0:
        container_accuracy_dfs = {
            "group_1": container_group1_accuracy_df,
            "group_2": container_group2_accuracy_df,
            "group_3": container_group3_accuracy_df,
        }

        for group_key, group_name in group_terminology.items():
            # Map group_key to the corresponding flag name for container files
            flag_name = f"OUTPUT_CONTAINER_ACCURACY_BREAKDOWN_{group_key.upper()}"

            # Check if this container accuracy breakdown is enabled in the config
            if output_files_enabled.get(flag_name, False):
                sanitized_name = metrics_calculators.sanitize_for_filename(group_name)
                filename = (
                    output_dir / f"container_accuracy_breakdown_{sanitized_name}.csv"
                )
                container_accuracy_dfs[group_key].to_csv(filename, index=False)

    # Write JSON outputs
    if output_files_enabled.get("OUTPUT_SUMMARY", False):
        output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_SUMMARY"]
        io_utils.write_json(report_metrics, output_path)

    if output_files_enabled.get("OUTPUT_ERROR_ANALYSIS", False):
        output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_ERROR_ANALYSIS"]
        io_utils.write_json(error_breakdown, output_path)

    # Generate threshold analysis plots (if not guidance mode)
    if threshold_df is not None:
        if output_files_enabled.get("OUTPUT_THRESHOLD_PLOT", False):
            output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_THRESHOLD_PLOT"]
            plotting_generators.generate_threshold_accuracy_plot(
                threshold_df,
                output_path,
            )

        if output_files_enabled.get("OUTPUT_COVERAGE_ACCURACY_PLOT", False):
            output_path = output_dir / OUTPUT_FILENAMES["OUTPUT_COVERAGE_ACCURACY_PLOT"]
            plotting_generators.generate_coverage_accuracy_tradeoff_plot(
                threshold_df,
                output_path,
            )

    # Generate confusion matrix plots for each diagnosis group
    for group_key, group_name in group_terminology.items():
        # Map group_key to the corresponding flag name
        flag_name = f"OUTPUT_CONFUSION_MATRIX_{group_key.upper()}"

        # Check if this confusion matrix is enabled in the config
        if output_files_enabled.get(flag_name, False):
            sanitized_name = metrics_calculators.sanitize_for_filename(group_name)
            filename = output_dir / f"report_confusion_matrix_{sanitized_name}.png"

            plotting_generators.generate_diagnosis_group_confusion_matrix(
                report_df,
                group_key,
                filename,
                group_terminology,
                dictionary_excel_path,
            )

    # Generate container confusion matrix plots for each diagnosis group (if data is available)
    if len(container_df) > 0:
        for group_key, group_name in group_terminology.items():
            # Map group_key to the corresponding flag name for container files
            flag_name = f"OUTPUT_CONTAINER_CONFUSION_MATRIX_{group_key.upper()}"

            # Check if this container confusion matrix is enabled in the config
            if output_files_enabled.get(flag_name, False):
                sanitized_name = metrics_calculators.sanitize_for_filename(group_name)
                filename = (
                    output_dir / f"container_confusion_matrix_{sanitized_name}.png"
                )

                # Build a DataFrame with renamed columns for plotting
                plot_df = container_df.copy()
                plot_df["gt_group1"] = plot_df["gt_container_group1"]
                plot_df["gt_group2"] = plot_df["gt_container_group2"]
                plot_df["gt_group3"] = plot_df["gt_container_group3"]

                plotting_generators.generate_diagnosis_group_confusion_matrix(
                    plot_df,
                    group_key,
                    filename,
                    group_terminology,
                    dictionary_excel_path,
                )


def main(run_name: str | None = None) -> None:
    """Main entry point for evaluation.

    Args:
        run_name: Name of the run/experiment. If provided, constructs paths from this.
                 If None, loads from config file.
    """
    config_path = "configs/evaluation.yaml"
    logger.info("Loading evaluation configuration from %s", config_path)
    config = io_utils.load_config(config_path)

    params = config.get("parameters", {})
    output_files_enabled = config.get("output_files", {})
    group_terminology = config.get("group_terminology")
    dictionary_excel_path = config.get("diagnosis_dictionary")
    ontology_matching = config.get("ontology_matching", False)

    # If run_name is provided, construct paths; otherwise load from config
    if run_name:
        predictions_jsonl = Path(f"outputs/{run_name}/ontology_{run_name}.jsonl")
        output_dir = Path(f"outputs/{run_name}/evaluation")
    else:
        logger.error("run_name must be provided to construct input/output paths.")
        raise ValueError("run_name must be provided to construct input/output paths.")

    run_evaluation(
        gt_excel=Path(config["validation_file"]),
        predictions_jsonl=predictions_jsonl,
        output_dir=output_dir,
        params=params,
        output_files_enabled=output_files_enabled,
        group_terminology=group_terminology,
        exclude_failed=params.get("EXCLUDE_FAILED", True),
        ontology_matching=ontology_matching,
        dictionary_excel_path=dictionary_excel_path,
    )


if __name__ == "__main__":
    main()

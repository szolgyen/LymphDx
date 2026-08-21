"""
Performance comparison script.

Loads evaluation results from configured directories, builds a metrics dataframe,
and generates comparison figures using modular figure generators.
"""

import json
from pathlib import Path

import pandas as pd
import yaml

from .comparison_modules import (
    AccuracyGivenCorrectFigure,
    ErrorAnalysisFigure,
    GroupBreakdownMatricesFigure,
    GroupedBarFigure,
    MCCMetricsFigure,
    TableFigure,
    iCLASSiAccuracyFigure,
)

CONFIG_FILE = "configs/comparison.yaml"


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def extract_run_name(path_str: str) -> str:
    """
    Extract run name from evaluation path.

    Example:
        outputs/20260707_173612/evaluation -> 20260707_173612
    """
    parts = Path(path_str).parts
    return parts[-2]


def load_metrics(json_path: str, metrics: list) -> dict:
    """Load specified metrics from a JSON file.

    Args:
        json_path: Path to the JSON file.
        metrics: List of metric names to extract.

    Returns:
        Dictionary mapping metric names to their values (None if not found).
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    result = {}
    for metric in metrics:
        result[metric] = data.get(metric, None)

    return result


def build_dataframe(config: dict) -> pd.DataFrame:
    """
    Build a dataframe from results directories.

    Args:
        config: Configuration dictionary containing results, summary_file, metrics, and name_mappings.

    Returns:
        DataFrame with run names as index and metrics as columns.
    """
    summary_filename = config["summary_file"]
    error_analysis_filename = config.get("error_analysis_file", "error_analysis.json")
    metrics = config["metrics"]
    error_analysis_metrics = config.get("error_analysis_metrics", [])
    name_mappings = config["name_mappings"]

    rows = []
    index = []

    for result_dir in config["results"]:
        # Load evaluation summary metrics
        json_file = Path(result_dir) / summary_filename
        run_metrics = load_metrics(str(json_file), metrics)

        # Load error analysis metrics if any are configured
        if error_analysis_metrics:
            error_analysis_file = Path(result_dir) / error_analysis_filename
            error_metrics = load_metrics(
                str(error_analysis_file), error_analysis_metrics
            )
            run_metrics.update(error_metrics)

        rows.append(run_metrics)
        index.append(extract_run_name(result_dir))

    df = pd.DataFrame(rows, index=index)

    # Apply name mappings
    df.rename(
        index=name_mappings,
        columns=name_mappings,
        inplace=True,
    )

    return df


def generate_figures(df: pd.DataFrame, config: dict) -> None:
    """
    Generate all configured figures.

    Args:
        df: DataFrame containing performance metrics.
        config: Configuration dictionary containing figures settings.
    """
    figures_config = config.get("figures", {})
    name_mappings = config.get("name_mappings", {})

    if "table" in figures_config:
        table_fig = TableFigure(figures_config["table"])
        table_fig.generate(df, name_mappings)

    if "grouped_bar" in figures_config:
        bar_fig = GroupedBarFigure(figures_config["grouped_bar"])
        bar_fig.generate(df, name_mappings)

    if "mcc_metrics" in figures_config:
        mcc_fig = MCCMetricsFigure(figures_config["mcc_metrics"])
        mcc_fig.generate(df, name_mappings)

    if "accuracy_given_correct" in figures_config:
        agc_fig = AccuracyGivenCorrectFigure(figures_config["accuracy_given_correct"])
        agc_fig.generate(df, name_mappings)

    if "iclassi_accuracy" in figures_config:
        iclassi_fig = iCLASSiAccuracyFigure(figures_config["iclassi_accuracy"])
        iclassi_fig.generate(df, name_mappings)

    if "error_analysis" in figures_config:
        error_fig = ErrorAnalysisFigure(figures_config["error_analysis"])
        error_fig.generate(df, name_mappings)

    if "group_breakdown_matrices" in figures_config:
        dictionary_path = config.get("diagnosis_dictionary")
        group_fig = GroupBreakdownMatricesFigure(
            figures_config["group_breakdown_matrices"]
        )
        group_fig.generate(df, name_mappings, dictionary_path=dictionary_path)


def main():
    """Main orchestration function."""
    config = load_config(CONFIG_FILE)
    df = build_dataframe(config)
    generate_figures(df, config)


if __name__ == "__main__":
    main()

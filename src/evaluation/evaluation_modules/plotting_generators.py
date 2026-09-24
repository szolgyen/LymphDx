"""Plotting generators for visualization of evaluation results.

Creates publication-ready figures for threshold analysis and confusion matrices.
"""

import logging
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from . import metrics_calculators as metrics

logger = logging.getLogger(__name__)


def generate_threshold_accuracy_plot(
    threshold_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate dual-axis plot showing accuracy and coverage vs. threshold.

    Creates a plot with:
    - Left y-axis: Top-1, Top-3, Top-5 accuracy (blue)
    - Right y-axis: Coverage (red)
    - X-axis: Prediction score threshold

    Args:
        threshold_df: DataFrame with threshold, coverage, and accuracy columns.
        output_path: Path to save the generated PNG file.
    """
    fig, ax_accuracy = plt.subplots(figsize=(8, 6))

    # Plot accuracy metrics on left y-axis
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

    # Plot coverage on right y-axis
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


def generate_coverage_accuracy_tradeoff_plot(
    threshold_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate plot showing accuracy-coverage tradeoff curve.

    Creates a plot showing how accuracy varies with coverage for different
    top-k predictions, illustrating the accuracy-coverage tradeoff.

    Args:
        threshold_df: DataFrame with coverage and accuracy columns.
        output_path: Path to save the generated PNG file.
    """
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


def _build_label_to_major_mapping_from_codes(
    report_df: pd.DataFrame,
    group_key: str,
    major_group: str,
) -> dict[str, str]:
    """Build a mapping from diagnosis labels to their major_group classification.

    Uses the actual diagnosis codes and their direct group mappings from code_to_groups,
    avoiding generalized group-to-group mappings that may lose code-level specificity.
    Checks both GT and predicted columns to handle predicted-only labels.

    Args:
        report_df: Report-level DataFrame containing gt_code, pred_code, and group columns.
        group_key: Source group key (e.g., "group_2").
        major_group: Target major group key (e.g., "group_4").

    Returns:
        Dictionary mapping label values to their major_group values.
    """
    mapping = {}
    # Normalize group keys by removing underscore (group_1 -> group1)
    group_normalized = group_key.replace("_", "")
    major_normalized = major_group.replace("_", "")

    gt_group_column = f"gt_{group_normalized}"
    pred_group_column = f"pred_{group_normalized}"
    gt_major_column = f"gt_{major_normalized}"
    pred_major_column = f"pred_{major_normalized}"

    # Collect labels from both GT and predicted columns
    all_group_labels = set()
    if gt_group_column in report_df.columns:
        all_group_labels.update(report_df[gt_group_column].dropna().unique())
    if pred_group_column in report_df.columns:
        all_group_labels.update(report_df[pred_group_column].dropna().unique())

    if not all_group_labels:
        logger.warning(f"No labels found in {gt_group_column} or {pred_group_column}")
        return mapping

    # For each unique label, find its major_group values from both GT and predicted rows
    for label in all_group_labels:
        major_values = []

        # Look in GT rows where this label appears in GT column
        if (
            gt_group_column in report_df.columns
            and gt_major_column in report_df.columns
        ):
            subset_gt = report_df[report_df[gt_group_column] == label]
            major_values.extend(subset_gt[gt_major_column].dropna().unique().tolist())

        # Look in predicted rows where this label appears in pred column
        if (
            pred_group_column in report_df.columns
            and pred_major_column in report_df.columns
        ):
            subset_pred = report_df[report_df[pred_group_column] == label]
            major_values.extend(
                subset_pred[pred_major_column].dropna().unique().tolist()
            )

        if not major_values:
            continue

        if len(major_values) == 1:
            # Simple case: all codes with this label map to the same major_group
            mapping[label] = major_values[0]
        else:
            # Multiple major_group values for this label; use most common
            from collections import Counter

            major_counts = Counter(major_values)
            most_common = major_counts.most_common(1)[0][0]
            mapping[label] = most_common

            logger.debug(
                f"Label '{label}' has multiple major_group mappings: {major_counts}; "
                f"using most common: {most_common}"
            )

    return mapping


def _get_major_group_categories_and_colors(
    source_to_major_mapping: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Extract unique major group categories from mapping and generate colors.

    Args:
        source_to_major_mapping: Mapping from source group to major group categories.

    Returns:
        Tuple of (ordered_major_categories, colors_map).
    """
    # Extract unique major group categories in order of first appearance
    unique_major = []
    seen = set()
    for major in source_to_major_mapping.values():
        if major not in seen:
            unique_major.append(major)
            seen.add(major)

    # Generate colors dynamically for all categories
    color_palette = [
        "#d62728",  # Red (Malignant)
        "#2ca02c",  # Green (Reactive)
        "#ff7f0e",  # Orange (Infectious)
        "#9467bd",  # Purple (Miscellaneous)
        "#1f77b4",  # Blue
        "#17becf",  # Cyan
        "#bcbd22",  # Yellow-green
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
    ]

    colors_map = {}
    for i, major in enumerate(unique_major):
        colors_map[major] = color_palette[i % len(color_palette)]

    return unique_major, colors_map


def generate_diagnosis_group_confusion_matrix(
    report_df: pd.DataFrame,
    group_key: str,
    output_path: Path,
    group_terminology: dict[str, str] | None = None,
    dictionary_excel_path: Path | str | None = None,
    major_group: str | None = None,
) -> None:
    """Generate normalized confusion matrix heatmap for a diagnosis group.

    Creates a confusion matrix visualization with special handling for grouping by major_group:
    - Sorts categories by their actual underlying code's group classification
    - Color-codes labels by major_group category
    - Adds separators between major_group regions
    - Includes legend for disease behavior categories

    Args:
        report_df: Report-level DataFrame.
        group_key: Group key to analyze (e.g., "group_1", "group_2", "group_3", "group_4").
        output_path: Path to save the generated PNG file.
        group_terminology: Mapping of group keys to display names.
        dictionary_excel_path: Path to the Excel file containing diagnosis group mappings.
        major_group: Group to use for organizing axis labels and colors (e.g., "group_3", "group_4").
                    Defaults to "group_3" if not specified.
        code_to_groups: Mapping of diagnosis codes to their group classifications.

    Raises:
        ValueError: If group_terminology is not provided.
    """
    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    # Default to group_3 if major_group not specified
    if major_group is None:
        major_group = "group_3"

    # Get display name for this group
    group_display_name = group_terminology.get(group_key, group_key)

    # Normalize group key for column access (e.g., "group_2" -> "group2")
    group_normalized = metrics.normalize_group_key(group_key)

    # Map group_normalized to Excel column name (e.g., "group2" -> "WHO-like Categories")
    group_column_mapping = {
        "group1": "WHO-like Subcategories",
        "group2": "WHO-like Categories",
        "group3": "WHO-like Major Sections/Lineages",
        "group4": "Diagnostic group 4",
    }
    excel_column_name = group_column_mapping.get(group_normalized)

    if not excel_column_name:
        raise ValueError(f"Unknown group normalized key: {group_normalized}")

    # Load the Excel dictionary to map codes to groups
    if not dictionary_excel_path:
        raise ValueError("dictionary_excel_path is required for confusion matrix generation")

    try:
        code_dict_df = pd.read_excel(dictionary_excel_path)
    except Exception as e:
        logger.error(f"Failed to load dictionary Excel file: {e}")
        raise

    # Build mapping from code to group category
    code_to_group = {}
    for _, row in code_dict_df.iterrows():
        code = row.get("Code")
        group_value = row.get(excel_column_name)
        if pd.notna(code) and pd.notna(group_value):
            code_to_group[int(code) if isinstance(code, float) else code] = group_value

    # Get GT codes and prediction codes, filtering to valid rows
    valid_mask = (
        report_df["gt_code"].notna()
        & report_df["pred_code"].notna()
    )

    gt_codes = report_df.loc[valid_mask, "gt_code"].tolist()
    pred_codes = report_df.loc[valid_mask, "pred_code"].tolist()

    # Map codes to their group categories using the dictionary
    y_true = [code_to_group.get(code, "N/A") for code in gt_codes]
    y_pred = [code_to_group.get(code, "N/A") for code in pred_codes]

    # Collect all unique categories from both GT and predictions
    all_labels = sorted(set(y_true) | set(y_pred))

    # Build mapping from current group to major_group using actual code classifications
    source_to_major_mapping = {}
    major_order = []
    colors_map = {}

    if group_key != major_group:
        # Use actual code classifications instead of generalized group mappings
        source_to_major_mapping = _build_label_to_major_mapping_from_codes(
            report_df, group_key, major_group
        )

    if source_to_major_mapping:
        major_order, colors_map = _get_major_group_categories_and_colors(
            source_to_major_mapping
        )

        # Sort labels by their major_group category
        def get_sort_key(label: str) -> tuple:
            major = source_to_major_mapping.get(label, "N/A")
            try:
                major_priority = major_order.index(major)
            except ValueError:
                major_priority = len(major_order)
            return (major_priority, label)

        all_labels = sorted(all_labels, key=get_sort_key)

    if "N/A" in all_labels:
        all_labels.remove("N/A")
        all_labels.append("N/A")

    # # Compute normalized confusion matrix
    # cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize="true")
    # cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    # annot = cm_df.map(lambda x: "0" if x == 0 else f"{x:.2f}")
    # # annot = cm_df.map(lambda x: "" if x == 0 else f"{x:d}")

    # # TODO
    # Normalized matrix — used for colors
    cm_normalized = confusion_matrix(
        y_true,
        y_pred,
        labels=all_labels,
        normalize="true",
    )
    cm_normalized_df = pd.DataFrame(
        cm_normalized,
        index=all_labels,
        columns=all_labels,
    )

    # Raw matrix — used for annotations
    cm_counts = confusion_matrix(
        y_true,
        y_pred,
        labels=all_labels,
    )
    cm_counts_df = pd.DataFrame(
        cm_counts,
        index=all_labels,
        columns=all_labels,
    )

    # Display actual counts in the cells
    annot = cm_counts_df.map(lambda x: f"{x:d}")
    # TODO

    # from matplotlib.colors import LogNorm

    # Create figure and heatmap
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        cm_normalized_df,       # <-- controls the colors
        annot=annot,            # <-- displays the raw counts
        fmt="",
        cmap="Blues",
        ax=ax,
        annot_kws={"fontsize": 12},
        cbar_kws={
            "location": "left",
            "shrink": 1.0,
            "pad": 0.02,
        },
    )

    cbar = ax.collections[-1].colorbar
    cbar.set_label("Accuracy", fontsize=12)
    cbar.ax.tick_params(labelsize=12)

    # Flip y-axis to show categories right-aligned
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", labelleft=False, labelright=True)

    # Add boxes around diagonal blocks for mapped groups
    if source_to_major_mapping:
        major_regions = {}
        for idx, label in enumerate(all_labels):
            major = source_to_major_mapping.get(label, "Miscellaneous")
            if major not in major_regions:
                major_regions[major] = [idx, idx]
            else:
                major_regions[major][1] = idx

        # Draw rectangle for each major_group block
        for major, (start, end) in major_regions.items():
            size = end - start + 1
            rect = Rectangle(
                (start, start),
                size,
                size,
                fill=False,
                edgecolor="black",
                linewidth=3,
                clip_on=False,
            )
            ax.add_patch(rect)

    # Add special formatting for group_1 and group_2 when grouped by major_group
    if group_key in ("group_1", "group_2") and source_to_major_mapping:
        major_group_display = group_terminology.get(major_group, major_group)
        _apply_group_formatting(
            ax,
            all_labels,
            source_to_major_mapping,
            major_order,
            colors_map,
            major_group_display,
        )

    # Wrap long labels for readability
    ax.set_yticklabels(
        [textwrap.fill(label.get_text(), width=30) for label in ax.get_yticklabels()],
        fontsize=10,
        rotation=0,
    )
    ax.set_xticklabels(
        [textwrap.fill(label.get_text(), width=30) for label in ax.get_xticklabels()],
        fontsize=10,
        rotation=90,
        ha="right",
    )

    plt.title(f"Confusion Matrix for {group_display_name}", fontsize=20)
    plt.ylabel("True Label", fontsize=15)
    plt.xlabel("Predicted Label", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _apply_group_formatting(
    ax: plt.Axes,
    all_labels: list[str],
    source_to_major_mapping: dict[str, str],
    major_order: list[str],
    colors_map: dict[str, str],
    major_group_display: str = "Major Group",
) -> None:
    """Apply group-specific formatting: colors, separators, and legend.

    Args:
        ax: Matplotlib axes to format.
        all_labels: Sorted list of all group labels.
        source_to_major_mapping: Mapping from source group labels to major group categories.
        major_order: Ordered list of major group categories.
        colors_map: Mapping from major group categories to hex colors.
        major_group_display: Display name for the major group (used in legend title).
    """
    # Color y-axis (true) labels
    for label in ax.get_yticklabels():
        label_text = label.get_text()
        if label_text:
            major = source_to_major_mapping.get(
                label_text, major_order[-1] if major_order else "Miscellaneous"
            )
            color = colors_map.get(major, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Color x-axis (predicted) labels
    for label in ax.get_xticklabels():
        label_text = label.get_text()
        if label_text:
            major = source_to_major_mapping.get(
                label_text, major_order[-1] if major_order else "Miscellaneous"
            )
            color = colors_map.get(major, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Add vertical separators between major group categories
    current_major = None
    for idx, label in enumerate(all_labels):
        major = source_to_major_mapping.get(
            label, major_order[-1] if major_order else "Miscellaneous"
        )
        if current_major is not None and major != current_major:
            ax.axvline(x=idx, color="black", linewidth=0.5, linestyle="--")
        current_major = major

    # Add horizontal separators between major group categories
    current_major = None
    for idx, label in enumerate(all_labels):
        major = source_to_major_mapping.get(
            label, major_order[-1] if major_order else "Miscellaneous"
        )
        if current_major is not None and major != current_major:
            ax.axhline(y=idx, color="black", linewidth=0.5, linestyle="--")
        current_major = major

    # Add legend for major group categories
    legend_elements = [
        Patch(facecolor=colors_map[major], label=major)
        for major in major_order
        if major
        in [
            source_to_major_mapping.get(
                name, major_order[-1] if major_order else "Miscellaneous"
            )
            for name in all_labels
        ]
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.97, -0.19),
        frameon=True,
        title_fontsize=12,
        title="Classification",  # major_group_display,
        fontsize=12,
    )

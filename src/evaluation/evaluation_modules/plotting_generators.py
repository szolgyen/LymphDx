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


def _load_group2_to_group3_mapping(dictionary_excel_path: Path | str) -> dict[str, str]:
    """Load mapping between diagnosis group 2 and group 3 from Excel reference file.

    Args:
        dictionary_excel_path: Path to the Excel file containing diagnosis group mappings.

    Returns:
        Dictionary mapping group 2 categories to group 3 categories.
    """
    mapping = {}
    excel_path = Path(dictionary_excel_path)

    if excel_path.exists():
        try:
            df_mapping = pd.read_excel(excel_path)
            if (
                "Diagnostic group 2" in df_mapping.columns
                and "Diagnostic group 3" in df_mapping.columns
            ):
                for idx, row in df_mapping.iterrows():
                    g2 = row["Diagnostic group 2"]
                    g3 = row["Diagnostic group 3"]
                    if pd.notna(g2) and pd.notna(g3):
                        mapping[str(g2).strip()] = str(g3).strip()
        except Exception as e:
            logger.warning(f"Could not load mapping from {excel_path}: {e}")
    else:
        logger.warning(f"Excel dictionary file not found at {excel_path}")

    return mapping


def generate_diagnosis_group_confusion_matrix(
    report_df: pd.DataFrame,
    group_key: str,
    output_path: Path,
    group_terminology: dict[str, str] | None = None,
    dictionary_excel_path: Path | str | None = None,
) -> None:
    """Generate normalized confusion matrix heatmap for a diagnosis group.

    Creates a confusion matrix visualization with special handling for group_2:
    - Sorts categories by group_3 disease behavior
    - Color-codes labels by group_3 category
    - Adds separators between group_3 regions
    - Includes legend for disease behavior categories

    Args:
        report_df: Report-level DataFrame.
        group_key: Group key to analyze (e.g., "group_1", "group_2", "group_3").
        output_path: Path to save the generated PNG file.
        group_terminology: Mapping of group keys to display names.
        dictionary_excel_path: Path to the Excel file containing diagnosis group mappings.

    Raises:
        ValueError: If group_terminology is not provided.
    """
    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    # Get display name for this group
    group_display_name = group_terminology.get(group_key, group_key)

    # Normalize group key for column access
    group_normalized = metrics.normalize_group_key(group_key)

    # Filter to rows with both GT and prediction present
    valid_mask = (
        report_df[f"gt_{group_normalized}"].notna()
        & report_df[f"pred_{group_normalized}"].notna()
    )

    y_true = report_df.loc[valid_mask, f"gt_{group_normalized}"].tolist()
    y_pred = report_df.loc[valid_mask, f"pred_{group_normalized}"].tolist()

    # Collect all unique categories from both GT and predictions
    all_labels = sorted(set(y_true) | set(y_pred))

    # For group_2, load mapping and customize sorting/appearance
    g2_to_g3_mapping = {}
    if group_key == "group_2" and dictionary_excel_path:
        g2_to_g3_mapping = _load_group2_to_group3_mapping(dictionary_excel_path)
        if g2_to_g3_mapping:
            # Define disease behavior category order
            group3_order = [
                "Malignant/neoplastic",
                "Reactive/inflammatory",
                "Infectious Lymphadenitis",
                "Miscellaneous",
            ]

            # Sort labels by their group_3 category
            def get_sort_key(label: str) -> tuple:
                group3 = g2_to_g3_mapping.get(label, "Miscellaneous")
                try:
                    g3_priority = group3_order.index(group3)
                except ValueError:
                    g3_priority = len(group3_order)
                return (g3_priority, label)

            all_labels = sorted(all_labels, key=get_sort_key)

    # Compute normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=all_labels, normalize="true")
    cm_df = pd.DataFrame(cm, index=all_labels, columns=all_labels)
    annot = cm_df.map(lambda x: "0" if x == 0 else f"{x:.2f}")

    # Create figure and heatmap
    fig, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(
        cm_df,
        annot=annot,
        fmt="",
        cmap="Blues",
        ax=ax,
        cbar_kws={
            "location": "left",
            "shrink": 1.0,
            "pad": 0.02,
        },
    )

    # Flip y-axis to show categories right-aligned
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", labelleft=False, labelright=True)

    # Add boxes around diagonal blocks for group_2
    if g2_to_g3_mapping:
        group3_regions = {}
        for idx, label in enumerate(all_labels):
            group3 = g2_to_g3_mapping.get(label, "Miscellaneous")
            if group3 not in group3_regions:
                group3_regions[group3] = [idx, idx]
            else:
                group3_regions[group3][1] = idx

        # Draw rectangle for each group3 block
        for group3, (start, end) in group3_regions.items():
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

    # Add special formatting for group_2
    if group_key == "group_2" and g2_to_g3_mapping:
        _apply_group2_formatting(ax, all_labels, g2_to_g3_mapping)

    # Wrap long labels for readability
    ax.set_yticklabels(
        [textwrap.fill(label.get_text(), width=30) for label in ax.get_yticklabels()],
        fontsize=8,
        rotation=0,
    )
    ax.set_xticklabels(
        [textwrap.fill(label.get_text(), width=30) for label in ax.get_xticklabels()],
        fontsize=8,
        rotation=45,
        ha="right",
    )

    plt.title(f"Confusion Matrix for {group_display_name}", fontsize=15)
    plt.ylabel("True Label", fontsize=15)
    plt.xlabel("Predicted Label", fontsize=15)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def _apply_group2_formatting(
    ax: plt.Axes,
    all_labels: list[str],
    g2_to_g3_mapping: dict[str, str],
) -> None:
    """Apply group_2-specific formatting: colors, separators, and legend.

    Args:
        ax: Matplotlib axes to format.
        all_labels: Sorted list of all group_2 labels.
        g2_to_g3_mapping: Mapping from group_2 labels to group_3 categories.
    """
    group3_order = [
        "Malignant/neoplastic",
        "Reactive/inflammatory",
        "Infectious Lymphadenitis",
        "Miscellaneous",
    ]
    colors_map = {
        "Malignant/neoplastic": "#d62728",
        "Reactive/inflammatory": "#2ca02c",
        "Infectious Lymphadenitis": "#ff7f0e",
        "Miscellaneous": "#9467bd",
    }

    # Assign colors to any unknown categories
    unknown_categories = set(g2_to_g3_mapping.values()) - set(colors_map.keys())
    color_palette = ["#1f77b4", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
    for i, g3 in enumerate(sorted(unknown_categories)):
        colors_map[g3] = color_palette[i % len(color_palette)]

    # Color y-axis (true) labels
    for label in ax.get_yticklabels():
        label_text = label.get_text()
        if label_text:
            group3 = g2_to_g3_mapping.get(label_text, "Miscellaneous")
            color = colors_map.get(group3, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Color x-axis (predicted) labels
    for label in ax.get_xticklabels():
        label_text = label.get_text()
        if label_text:
            group3 = g2_to_g3_mapping.get(label_text, "Miscellaneous")
            color = colors_map.get(group3, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Add vertical separators between group_3 categories
    current_g3 = None
    for idx, label in enumerate(all_labels):
        group3 = g2_to_g3_mapping.get(label, "Miscellaneous")
        if current_g3 is not None and group3 != current_g3:
            ax.axvline(x=idx, color="black", linewidth=0.5, linestyle="--")
        current_g3 = group3

    # Add horizontal separators between group_3 categories
    current_g3 = None
    for idx, label in enumerate(all_labels):
        group3 = g2_to_g3_mapping.get(label, "Miscellaneous")
        if current_g3 is not None and group3 != current_g3:
            ax.axhline(y=idx, color="black", linewidth=0.5, linestyle="--")
        current_g3 = group3

    # Add legend for disease behavior categories
    legend_elements = [
        Patch(facecolor=colors_map[g3], label=g3)
        for g3 in group3_order
        if g3 in [g2_to_g3_mapping.get(name, "Miscellaneous") for name in all_labels]
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.97, -0.19),
        frameon=True,
        title="Disease Behavior",
        fontsize=10,
    )

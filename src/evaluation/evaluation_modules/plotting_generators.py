"""Plotting generators for visualization of evaluation results.

Creates publication-ready figures for threshold analysis and confusion matrices.
"""

import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

from . import metrics_calculators as metrics

from ..utils import (
    build_label_to_major_mapping,
    get_major_group_categories_and_colors,
    apply_major_group_formatting,
    sort_labels_by_major_group
)

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


def generate_diagnosis_group_confusion_matrix(
        report_df: pd.DataFrame,
        group_key: str,
        output_path: Path,
        group_terminology: dict[str, str] | None = None,
        major_group: str | None = None,
    ) -> None:
    """Generate normalized confusion matrix heatmap for a diagnosis group.

    Args:
        report_df: Report-level DataFrame.
        group_key: Group key to analyze (e.g., "group_1", "group_2", "group_3", "group_4").
        output_path: Path to save the generated PNG file.
        group_terminology: Mapping of group keys to display names.
        major_group: Group to use for organizing axis labels and colors (e.g., "group_3", "group_4").

    Raises:
        ValueError: If group_terminology is not provided.
    """
    if group_terminology is None:
        logger.error("group_terminology must be provided in the configuration.")
        raise ValueError("group_terminology must be provided in the configuration.")

    
    if major_group is None:
        major_group = "group_4"

    # Get display name for this group
    group_display_name = group_terminology.get(group_key, group_key)

    # Normalize group key for column access (e.g., "group_2" -> "group2")
    group_normalized = metrics.normalize_group_key(group_key)

    if group_normalized == "group4":
        y_true = report_df["gt_group4"].fillna("N/A").astype(str).str.strip().tolist()
        y_pred = report_df["pred_group4"].fillna("N/A").astype(str).str.strip().tolist()

        source_to_major_mapping = {
            label: label
            for label in set(y_true) | set(y_pred)
            if label != "N/A"
        }

    else:
        group_column = f"gt_{group_normalized}"
        pred_group_column = f"pred_{group_normalized}"

        y_true = report_df[group_column].fillna("N/A").astype(str).str.strip().tolist()
        y_pred = report_df[pred_group_column].fillna("N/A").astype(str).str.strip().tolist()

        source_to_major_mapping = build_label_to_major_mapping(
            report_df,
            group_key,
            major_group,
        )

    all_labels = sorted(set(y_true) | set(y_pred))

    major_order = []
    colors_map = {}

    if source_to_major_mapping:
        major_order, colors_map = get_major_group_categories_and_colors(
            source_to_major_mapping
        )

        # Sort labels by their major_group category
        all_labels = sort_labels_by_major_group(
            all_labels,
            source_to_major_mapping,
        )

    if "N/A" in all_labels:
        all_labels.remove("N/A")
        all_labels.append("N/A")

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
            major = source_to_major_mapping.get(label)

            if major is None:
                logger.warning("No major-group mapping found for label: %r", label)
                major = "N/A"

            major = major.strip()

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

    # Add special formatting grouped by major_group
    if group_key in ("group_0", "group_1", "group_2", "group_3", "group_4") and source_to_major_mapping:
        apply_major_group_formatting(
            ax=ax,
            labels=all_labels,
            mapping=source_to_major_mapping,
            legend_title="Classification",
            add_vertical_separators=True,
            add_horizontal_separators=True,
            legend_kwargs={
                "loc": "lower left",
                "bbox_to_anchor": (0.97, -0.19),
                "fontsize": 12,
                "title_fontsize": 12,
            },
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

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


def _load_group_to_group3_mapping(
    dictionary_excel_path: Path | str,
    source_column: str,
    target_column: str = "WHO-like Major Sections/Lineages",
) -> dict[str, str]:
    """Load mapping from a diagnosis group to group 3 from Excel reference file.

    Args:
        dictionary_excel_path: Path to the Excel file containing diagnosis group mappings.
        source_column: Column name to map from (e.g., "WHO-like Categories").
        target_column: Column name to map to (default: "WHO-like Major Sections/Lineages").

    Returns:
        Dictionary mapping source group categories to group 3 categories.
    """
    mapping = {}
    excel_path = Path(dictionary_excel_path)

    if excel_path.exists():
        try:
            df_mapping = pd.read_excel(excel_path)
            if (
                source_column in df_mapping.columns
                and target_column in df_mapping.columns
            ):
                for idx, row in df_mapping.iterrows():
                    source = row[source_column]
                    target = row[target_column]
                    if pd.notna(source) and pd.notna(target):
                        mapping[str(source).strip()] = str(target).strip()
        except Exception as e:
            logger.warning(
                f"Could not load mapping from {excel_path} ({source_column} -> {target_column}): {e}"
            )
    else:
        logger.warning(f"Excel dictionary file not found at {excel_path}")

    return mapping


def _load_group2_to_group3_mapping(dictionary_excel_path: Path | str) -> dict[str, str]:
    """Load mapping between diagnosis group 2 and group 3 from Excel reference file.

    Args:
        dictionary_excel_path: Path to the Excel file containing diagnosis group mappings.

    Returns:
        Dictionary mapping group 2 categories to group 3 categories.
    """
    return _load_group_to_group3_mapping(
        dictionary_excel_path,
        source_column="WHO-like Categories",
        target_column="WHO-like Major Sections/Lineages",
    )


def _get_group3_categories_and_colors(
    g2_to_g3_mapping: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    """Extract unique group3 categories from mapping and generate colors.

    Args:
        g2_to_g3_mapping: Mapping from group2 to group3 categories.

    Returns:
        Tuple of (ordered_group3_categories, colors_map).
    """
    # Extract unique group3 categories in order of first appearance
    unique_g3 = []
    seen = set()
    for g3 in g2_to_g3_mapping.values():
        if g3 not in seen:
            unique_g3.append(g3)
            seen.add(g3)

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
    for i, g3 in enumerate(unique_g3):
        colors_map[g3] = color_palette[i % len(color_palette)]

    return unique_g3, colors_map


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

    # Load group-to-group3 mapping and customize sorting/appearance
    group_to_g3_mapping = {}
    group3_order = []
    colors_map = {}

    if dictionary_excel_path:
        if group_key == "group_1":
            group_to_g3_mapping = _load_group_to_group3_mapping(
                dictionary_excel_path,
                source_column="WHO-like Subcategories",
                target_column="WHO-like Major Sections/Lineages",
            )
        elif group_key == "group_2":
            group_to_g3_mapping = _load_group2_to_group3_mapping(dictionary_excel_path)

    if group_to_g3_mapping:
        group3_order, colors_map = _get_group3_categories_and_colors(
            group_to_g3_mapping
        )

        # Sort labels by their group_3 category
        def get_sort_key(label: str) -> tuple:
            group3 = group_to_g3_mapping.get(label, "Miscellaneous")
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

    # Add boxes around diagonal blocks for group_1 and group_2
    if group_to_g3_mapping:
        group3_regions = {}
        for idx, label in enumerate(all_labels):
            group3 = group_to_g3_mapping.get(label, "Miscellaneous")
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

    # Add special formatting for group_1 and group_2
    if group_key in ("group_1", "group_2") and group_to_g3_mapping:
        _apply_group_formatting(
            ax, all_labels, group_to_g3_mapping, group3_order, colors_map
        )

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


def _apply_group_formatting(
    ax: plt.Axes,
    all_labels: list[str],
    group_to_g3_mapping: dict[str, str],
    group3_order: list[str],
    colors_map: dict[str, str],
) -> None:
    """Apply group-specific formatting: colors, separators, and legend.

    Args:
        ax: Matplotlib axes to format.
        all_labels: Sorted list of all group labels.
        group_to_g3_mapping: Mapping from group labels to group_3 categories.
        group3_order: Ordered list of group_3 categories.
        colors_map: Mapping from group_3 categories to hex colors.
    """
    # Color y-axis (true) labels
    for label in ax.get_yticklabels():
        label_text = label.get_text()
        if label_text:
            group3 = group_to_g3_mapping.get(
                label_text, group3_order[-1] if group3_order else "Miscellaneous"
            )
            color = colors_map.get(group3, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Color x-axis (predicted) labels
    for label in ax.get_xticklabels():
        label_text = label.get_text()
        if label_text:
            group3 = group_to_g3_mapping.get(
                label_text, group3_order[-1] if group3_order else "Miscellaneous"
            )
            color = colors_map.get(group3, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Add vertical separators between group_3 categories
    current_g3 = None
    for idx, label in enumerate(all_labels):
        group3 = group_to_g3_mapping.get(
            label, group3_order[-1] if group3_order else "Miscellaneous"
        )
        if current_g3 is not None and group3 != current_g3:
            ax.axvline(x=idx, color="black", linewidth=0.5, linestyle="--")
        current_g3 = group3

    # Add horizontal separators between group_3 categories
    current_g3 = None
    for idx, label in enumerate(all_labels):
        group3 = group_to_g3_mapping.get(
            label, group3_order[-1] if group3_order else "Miscellaneous"
        )
        if current_g3 is not None and group3 != current_g3:
            ax.axhline(y=idx, color="black", linewidth=0.5, linestyle="--")
        current_g3 = group3

    # Add legend for disease behavior categories
    legend_elements = [
        Patch(facecolor=colors_map[g3], label=g3)
        for g3 in group3_order
        if g3
        in [
            group_to_g3_mapping.get(
                name, group3_order[-1] if group3_order else "Miscellaneous"
            )
            for name in all_labels
        ]
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(0.97, -0.19),
        frameon=True,
        title="WHO-like Major Sections/Lineages",
        fontsize=10,
    )

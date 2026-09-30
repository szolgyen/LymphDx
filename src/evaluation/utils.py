
import logging
from collections import Counter
from pathlib import Path

import pandas as pd
from matplotlib.patches import Patch

logger = logging.getLogger(__name__)


def load_report_df(result_dir: str | Path) -> pd.DataFrame | None:
    report_path = Path(result_dir) / "report_level_metrics.csv"

    if not report_path.exists():
        logger.warning(f"Report dataframe not found: {report_path}")
        return None

    return pd.read_csv(report_path)


def build_label_to_major_mapping(
        report_df: pd.DataFrame,
        group_key: str,
        major_group: str,
    ) -> dict[str, str]:
    """Build a mapping from diagnosis labels to their major_group classification.

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
            mapping[label] = str(major_values[0]).strip()
        else:
            # Multiple major_group values for this label; use most common
            major_counts = Counter(major_values)
            most_common = major_counts.most_common(1)[0][0]
            mapping[label] = str(most_common).strip()

            logger.debug(
                f"Label '{label}' has multiple major_group mappings: {major_counts}; "
                f"using most common: {most_common}"
            )

    return mapping


def get_major_group_categories_and_colors(
        source_to_major_mapping: dict[str, str],
    ) -> tuple[list[str], dict[str, str]]:
    """Extract unique major group categories from mapping and generate colors.

    Args:
        source_to_major_mapping: Mapping from source group to major group categories.

    Returns:
        Tuple of (ordered_major_categories, colors_map).
    """
    # Define preferred order for major categories
    preferred_order = [
        "Malignant/neoplastic",
        "Reactive/Inflammatory",
        "Infectious Lymphadenitis",
        "Vascular/hamartomatous",
        "Miscellaneous",
    ]

    # Extract unique major group categories from the mapping
    unique_major = {item.strip() for item in source_to_major_mapping.values()}

    # Sort: preferred categories first (in order), then any others alphabetically
    ordered_major = []
    for category in preferred_order:
        if category in unique_major:
            ordered_major.append(category)
            unique_major.remove(category)

    # Add any remaining categories not in preferred order
    ordered_major.extend(sorted(unique_major))

    # Generate colors dynamically for all categories
    color_palette = [
        "#d62728",  # Red (Malignant)
        "#1f77b4",  # Blue (Reactive)
        "#2ca02c",  # Green (Infectious)
        "#ff7f0e",  # Orange (Vascular)
        "#9467bd",  # Purple (Miscellaneous)
        "#17becf",  # Cyan
        "#bcbd22",  # Yellow-green
        "#e377c2",  # Pink
        "#7f7f7f",  # Gray
    ]

    colors_map = {}
    for i, major in enumerate(ordered_major):
        colors_map[major] = color_palette[i % len(color_palette)]

    return ordered_major, colors_map


def sort_labels_by_major_group(
        labels: list[str],
        mapping: dict[str, str],
    ) -> list[str]:
    """Sort labels by major-group category, then alphabetically."""

    if not mapping:
        return sorted(labels)

    major_order, _ = get_major_group_categories_and_colors(mapping)

    def sort_key(label: str) -> tuple[int, str]:
        major = mapping.get(label, "Miscellaneous")

        try:
            major_priority = major_order.index(major)
        except ValueError:
            major_priority = len(major_order)

        return major_priority, label

    return sorted(labels, key=sort_key)


def apply_major_group_formatting(
        ax,
        labels: list[str],
        mapping: dict[str, str],
        legend_title: str = "Classification",
        add_vertical_separators: bool = False,
        add_horizontal_separators: bool = True,
        legend_kwargs: dict | None = None,
    ) -> None:
    """Apply major-group coloring, separators, and legend.

    Args:
        ax: Matplotlib axis.
        labels: Labels in display order.
        mapping: Source-label -> major-group mapping.
        legend_title: Legend title.
        add_vertical_separators: Whether to draw vertical separators.
        add_horizontal_separators: Whether to draw horizontal separators.
        legend_kwargs: Extra kwargs passed to ax.legend().
    """
    if not mapping:
        return

    major_order, colors_map = get_major_group_categories_and_colors(mapping)

    # Color y-axis labels
    for label in ax.get_yticklabels():
        label_text = label.get_text()
        major = mapping.get(label_text)

        if major is not None:
            label.set_color(colors_map.get(major, "black"))
            label.set_fontweight("bold")

    # Color x-axis labels
    for label in ax.get_xticklabels():
        label_text = label.get_text()
        major = mapping.get(label_text)

        if major is not None:
            label.set_color(colors_map.get(major, "black"))
            label.set_fontweight("bold")

    # Draw separators
    current_major = None

    for idx, label in enumerate(labels):
        major = mapping.get(label)

        if (
            current_major is not None
            and major != current_major
        ):
            if add_vertical_separators:
                ax.axvline(
                    x=idx,
                    color="black",
                    linewidth=0.5,
                    linestyle="--",
                )

            if add_horizontal_separators:
                ax.axhline(
                    y=idx,
                    color="black",
                    linewidth=0.5,
                    linestyle="--",
                )

        current_major = major

    # Create legend
    present_groups = {
        mapping[label]
        for label in labels
        if label in mapping
    }

    legend_elements = [
        Patch(
            facecolor=colors_map[major],
            label=major,
        )
        for major in major_order
        if major in present_groups
    ]

    legend_kwargs = legend_kwargs or {}

    ax.legend(
        handles=legend_elements,
        title=legend_title,
        frameon=True,
        **legend_kwargs,
    )
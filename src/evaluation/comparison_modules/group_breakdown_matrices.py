"""Group breakdown accuracy matrices figure module.

Generates 4 heatmap figures showing Primary_top1_accuracy broken down by:
- diagnosis (group0, from report_level_metrics.csv),
- diagnosis groups (group1, group2, group3) across models.
"""

import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from .base import BaseFigure


def _load_group_mapping(
    dictionary_path: str | None,
    source_col: str,
    target_col: str = "WHO-like Major Sections/Lineages",
) -> dict:
    """
    Load a mapping from source to target diagnostic group from the Excel dictionary.

    Args:
        dictionary_path: Path to the Excel dictionary file.
        source_col: Source column name (e.g., "Diagnosis", "WHO-like Subcategories").
        target_col: Target column name (default: "WHO-like Major Sections/Lineages").

    Returns:
        Dictionary mapping source to target categories.
    """
    mapping = {}
    dictionary_path = Path(dictionary_path) if dictionary_path else None

    if dictionary_path and dictionary_path.exists():
        try:
            df = pd.read_excel(dictionary_path)
            if source_col in df.columns and target_col in df.columns:
                for idx, row in df.iterrows():
                    source = row[source_col]
                    target = row[target_col]
                    if pd.notna(source) and pd.notna(target):
                        mapping[str(source).strip()] = str(target).strip()
        except (FileNotFoundError, ValueError, KeyError) as e:
            print(f"Warning: Could not load mapping from {dictionary_path}: {e}")
    else:
        print(f"Warning: Excel dictionary file not found at {dictionary_path}")

    return mapping


def load_group0_to_group3_mapping(dictionary_path: str) -> dict:
    """Load diagnosis to group3 mapping."""
    return _load_group_mapping(dictionary_path, "Code")


def load_group1_to_group3_mapping(dictionary_path: str) -> dict:
    """Load group1 to group3 mapping."""
    return _load_group_mapping(dictionary_path, "WHO-like Subcategories")


def load_group2_to_group3_mapping(dictionary_path: str) -> dict:
    """Load group2 to group3 mapping."""
    return _load_group_mapping(dictionary_path, "WHO-like Categories")


def _get_group_mapping_for_number(group_num: int, dictionary_path: str | None) -> dict:
    """Get the appropriate mapping function for a group number."""
    if group_num == 0:
        return load_group0_to_group3_mapping(dictionary_path)
    elif group_num == 1:
        return load_group1_to_group3_mapping(dictionary_path)
    elif group_num == 2:
        return load_group2_to_group3_mapping(dictionary_path)
    return {}


def _load_group_data_from_report(
    result_dir: Path, model_name: str, group_num: int, dictionary_path: str | None
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Load group accuracy data from report_level_metrics.csv for Groups 0-2.

    For Group0, accuracy is calculated per diagnosis code.
    For Groups 1-2, accuracy is calculated per group category (via Excel dictionary mapping).

    Returns:
        Tuple of (accuracy_series, n_series) or (None, None) if file not found or invalid.
    """
    csv_path = result_dir / "report_level_metrics.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping {model_name}")
        return None, None

    df_report = pd.read_csv(csv_path)

    if "gt_code" not in df_report.columns or "pred_code" not in df_report.columns:
        print(f"Warning: 'gt_code' or 'pred_code' columns not found in {csv_path}")
        return None, None

    # For Group0, use diagnosis code directly
    if group_num == 0:
        accuracy_by_group = {}
        n_by_group = {}

        for diagnosis in df_report["gt_code"].unique():
            if pd.isna(diagnosis):
                continue
            diagnosis_str = (
                str(int(diagnosis))
                if isinstance(diagnosis, (int, float))
                else str(diagnosis).strip()
            )
            mask = df_report["gt_code"] == diagnosis

            correct = (
                df_report.loc[mask, "gt_code"] == df_report.loc[mask, "pred_code"]
            ).sum()
            total = mask.sum()
            accuracy_by_group[diagnosis_str] = correct / total if total > 0 else 0
            n_by_group[diagnosis_str] = total

        return pd.Series(accuracy_by_group), pd.Series(n_by_group)

    # For Groups 1-2, use the pre-computed group columns in the report
    if group_num not in [1, 2]:
        return None, None

    gt_group_col = f"gt_group{group_num}"
    pred_group_col = f"pred_group{group_num}"

    if gt_group_col not in df_report.columns or pred_group_col not in df_report.columns:
        print(
            f"Warning: '{gt_group_col}' or '{pred_group_col}' columns not found in {csv_path}"
        )
        return None, None

    # Calculate accuracy per group
    accuracy_by_group = {}
    n_by_group = {}

    for group in df_report[gt_group_col].unique():
        if pd.isna(group):
            continue
        mask = df_report[gt_group_col] == group
        correct = (
            df_report.loc[mask, gt_group_col] == df_report.loc[mask, pred_group_col]
        ).sum()
        total = mask.sum()
        accuracy_by_group[group] = correct / total if total > 0 else 0
        n_by_group[group] = total

    return pd.Series(accuracy_by_group), pd.Series(n_by_group)


def _load_group_data(
    group_num: int, model_names: list, results: list, dictionary_path: str | None
) -> tuple[dict, pd.Series | None]:
    """
    Load group data for all models.

    Returns:
        Tuple of (group_data_dict, group_n_values) where group_data_dict maps
        model_name to pd.Series of accuracies.
    """
    group_data = {}
    group_n_values = None

    for result_dir, model_name in zip(results, model_names):
        result_dir = Path(result_dir)

        if group_num in [0, 1, 2]:
            accuracy, n_values = _load_group_data_from_report(
                result_dir, model_name, group_num, dictionary_path
            )
        elif group_num == 3:
            accuracy, n_values = _load_group3_data(result_dir, model_name)
        else:
            accuracy, n_values = None, None

        if accuracy is not None:
            group_data[model_name] = accuracy
            if group_n_values is None and n_values is not None:
                group_n_values = n_values

    return group_data, group_n_values


def _load_group3_data(
    result_dir: Path, model_name: str
) -> tuple[pd.Series | None, pd.Series | None]:
    """
    Load Group3 (top-level) accuracy data from accuracy_breakdown_group3.csv.

    Returns:
        Tuple of (accuracy_series, n_series) or (None, None) if file not found or invalid.
    """
    csv_path = result_dir / "accuracy_breakdown_group3.csv"

    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping {model_name}")
        return None, None

    df_group = pd.read_csv(csv_path)

    if "primary_top1_accuracy" not in df_group.columns:
        print(f"Warning: 'primary_top1_accuracy' column not found in {csv_path}")
        return None, None

    if "group" not in df_group.columns:
        return None, None

    df_group = df_group.set_index("group")
    accuracy = df_group["primary_top1_accuracy"]
    n_values = df_group.get("n") if "n" in df_group.columns else None

    return accuracy, n_values


def _prepare_heatmap_dataframe(
    group_data: dict, group_n_values: pd.Series | None, model_names: list
) -> pd.DataFrame:
    """
    Prepare heatmap dataframe with proper column ordering and N values.

    Returns:
        Formatted DataFrame with N column (if available) as first column.
    """
    heatmap_df = pd.DataFrame(group_data)

    if group_n_values is not None:
        heatmap_df.insert(0, "N", group_n_values)

    # Reorder columns to match model_names order (keeping N first if present)
    if "N" in heatmap_df.columns:
        available_columns = ["N"] + [
            col for col in model_names if col in heatmap_df.columns
        ]
    else:
        available_columns = [col for col in model_names if col in heatmap_df.columns]

    return heatmap_df[available_columns]


def _sort_dataframe_by_group3(
    heatmap_df: pd.DataFrame, group_num: int, group3_mapping: dict
) -> pd.DataFrame:
    """Sort dataframe by group3 category mapping."""
    if not group3_mapping:
        return heatmap_df

    group3_order = [
        "Malignant/neoplastic",
        "Reactive/inflammatory",
        "Infectious Lymphadenitis",
        "Miscellaneous",
    ]

    sort_keys = []
    for item_name in heatmap_df.index:
        group3 = group3_mapping.get(item_name, "Miscellaneous")
        try:
            g3_priority = group3_order.index(group3)
        except ValueError:
            g3_priority = len(group3_order)
        sort_keys.append((g3_priority, item_name))

    sorted_indices = sorted(range(len(sort_keys)), key=lambda i: sort_keys[i])
    return heatmap_df.iloc[sorted_indices]


def _prepare_annotations(heatmap_df: pd.DataFrame) -> list:
    """Prepare annotation matrix for heatmap."""
    annot_matrix = []
    for row_idx in heatmap_df.index:
        row_annot = []
        for col in heatmap_df.columns:
            val = heatmap_df.loc[row_idx, col]
            if col == "N":
                row_annot.append(f"{int(val)}")
            else:
                row_annot.append(f"{val:.3f}")
        annot_matrix.append(row_annot)
    return annot_matrix


def _create_heatmap_mask(heatmap_df: pd.DataFrame) -> pd.DataFrame | None:
    """Create mask to exclude N column from colormap."""
    if "N" not in heatmap_df.columns:
        return None
    mask = pd.DataFrame(False, index=heatmap_df.index, columns=heatmap_df.columns)
    mask["N"] = True
    return mask


def _draw_separators(ax, heatmap_df: pd.DataFrame, group3_mapping: dict) -> None:
    """Draw visual separators between group3 categories."""
    if not group3_mapping:
        return

    current_g3 = None
    separators = []

    for idx, item_name in enumerate(heatmap_df.index):
        group3 = group3_mapping.get(item_name, "Miscellaneous")
        if current_g3 is not None and group3 != current_g3:
            separators.append(idx)
        current_g3 = group3

    for sep_idx in separators:
        ax.axhline(y=sep_idx, color="black", linewidth=2.5)


def _style_labels_and_legend(
    ax, heatmap_df: pd.DataFrame, group3_mapping: dict
) -> None:
    """Add color styling to labels and create legend based on group3 mapping."""
    if not group3_mapping:
        return

    # Get unique group3 categories in the data
    group3_categories_in_data = {
        group3_mapping.get(name, "Miscellaneous") for name in heatmap_df.index
    }

    # Define order and colors
    group3_order = [
        "Malignant/neoplastic",
        "Reactive/inflammatory",
        "Infectious Lymphadenitis",
        "Miscellaneous",
    ]

    group3_order_extended = list(group3_order)
    for g3 in sorted(group3_categories_in_data):
        if g3 not in group3_order_extended:
            group3_order_extended.append(g3)

    colors_map = {
        "Malignant/neoplastic": "#d62728",  # red
        "Reactive/inflammatory": "#2ca02c",  # green
        "Infectious Lymphadenitis": "#ff7f0e",  # orange
        "Miscellaneous": "#9467bd",  # purple
    }

    # Assign colors to unknown categories
    unknown_categories = [g3 for g3 in group3_order_extended if g3 not in colors_map]
    color_palette = ["#1f77b4", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
    for i, g3 in enumerate(unknown_categories):
        colors_map[g3] = color_palette[i % len(color_palette)]

    # Color y-axis labels
    yticklabels = ax.get_yticklabels()
    for label in yticklabels:
        item_name = label.get_text()
        if item_name:
            group3 = group3_mapping.get(item_name, "Miscellaneous")
            color = colors_map.get(group3, "black")
            label.set_color(color)
            label.set_fontweight("bold")

    # Create legend
    legend_elements = [
        Patch(facecolor=colors_map[g3], label=g3)
        for g3 in group3_order_extended
        if g3 in group3_categories_in_data
    ]

    ax.legend(
        handles=legend_elements,
        loc="lower left",
        bbox_to_anchor=(-0.35, -0.2),
        frameon=True,
        title="Disease Behavior",
        fontsize=10,
    )


def _format_plot_labels(ax) -> None:
    """Format and wrap axis labels for readability."""
    ax.set_yticklabels(
        [textwrap.fill(label.get_text(), width=40) for label in ax.get_yticklabels()],
        fontsize=8,
    )
    ax.set_xticklabels(
        [textwrap.fill(label.get_text(), width=15) for label in ax.get_xticklabels()],
        rotation=45,
        ha="right",
    )
    plt.yticks(rotation=0)


class GroupBreakdownMatricesFigure(BaseFigure):
    """Generate heatmap matrices for group accuracy breakdown."""

    def generate(
        self,
        df: pd.DataFrame,
        name_mappings: dict | None = None,
        dictionary_path: str | None = None,
    ) -> None:
        """
        Generate 4 heatmap figures from group breakdown CSV files and diagnosis data.

        Args:
            df: DataFrame with performance metrics (used for model names if not in config).
            name_mappings: Dictionary for mapping metric names to display names.
            dictionary_path: Path to the diagnosis dictionary for group0-3 to group3 mapping.
        """
        # Get results directories from config
        if "results" not in self.config:
            print("No results directories configured for group breakdown matrices.")
            return

        results = self.config["results"]

        # Extract date-based names from paths
        date_based_names = [Path(r).parts[-2] for r in results]

        # Apply name mappings if available (use passed parameter or config)
        if name_mappings is None:
            name_mappings = self.config.get("name_mappings", {})
        model_names = [name_mappings.get(name, name) for name in date_based_names]

        # Generate 4 figures for group0 (diagnosis), group1, group2, and group3
        for group_num in range(4):
            self._generate_group_figure(
                group_num, model_names, results, dictionary_path
            )

    def _generate_group_figure(
        self,
        group_num: int,
        model_names: list,
        results: list,
        dictionary_path: str | None,
    ) -> None:
        """
        Generate a single heatmap figure for a group.

        Args:
            group_num: Group number (0 for diagnosis, 1, 2, or 3).
            model_names: List of model names.
            results: List of results directories.
            dictionary_path: Path to the diagnosis dictionary.
        """
        # Load data
        group_data, group_n_values = _load_group_data(
            group_num, model_names, results, dictionary_path
        )

        if not group_data:
            print(f"No data loaded for group{group_num}, skipping figure generation.")
            return

        # Prepare dataframe
        heatmap_df = _prepare_heatmap_dataframe(group_data, group_n_values, model_names)

        # Sort by group3 mapping if applicable
        if group_num in [0, 1, 2]:
            group3_mapping = _get_group_mapping_for_number(group_num, dictionary_path)
            heatmap_df = _sort_dataframe_by_group3(
                heatmap_df, group_num, group3_mapping
            )

        # Create figure
        self._plot_heatmap(group_num, heatmap_df, dictionary_path)

    def _plot_heatmap(
        self, group_num: int, heatmap_df: pd.DataFrame, dictionary_path: str | None
    ) -> None:
        """Create and save the heatmap figure."""
        figsize = tuple(self.config.get("figsize", [12, 10]))
        dpi = self.config.get("dpi", 300)
        cmap = self.config.get("cmap", "RdYlGn")
        vmin = self.config.get("vmin", 0)
        vmax = self.config.get("vmax", 1)

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Prepare data for heatmap
        annot_matrix = _prepare_annotations(heatmap_df)
        mask = _create_heatmap_mask(heatmap_df)

        # Plot N column separately if present
        if "N" in heatmap_df.columns:
            sns.heatmap(
                heatmap_df[["N"]],
                cmap=sns.color_palette(["white"], as_cmap=True),
                cbar=False,
                annot=True,
                fmt=".0f",
                annot_kws={"color": "black"},
                linewidths=0.5,
                linecolor="gray",
                ax=ax,
            )

        # Plot main heatmap with custom annotations
        sns.heatmap(
            heatmap_df,
            annot=annot_matrix,
            fmt="",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={"label": "Accuracy"},
            mask=mask,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"color": "black"},
        )

        # Add styling for groups 0-2
        if group_num in [0, 1, 2]:
            group3_mapping = _get_group_mapping_for_number(group_num, dictionary_path)
            _draw_separators(ax, heatmap_df, group3_mapping)
            _style_labels_and_legend(ax, heatmap_df, group3_mapping)

        # Customize plot
        title = (
            self.config.get(f"group{group_num}_title", {}) + " Accuracy Breakdown"
            if self.config.get(f"group{group_num}_title")
            else f"Group {group_num} Accuracy Breakdown"
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        ax.set_ylabel("")

        # Format labels
        _format_plot_labels(ax)

        # Save figure
        output_key = f"output_group{group_num}"
        output_path = self.config.get(
            output_key,
            f"outputs/compare/group{group_num}_accuracy_breakdown_heatmap.png",
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved group {group_num} accuracy breakdown heatmap to: {output_path}")
        plt.close()

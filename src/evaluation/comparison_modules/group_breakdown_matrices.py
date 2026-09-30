"""Group breakdown accuracy matrices figure module.

Generates heatmap figures showing accuracy broken down by diagnosis groups.
Generates heatmap figures showing per-category accuracy
from report-level evaluation DataFrames.

Supports configurable major_group parameter to determine which group is used
for organizing and coloring the heatmap (e.g., major_group=group_4).
"""

import logging
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .base import BaseFigure

from ..utils import (
    build_label_to_major_mapping,
    apply_major_group_formatting,
    load_report_df,
    sort_labels_by_major_group
)

logger = logging.getLogger(__name__)


def calculate_group_accuracy_from_report_df(report_df: pd.DataFrame) -> dict:
    """Calculate accuracy breakdown for each group from the report DataFrame.

    Args:
        report_df: Report-level DataFrame containing gt_code, pred_code, and group columns.

    Returns:
        Dictionary containing accuracy breakdown for each group.
    """
    group_columns = {
        "GT Report Diagnosis Code": ("gt_code", "pred_code"),
        "GT Report WHO-like Subcategories": ("gt_group1", "pred_group1"),
        "GT Report WHO-like Categories": ("gt_group2", "pred_group2"),
        "GT Report WHO-like Major Sections/Lineages": (
            "gt_group3",
            "pred_group3",
        ),
        "GT Report Diagnosis Group 4": ("gt_group4", "pred_group4"),
    }

    results = {}

    for name, (gt_col, pred_col) in group_columns.items():

        gt = (report_df[gt_col].fillna("N/A").astype(str).str.strip())
        pred = (report_df[pred_col].fillna("N/A").astype(str).str.strip())

        stats = (
            pd.DataFrame(
                {
                    "GT": gt,
                    "Correct": (gt == pred).astype(int),
                }
            )
            .loc[lambda x: x["GT"] != "N/A"]
            .groupby("GT")["Correct"]
            .agg(
                N="count",
                Correct="sum",
            )
        )

        results[name] = stats.to_dict("index")

    return results


def prepare_heatmap_dataframe(
        all_group_data: dict[str, dict],
        category_type: int = 2,
    ) -> pd.DataFrame:
    """
    Prepare heatmap dataframe with proper column ordering and N values.

    Returns:
        Formatted DataFrame with N column (if available) as first column.
    """
    category_type_map = {
        0: "GT Report Diagnosis Code",
        1: "GT Report WHO-like Subcategories",
        2: "GT Report WHO-like Categories",
        3: "GT Report WHO-like Major Sections/Lineages",
        4: "GT Report Diagnosis Group 4",
    }

    # Extract model names from the keys of all_group_data
    model_names = list(all_group_data.keys())

    heatmap_df = pd.DataFrame()

    for model_name in model_names:
        for heatmap_name, data in all_group_data[model_name].items():
            if heatmap_name == category_type_map[category_type]:
                for key, value in data.items():

                    heatmap_df.loc[key, model_name] = value["Correct"] / value["N"] if value["N"] > 0 else 0
                    heatmap_df.loc[key, "N"] = value["N"]

    # Reorder columns to match model_names order (keeping N first if present)
    if "N" in heatmap_df.columns:
        available_columns = ["N"] + [
            col for col in model_names if col in heatmap_df.columns
        ]
    else:
        available_columns = [col for col in model_names if col in heatmap_df.columns]


    return heatmap_df[available_columns]


def prepare_annotations(heatmap_df: pd.DataFrame) -> list:
    """Prepare annotation matrix for heatmap.
    
    Args:
        heatmap_df: DataFrame prepared for heatmap.

    Returns:
        List of lists containing the annotations for each cell in the heatmap.
    """
    annot_matrix = []
    for row_idx in heatmap_df.index:
        row_annot = []
        for col in heatmap_df.columns:
            val = heatmap_df.loc[row_idx, col]
            if col == "N":
                if pd.isna(val):
                    row_annot.append("N/A")
                else:
                    row_annot.append(f"{int(val)}")
            else:
                row_annot.append(f"{val:.2f}")
        annot_matrix.append(row_annot)
    return annot_matrix


def create_heatmap_mask(heatmap_df: pd.DataFrame) -> pd.DataFrame | None:
    """Create mask to exclude N column from colormap.
    
    Args:
        heatmap_df: DataFrame prepared for heatmap.

    Returns:
        Mask DataFrame with True values for the N column.
    """
    if "N" not in heatmap_df.columns:
        return None
    mask = pd.DataFrame(False, index=heatmap_df.index, columns=heatmap_df.columns)
    mask["N"] = True
    return mask


def format_plot_labels(ax) -> None:
    """Format and wrap axis labels for readability.
    
    Args:
        ax: Matplotlib Axes object containing the heatmap.
    """
    ax.set_yticklabels(
        [textwrap.fill(label.get_text(), width=40) for label in ax.get_yticklabels()],
        fontsize=12,
    )
    ax.set_xticklabels(
        [textwrap.fill(label.get_text(), width=40) for label in ax.get_xticklabels()],
        rotation=90,
        fontsize=12,
        ha="right",
    )
    plt.yticks(rotation=0)


class GroupBreakdownMatricesFigure(BaseFigure):
    """Generate heatmap matrices for group accuracy breakdown."""

    def generate(self, name_mappings: dict | None = None) -> None:
        """
        Generate heatmap figures from ontology JSONL predictions vs validation Excel.

        Args:
            name_mappings: Dictionary for mapping metric names to display names.
        """
        # Get results directories from config
        if "results" not in self.config:
            logger.info("No results directories configured for group breakdown matrices.")
            return

        results = self.config["results"]
        major_group = self.config.get("major_group", "group_3")

        # Apply name mappings
        if name_mappings is None:
            name_mappings = self.config.get("name_mappings", {})
        
        # Extract model names from result paths
        model_names = []
        for result_dir in results:
            # Try to get date-based name from path
            path_parts = Path(result_dir).parts
            if len(path_parts) >= 2:
                date_name = path_parts[-2]
            else:
                date_name = Path(result_dir).name
            model_names.append(name_mappings.get(date_name, date_name))

        # Determine which groups to generate
        max_group_num = 4 if major_group == "group_4" else 3

        # Generate figures for each group
        for group_num in range(max_group_num + 1):
            self.generate_group_figure(
                group_num,
                model_names,
                results,
                major_group=major_group,
            )

    def generate_group_figure(
            self,
            group_num: int,
            model_names: list,
            results: list,
            major_group: str = "group_3",
        ) -> None:
        """
        Generate a single heatmap figure for a group.

        Args:
            group_num: Group number (0, 1, 2, 3, or 4).
            model_names: List of model names.
            results: List of results directories.
            major_group: Major group for organizing heatmap (e.g., "group_3" or "group_4").
        """
        # Load predictions for all models and calculate accuracy
        group_data = {}
        all_group_data = {}
        mapping_report_df = None

        for model_name, result_dir in zip(model_names, results):
            # result_path = Path(result_dir).parent
            report_df = load_report_df(result_dir)

            if (
                    mapping_report_df is None
                    and report_df is not None
                    and not report_df.empty
                ):
                mapping_report_df = report_df

            if report_df is None or report_df.empty:
                logger.warning(
                    f"No report dataframe available for {model_name}"
                )
                continue

            group_data = calculate_group_accuracy_from_report_df(
                report_df
            )

            all_group_data[model_name] = group_data

        # Prepare dataframe
        heatmap_df = prepare_heatmap_dataframe(all_group_data, category_type=group_num)

        if heatmap_df.empty:
            logger.warning(
                f"No data available for group {group_num}"
            )
            return

        # Build mapping to major_group if needed
        major_mapping = {}
        if group_num != int(major_group.replace("group_", "")):
            if group_num > 0:
                major_mapping = build_label_to_major_mapping(
                    mapping_report_df,
                    group_key=f"group_{group_num}",
                    major_group=major_group,
                )
            if major_mapping:
                sorted_labels = sort_labels_by_major_group(
                    list(heatmap_df.index),
                    major_mapping,
                )

                heatmap_df = heatmap_df.loc[sorted_labels]

        # Create figure
        self.plot_heatmap(
            group_num,
            heatmap_df,
            major_group=major_group,
            major_mapping=major_mapping,
        )

    def plot_heatmap(
            self,
            group_num: int,
            heatmap_df: pd.DataFrame,
            major_group: str = "group_3",
            major_mapping: dict | None = None,
        ) -> None:
        """Create and save the heatmap figure.

        Args:
            group_num: Group number (0, 1, 2, 3, or 4)
            heatmap_df: DataFrame prepared for heatmap
            major_group: Major group for organizing heatmap
            major_mapping: Mapping from labels to major_group values
        """
        figsize = tuple(self.config.get("figsize", [12, 10]))
        dpi = self.config.get("dpi", 300)
        cmap = self.config.get("cmap", "RdYlGn")
        vmin = self.config.get("vmin", 0)
        vmax = self.config.get("vmax", 1)

        _, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Prepare data for heatmap
        annot_matrix = prepare_annotations(heatmap_df)
        mask = create_heatmap_mask(heatmap_df)

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
            cbar_kws={"pad": 0.02},
            mask=mask,
            ax=ax,
            linewidths=0.5,
            linecolor="gray",
            annot_kws={"color": "black", "fontsize": 12},
        )

        cbar = ax.collections[-1].colorbar
        cbar.set_label("Accuracy", fontsize=12)
        cbar.ax.tick_params(labelsize=12)

        # # Add styling if mapping is available
        # if major_mapping:
        #     _draw_separators(ax, heatmap_df, major_mapping)
        #     _style_labels_and_legend(ax, heatmap_df, major_mapping, major_group)
        if major_mapping:
            apply_major_group_formatting(
                ax=ax,
                labels=list(heatmap_df.index),
                mapping=major_mapping,
                legend_title=(
                    "Classification"
                    if major_group == "group_4"
                    else "WHO-like Major Sections/Lineages"
                ),
                add_vertical_separators=False,
                add_horizontal_separators=True,
                legend_kwargs={
                    "loc": "lower left",
                    "bbox_to_anchor": (-0.25, -0.15),
                    "fontsize": 12,
                    "title_fontsize": 12,
                },
            )

        # Customize plot
        title = (
            self.config.get(f"group{group_num}_title", {}) + " Accuracy Breakdown"
            if self.config.get(f"group{group_num}_title")
            else f"Group {group_num} Accuracy Breakdown"
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel("")

        # Format labels
        format_plot_labels(ax)

        # Save figure
        output_key = f"output_group{group_num}"
        output_path = self.config.get(
            output_key,
            f"outputs/compare/group{group_num}_accuracy_breakdown_heatmap.png",
        )

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved group {group_num} accuracy breakdown heatmap to: {output_path}")
        plt.close()

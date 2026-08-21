"""
Group breakdown accuracy matrices figure module.

Generates 3 heatmap figures showing Primary_top1_accuracy broken down by
diagnosis groups (group1, group2, group3) across models.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .base import BaseFigure


def load_group2_to_group3_mapping(dictionary_path: str) -> dict:
    """
    Extract the mapping between group2 and group3 from the Excel dictionary.

    Args:
        dictionary_path: Path to the Excel dictionary file.

    Returns:
        Dictionary mapping group2 categories to group3 categories.
    """
    mapping = {}
    dictionary_path = Path(dictionary_path) if dictionary_path else None

    if dictionary_path and dictionary_path.exists():
        try:
            df = pd.read_excel(dictionary_path)

            # Create mapping from "Diagnostic group 2" to "Diagnostic group 3"
            if (
                "Diagnostic group 2" in df.columns
                and "Diagnostic group 3" in df.columns
            ):
                for idx, row in df.iterrows():
                    g2 = row["Diagnostic group 2"]
                    g3 = row["Diagnostic group 3"]
                    if pd.notna(g2) and pd.notna(g3):
                        mapping[str(g2).strip()] = str(g3).strip()
        except Exception as e:
            print(f"Warning: Could not load mapping from {dictionary_path}: {e}")
    else:
        print(f"Warning: Excel dictionary file not found at {dictionary_path}")

    return mapping


class GroupBreakdownMatricesFigure(BaseFigure):
    """Generate heatmap matrices for group accuracy breakdown."""

    def generate(
        self, df: pd.DataFrame, name_mappings: dict = None, dictionary_path: str = None
    ) -> None:
        """
        Generate 3 heatmap figures from group breakdown CSV files.

        Args:
            df: DataFrame with performance metrics (used for model names if not in config).
            name_mappings: Dictionary for mapping metric names to display names.
            dictionary_path: Path to the diagnosis dictionary for group2 to group3 mapping.
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

        # Generate 3 figures for group1, group2, and group3
        for group_num in range(1, 4):
            self._generate_group_figure(
                group_num, model_names, results, dictionary_path
            )

    def _generate_group_figure(
        self, group_num: int, model_names: list, results: list, dictionary_path: str
    ) -> None:
        """
        Generate a single heatmap figure for a group.

        Args:
            group_num: Group number (1, 2, or 3).
            model_names: List of model names.
            results: List of results directories.
            dictionary_path: Path to the diagnosis dictionary.
        """
        # Load data from group CSV files for all results directories
        csv_filename = f"accuracy_breakdown_group{group_num}.csv"

        # Dictionary to store group data for each model
        group_data = {}
        group_n_values = None  # Store N values (should be same across models)

        for result_dir, model_name in zip(results, model_names):
            csv_path = Path(result_dir) / csv_filename

            if not csv_path.exists():
                print(f"Warning: {csv_path} not found, skipping {model_name}")
                continue

            # Load CSV and extract primary_top1_accuracy column
            df_group = pd.read_csv(csv_path)

            if "primary_top1_accuracy" not in df_group.columns:
                print(
                    f"Warning: 'primary_top1_accuracy' column not found in {csv_path}"
                )
                continue

            # Set group column as index and extract accuracy values
            if "group" in df_group.columns:
                df_group = df_group.set_index("group")
                group_data[model_name] = df_group["primary_top1_accuracy"]

                # Store N values from the first model (same for all models)
                if group_n_values is None and "n" in df_group.columns:
                    group_n_values = df_group["n"]

        if not group_data:
            print(f"No data loaded for group{group_num}, skipping figure generation.")
            return

        # Create dataframe with groups as rows and models as columns
        heatmap_df = pd.DataFrame(group_data)

        # Add N column as the leftmost column if available
        if group_n_values is not None:
            heatmap_df.insert(0, "N", group_n_values)

        # Reorder columns to match model_names order (keeping N first if present)
        if "N" in heatmap_df.columns:
            available_columns = ["N"] + [
                col for col in model_names if col in heatmap_df.columns
            ]
        else:
            available_columns = [
                col for col in model_names if col in heatmap_df.columns
            ]
        heatmap_df = heatmap_df[available_columns]

        # For group2, sort by group3 category + group2 name
        if group_num == 2 and len(results) > 0:
            g2_to_g3_mapping = load_group2_to_group3_mapping(
                dictionary_path=dictionary_path
            )
            if g2_to_g3_mapping:
                # Define group3 category order
                group3_order = [
                    "Malignant/neoplastic",
                    "Reactive/inflammatory",
                    "Infectious Lymphadenitis",
                    "Miscellaneous",
                ]

                # Create sort keys: (group3_priority, group2_name)
                sort_keys = []
                for group2_name in heatmap_df.index:
                    group3 = g2_to_g3_mapping.get(group2_name, "Miscellaneous")
                    # Find group3 priority
                    try:
                        g3_priority = group3_order.index(group3)
                    except ValueError:
                        g3_priority = len(group3_order)  # Unknown categories go last
                    sort_keys.append((g3_priority, group2_name))

                # Sort and reindex
                sorted_indices = sorted(
                    range(len(sort_keys)), key=lambda i: sort_keys[i]
                )
                heatmap_df = heatmap_df.iloc[sorted_indices]

        # Create the figure
        figsize = tuple(self.config.get("figsize", [12, 10]))
        dpi = self.config.get("dpi", 300)
        cmap = self.config.get("cmap", "RdYlGn")
        vmin = self.config.get("vmin", 0)
        vmax = self.config.get("vmax", 1)

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Prepare annotations with proper formatting for each column
        annot_matrix = []
        for idx, row_idx in enumerate(heatmap_df.index):
            row_annot = []
            for col in heatmap_df.columns:
                val = heatmap_df.loc[row_idx, col]
                if col == "N":
                    row_annot.append(f"{int(val)}")
                else:
                    row_annot.append(f"{val:.3f}")
            annot_matrix.append(row_annot)

        # Create a mask to make the N column white (no colormap applied)
        mask = None
        if "N" in heatmap_df.columns:
            mask = pd.DataFrame(
                False, index=heatmap_df.index, columns=heatmap_df.columns
            )
            mask["N"] = True

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

        # Create heatmap with custom annotations
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

        # For group2, add visual separators between group3 categories
        if group_num == 2 and len(results) > 0:
            g2_to_g3_mapping = load_group2_to_group3_mapping(
                dictionary_path=dictionary_path
            )
            if g2_to_g3_mapping:
                # Find boundaries between different group3 categories
                group3_order = [
                    "Malignant/neoplastic",
                    "Reactive/inflammatory",
                    "Infectious Lymphadenitis",
                    "Miscellaneous",
                ]

                current_g3 = None
                separators = []

                for idx, group2_name in enumerate(heatmap_df.index):
                    group3 = g2_to_g3_mapping.get(group2_name, "Miscellaneous")
                    if current_g3 is not None and group3 != current_g3:
                        # Found a category boundary
                        separators.append(idx)
                    current_g3 = group3

                # Draw thick lines at boundaries
                for sep_idx in separators:
                    ax.axhline(y=sep_idx, color="black", linewidth=2.5)

        # For group2, color the y-axis labels by group3 category
        if group_num == 2 and len(results) > 0:
            g2_to_g3_mapping = load_group2_to_group3_mapping(
                dictionary_path=dictionary_path
            )
            if g2_to_g3_mapping:
                # Get all unique group3 categories actually present in the data
                group3_categories_in_data = set(
                    g2_to_g3_mapping.get(name, "Miscellaneous")
                    for name in heatmap_df.index
                )

                # Define default order with known categories
                group3_order = [
                    "Malignant/neoplastic",
                    "Reactive/inflammatory",
                    "Infectious Lymphadenitis",
                    "Miscellaneous",
                ]

                # Add any categories from data that aren't in the default order
                for g3 in sorted(group3_categories_in_data):
                    if g3 not in group3_order:
                        group3_order.append(g3)

                # Define colors for known categories
                colors_map = {
                    "Malignant/neoplastic": "#d62728",  # red
                    "Reactive/inflammatory": "#2ca02c",  # green
                    "Infectious Lymphadenitis": "#ff7f0e",  # orange
                    "Miscellaneous": "#9467bd",  # purple
                }

                # Assign colors to unknown categories from a palette
                unknown_categories = [g3 for g3 in group3_order if g3 not in colors_map]
                color_palette = ["#1f77b4", "#17becf", "#bcbd22", "#e377c2", "#7f7f7f"]
                for i, g3 in enumerate(unknown_categories):
                    colors_map[g3] = color_palette[i % len(color_palette)]

                # Get the y-axis tick labels and their positions
                yticklabels = ax.get_yticklabels()
                yticks = ax.get_yticks()

                # Color each y-axis label based on its group3 category
                for i, label in enumerate(yticklabels):
                    group2_name = label.get_text()
                    if group2_name:
                        group3 = g2_to_g3_mapping.get(group2_name, "Miscellaneous")
                        color = colors_map.get(group3, "black")
                        label.set_color(color)
                        label.set_fontweight("bold")

                # Create legend patches for group3 categories
                from matplotlib.patches import Patch

                legend_elements = [
                    Patch(facecolor=colors_map[g3], label=g3)
                    for g3 in group3_order
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

        # Customize plot
        title = (
            self.config.get(f"group{group_num}_title", {}) + " Accuracy Breakdown"
            if self.config.get(f"group{group_num}_title")
            else f"Group {group_num} Accuracy Breakdown"
        )
        ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
        # ax.set_xlabel("Models", fontsize=12, fontweight="bold")
        # ax.set_ylabel(GROUPS[group_num], fontsize=12, fontweight="bold")
        ax.set_ylabel("")

        # Wrap long y-axis labels and rotate x-axis labels for better readability
        import textwrap

        ax.set_yticklabels(
            [
                textwrap.fill(label.get_text(), width=40)
                for label in ax.get_yticklabels()
            ],
            fontsize=8,
        )
        ax.set_xticklabels(
            [
                textwrap.fill(label.get_text(), width=15)
                for label in ax.get_xticklabels()
            ],
            rotation=45,
            ha="right",
        )
        plt.yticks(rotation=0)

        # Save figure
        output_key = f"output_group{group_num}"
        output_path = self.config.get(
            output_key,
            f"outputs/compare/group{group_num}_accuracy_breakdown_heatmap.png",
        )

        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved group {group_num} accuracy breakdown heatmap to: {output_path}")
        plt.close()

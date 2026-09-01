"""Group accuracy hierarchy figure generation module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .base import BaseFigure


class GroupAccuracyFigure(BaseFigure):
    """Generate a grouped bar chart for group accuracy across top-k levels."""

    def generate(self, df: pd.DataFrame, name_mappings: dict | None = None) -> None:
        """
        Generate and export the group accuracy grouped bar chart figure.

        Shows accuracy for each group level (1, 2, 3) at different top-k values (1, 3, 5),
        grouped by model with different shades for each top-k value.

        Args:
            df: DataFrame containing the group accuracy metrics.
            name_mappings: Dictionary for mapping metric names to display names.
        """
        output_path = self.config["output"]
        title = self.config.get("title", "Group Accuracy Hierarchy Comparison")
        ylabel = self.config.get("ylabel", "Accuracy")
        dpi = self.config.get("dpi", 300)
        figsize = tuple(self.config.get("figsize", [10, 6]))

        # Define the metrics organized by group level and top-k using original metric names
        # These will be looked up in the DataFrame after name mapping has been applied
        if name_mappings is None:
            name_mappings = {}

        category_names = {
            1: "subcategories",
            2: "categories",
            3: "major_sections_lineages",
        }

        metrics_by_group_topk = {}
        for group in [1, 2, 3]:
            for topk, topk_label in [(1, "Top-1"), (3, "Top-3"), (5, "Top-5")]:
                original_metric = f"who-like_{category_names[group]}_top{topk}_accuracy"
                # Get the display name from name_mappings if available
                if original_metric in name_mappings:
                    display_name = name_mappings[original_metric]
                else:
                    # Fallback: construct from group and topk
                    display_name = (
                        f"WHO-like {category_names[group]}\n{topk_label} Accuracy"
                    )
                metrics_by_group_topk[(group, topk_label)] = display_name

        # Filter dataframe by specified models if provided
        # Use substring matching since model names in config might be partial (e.g., "Medgemma-4b-it")
        # but actual index names include suffixes (e.g., "Medgemma-4b-it-om", "Medgemma-4b-it-cd")
        if "models" in self.config:
            models_to_include = self.config["models"]
            # Filter to rows where the index starts with any of the model prefixes
            mask = df.index.to_series().apply(
                lambda idx: any(idx.startswith(model) for model in models_to_include)
            )
            df = df[mask]

        _, ax = plt.subplots(figsize=figsize)

        n_models = len(df.index)
        n_groups = 3  # Levels 1, 2, 3
        n_topk = 3  # Top-1, Top-3, Top-5
        total_bars_per_model = n_groups * n_topk

        # Bar width and positioning
        bar_width = 0.6 / total_bars_per_model
        group_spacing = 0.8

        # X-axis positions for models
        x_models = np.arange(n_models) * group_spacing

        # Color scheme: different colors for group levels
        group_colors = {
            1: "#1f77b4",  # Blue
            2: "#ff7f0e",  # Orange
            3: "#2ca02c",  # Green
        }

        # Opacity shades for top-k values
        topk_alphas = {
            "Top-1": 0.4,
            "Top-3": 0.65,
            "Top-5": 0.9,
        }

        # Create custom legend handles
        from matplotlib.patches import Patch

        # Extract group names from name_mappings
        group_names = []
        for group in [1, 2, 3]:
            original_metric = f"who-like_{category_names[group]}_top1_accuracy"
            if original_metric in name_mappings:
                # Extract just the group name part (before \n)
                display_name = name_mappings[original_metric]
                group_name = display_name.split("\n")[0]  # Get first line
                group_names.append(group_name)
            else:
                group_names.append(f"Group {group}")

        group_handles = [
            Patch(color=group_colors[i], label=group_names[i - 1]) for i in range(1, 4)
        ]
        topk_handles = [
            Patch(color="gray", alpha=topk_alphas[topk], label=topk)
            for topk in ["Top-1", "Top-3", "Top-5"]
        ]

        # Plot bars for each model
        for model_idx, model_name in enumerate(df.index):
            # For each group level
            for group_idx in range(1, 4):
                group_color = group_colors[group_idx]

                # For each top-k value
                for topk_idx, topk in enumerate(["Top-1", "Top-3", "Top-5"]):
                    metric_col = metrics_by_group_topk[(group_idx, topk)]

                    if metric_col in df.columns:
                        value = df.loc[model_name, metric_col]

                        # Calculate x position within the model group
                        bar_idx = (group_idx - 1) * n_topk + topk_idx
                        offset = (bar_idx - (total_bars_per_model - 1) / 2) * bar_width
                        x_pos = x_models[model_idx] + offset

                        ax.bar(
                            x_pos,
                            value,
                            width=bar_width * 0.95,
                            color=group_color,
                            alpha=topk_alphas[topk],
                            edgecolor="black",
                            linewidth=0.5,
                        )

                        # Value labels
                        if pd.notna(value):
                            ax.text(
                                x_pos,
                                value + 0.01,
                                f"{value:.2f}",
                                ha="center",
                                va="bottom",
                                fontsize=6,
                            )

        # Set x-axis
        ax.set_xticks(x_models)
        ax.set_xticklabels(df.index)

        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Models")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)

        # Create two-row legend
        ax.legend(
            handles=group_handles + topk_handles,
            loc="lower right",
            fontsize=9,
            ncol=2,
            title="Classes (colors)\nTop-k Values (opacity)",
        )

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved group accuracy bar plot to: {output_path}")

"""i-CLASSi accuracy hierarchy figure generation module."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from .base import BaseFigure


class iCLASSiAccuracyFigure(BaseFigure):
    """Generate a grouped bar chart for i-CLASSi accuracy across top-k levels."""

    def generate(self, df: pd.DataFrame, name_mappings: dict = None) -> None:
        """
        Generate and export the i-CLASSi accuracy grouped bar chart figure.

        Shows accuracy for each i-CLASSi level (1, 2, 3) at different top-k values (1, 3, 5),
        grouped by model with different shades for each top-k value.

        Args:
            df: DataFrame containing the i-CLASSi accuracy metrics.
            name_mappings: Dictionary for mapping metric names to display names.
        """
        output_path = self.config["output"]
        title = self.config.get("title", "i-CLASSi Accuracy Hierarchy Comparison")
        ylabel = self.config.get("ylabel", "Accuracy")
        dpi = self.config.get("dpi", 300)
        figsize = tuple(self.config.get("figsize", [10, 6]))

        # Define the metrics organized by i-CLASSi level and top-k
        metrics_by_iclassi_topk = {
            (1, "Top-1"): "LEO class\nTop-1 Accuracy",
            (1, "Top-3"): "LEO class\nTop-3 Accuracy",
            (1, "Top-5"): "LEO class\nTop-5 Accuracy",
            (2, "Top-1"): "WHO class\nTop-1 Accuracy",
            (2, "Top-3"): "WHO class\nTop-3 Accuracy",
            (2, "Top-5"): "WHO class\nTop-5 Accuracy",
            (3, "Top-1"): "High level class\nTop-1 Accuracy",
            (3, "Top-3"): "High level class\nTop-3 Accuracy",
            (3, "Top-5"): "High level class\nTop-5 Accuracy",
        }

        # Filter dataframe by specified models if provided
        if "models" in self.config:
            models_to_include = self.config["models"]
            df = df.loc[df.index.isin(models_to_include)]

        fig, ax = plt.subplots(figsize=figsize)

        n_models = len(df.index)
        n_iclassi = 3  # Levels 1, 2, 3
        n_topk = 3  # Top-1, Top-3, Top-5
        total_bars_per_model = n_iclassi * n_topk

        # Bar width and positioning
        bar_width = 0.6 / total_bars_per_model
        group_spacing = 0.8

        # X-axis positions for models
        x_models = np.arange(n_models) * group_spacing

        # Color scheme: different colors for i-CLASSi levels
        iclassi_colors = {
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

        iclassi_labels = ["LEO class", "WHO class", "High level class"]
        iclassi_handles = [
            Patch(color=iclassi_colors[i], label=iclassi_labels[i - 1])
            for i in range(1, 4)
        ]
        topk_handles = [
            Patch(color="gray", alpha=topk_alphas[topk], label=topk)
            for topk in ["Top-1", "Top-3", "Top-5"]
        ]

        # Plot bars for each model
        for model_idx, model_name in enumerate(df.index):
            # For each i-CLASSi level
            for iclassi_idx in range(1, 4):
                iclassi_color = iclassi_colors[iclassi_idx]

                # For each top-k value
                for topk_idx, topk in enumerate(["Top-1", "Top-3", "Top-5"]):
                    metric_col = metrics_by_iclassi_topk[(iclassi_idx, topk)]

                    if metric_col in df.columns:
                        value = df.loc[model_name, metric_col]

                        # Calculate x position within the model group
                        bar_idx = (iclassi_idx - 1) * n_topk + topk_idx
                        offset = (bar_idx - (total_bars_per_model - 1) / 2) * bar_width
                        x_pos = x_models[model_idx] + offset

                        ax.bar(
                            x_pos,
                            value,
                            width=bar_width * 0.95,
                            color=iclassi_color,
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
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Models")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)

        # Create two-row legend
        ax.legend(
            handles=iclassi_handles + topk_handles,
            loc="lower right",
            fontsize=9,
            ncol=2,
            title="Classes (colors)\nTop-k Values (opacity)",
        )

        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved i-CLASSi accuracy bar plot to: {output_path}")

"""Grouped bar chart figure generation module."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .base import BaseFigure


def adjust_color_brightness(color, brightness_factor):
    """Adjust color brightness. Factor > 1 = lighter, Factor < 1 = darker."""
    # Convert color to RGB if it's a named color
    if isinstance(color, str):
        color = mcolors.to_rgb(color)

    # Adjust brightness
    adjusted = tuple(min(1, c * brightness_factor) for c in color)
    return adjusted


class GroupedBarFigure(BaseFigure):
    """Generate a grouped bar chart figure for performance comparison."""

    def generate(self, df: pd.DataFrame, name_mappings: dict = None) -> None:
        """
        Generate and export the grouped bar chart figure.

        Args:
            df: DataFrame containing the performance metrics.
            name_mappings: Dictionary for mapping metric names to display names.
        """
        output_path = self.config["output"]
        title = self.config.get("title", "Performance Comparison")
        ylabel = self.config.get("ylabel", "Accuracy")
        dpi = self.config.get("dpi", 300)
        figsize = tuple(self.config.get("figsize", [10, 6]))
        bar_columns = self.config.get("columns", [])

        # Resolve column names: map display names to actual DataFrame column names
        actual_columns = []
        if bar_columns and name_mappings:
            # Create reverse mapping from display name to actual column name
            reverse_mapping = {v: k for k, v in name_mappings.items()}
            for col in bar_columns:
                # Try the column as-is first (in case it's already the correct name)
                if col in df.columns:
                    actual_columns.append(col)
                # Then try to find it via reverse mapping
                elif col in reverse_mapping and reverse_mapping[col] in df.columns:
                    actual_columns.append(reverse_mapping[col])
                # If still not found, skip it
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame, skipping")
        elif bar_columns:
            actual_columns = bar_columns
        else:
            actual_columns = list(df.columns)

        # Filter to only resolved columns
        if actual_columns:
            bar_df = df[actual_columns]
        else:
            bar_df = df

        fig, ax = plt.subplots(figsize=figsize)

        n_models = len(bar_df.index)
        n_metrics = len(bar_df.columns)

        width = 0.8 / n_models
        x = range(n_metrics)
        colors_list = plt.get_cmap("tab10").colors

        # Create color mapping based on model base name
        # Group models by base name (without -om or -cd suffix)
        model_base_colors = {}
        base_model_index = 0

        for model_name in bar_df.index:
            # Extract base model name (remove -om or -cd suffix)
            base_model = model_name.replace("-om", "").replace("-cd", "")

            if base_model not in model_base_colors:
                model_base_colors[base_model] = colors_list[
                    base_model_index % len(colors_list)
                ]
                base_model_index += 1

        for i, model_name in enumerate(bar_df.index):
            offsets = [v + (i - (n_models - 1) / 2) * width for v in x]

            # Get base color for this model
            base_model = model_name.replace("-om", "").replace("-cd", "")
            base_color = model_base_colors[base_model]

            # Adjust brightness based on om vs cd
            # om (darker) gets brightness factor 0.8, cd (lighter) gets 1.2
            if "om" in model_name.lower():
                bar_color = adjust_color_brightness(base_color, 0.8)
            else:  # cd
                bar_color = adjust_color_brightness(base_color, 1.2)

            bars = ax.bar(
                offsets,
                bar_df.loc[model_name],
                width=width,
                label=model_name,
                color=bar_color,
            )

            # Value labels
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        height + 0.01,
                        f"{height:.3f}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                        rotation=90,
                    )

        ax.set_xticks(list(x))
        ax.set_xticklabels(bar_df.columns)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.2)
        ax.set_title(title)

        # Organize legend: "om" models in first column, "cd" models in second column
        handles, labels = ax.get_legend_handles_labels()
        om_indices = [i for i, label in enumerate(labels) if "om" in label.lower()]
        cd_indices = [i for i, label in enumerate(labels) if "cd" in label.lower()]

        sorted_handles = [handles[i] for i in om_indices] + [
            handles[i] for i in cd_indices
        ]
        sorted_labels = [labels[i] for i in om_indices] + [
            labels[i] for i in cd_indices
        ]

        ax.legend(sorted_handles, sorted_labels, ncols=2)
        ax.grid(axis="y", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved grouped bar plot to: {output_path}")

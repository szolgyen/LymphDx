"""Error analysis figure generation module."""

import pandas as pd
import matplotlib.pyplot as plt

from .base import BaseFigure


class ErrorAnalysisFigure(BaseFigure):
    """Generate a stacked bar chart figure for error analysis metrics (error fractions)."""

    def generate(self, df: pd.DataFrame, name_mappings: dict = None) -> None:
        """
        Generate and export the error analysis bar chart figure.

        Args:
            df: DataFrame containing the error analysis metrics.
            name_mappings: Dictionary for mapping metric names to display names.
        """
        output_path = self.config["output"]
        title = self.config.get("title", "Error Analysis Comparison")
        ylabel = self.config.get("ylabel", "Fraction")
        dpi = self.config.get("dpi", 300)
        figsize = tuple(self.config.get("figsize", [14, 6]))
        error_columns = self.config.get("columns", [])

        # Resolve column names: map display names to actual DataFrame column names
        actual_columns = []
        if error_columns and name_mappings:
            reverse_mapping = {v: k for k, v in name_mappings.items()}
            for col in error_columns:
                if col in df.columns:
                    actual_columns.append(col)
                elif col in reverse_mapping and reverse_mapping[col] in df.columns:
                    actual_columns.append(reverse_mapping[col])
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame, skipping")
        elif error_columns:
            actual_columns = error_columns
        else:
            actual_columns = list(df.columns)

        # Filter to only specified columns if provided
        if actual_columns:
            error_df = df[actual_columns]
        else:
            error_df = df.copy()

        # Drop columns that are entirely NaN (for backward compatibility with old error_analysis.json)
        error_df = error_df.dropna(axis=1, how="all")

        # Fill any remaining NaN values with 0 (for backward compatibility)
        error_df = error_df.fillna(0.0)

        # Calculate "other" fraction (1 - sum of specified fractions)
        error_df["Other"] = 1.0 - error_df.sum(axis=1)

        # Ensure all values are non-negative (handle floating point errors)
        error_df = error_df.clip(lower=0)

        fig, ax = plt.subplots(figsize=figsize)

        # Create horizontal stacked bar chart
        n_models = len(error_df.index)
        y_pos = range(n_models)

        # Use tab20 for up to 20 colors; for more, use hsv colormap
        n_colors = len(error_df.columns)
        if n_colors <= 20:
            cmap = plt.get_cmap("tab20")
            colors_list = [cmap(i) for i in range(n_colors)]
        else:
            cmap = plt.get_cmap("hsv")
            colors_list = [cmap(i / n_colors) for i in range(n_colors)]

        left = None
        for col_idx, column in enumerate(error_df.columns):
            color = colors_list[col_idx]

            if left is None:
                ax.barh(y_pos, error_df[column], label=column, color=color, height=0.6)
                left = error_df[column].values
            else:
                ax.barh(
                    y_pos,
                    error_df[column],
                    left=left,
                    label=column,
                    color=color,
                    height=0.6,
                )
                left = left + error_df[column].values

        ax.set_yticks(y_pos)
        ax.set_yticklabels(error_df.index)
        ax.set_xlabel(ylabel)
        ax.set_xlim(0, 1.05)
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved error analysis stacked bar plot to: {output_path}")

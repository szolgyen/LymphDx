"""MCC metrics bar chart figure generation module."""

import pandas as pd
import matplotlib.pyplot as plt

from .base import BaseFigure


class MCCMetricsFigure(BaseFigure):
    """Generate a bar chart figure for MCC (Matthews Correlation Coefficient) metrics."""

    def generate(self, df: pd.DataFrame, name_mappings: dict = None) -> None:
        """
        Generate and export the MCC metrics bar chart figure.

        Args:
            df: DataFrame containing the performance metrics (must include MCC columns).
            name_mappings: Dictionary for mapping metric names to display names.
        """
        output_path = self.config["output"]
        title = self.config.get("title", "MCC Metrics Comparison")
        ylabel = self.config.get("ylabel", "Matthews Correlation Coefficient")
        dpi = self.config.get("dpi", 300)
        figsize = tuple(self.config.get("figsize", [12, 6]))
        mcc_columns = self.config.get("columns", [])

        # Resolve column names: map display names to actual DataFrame column names
        actual_columns = []
        if mcc_columns and name_mappings:
            reverse_mapping = {v: k for k, v in name_mappings.items()}
            for col in mcc_columns:
                if col in df.columns:
                    actual_columns.append(col)
                elif col in reverse_mapping and reverse_mapping[col] in df.columns:
                    actual_columns.append(reverse_mapping[col])
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame, skipping")
        elif mcc_columns:
            actual_columns = mcc_columns
        else:
            actual_columns = list(df.columns)

        # Filter to only MCC columns
        if actual_columns:
            mcc_df = df[actual_columns]
        else:
            mcc_df = df

        fig, ax = plt.subplots(figsize=figsize)

        n_models = len(mcc_df.index)
        n_metrics = len(mcc_df.columns)

        width = 0.8 / n_models
        x = range(n_metrics)
        colors_list = plt.get_cmap("tab10").colors

        for i, model_name in enumerate(mcc_df.index):
            offsets = [v + (i - (n_models - 1) / 2) * width for v in x]

            bars = ax.bar(
                offsets,
                mcc_df.loc[model_name],
                width=width,
                label=model_name,
                color=colors_list[i % len(colors_list)],
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
        ax.set_xticklabels(mcc_df.columns)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(mcc_df.max()) * 1.15)
        ax.set_title(title)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved MCC metrics bar plot to: {output_path}")

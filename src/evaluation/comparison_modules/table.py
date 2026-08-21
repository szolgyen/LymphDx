"""Table figure generation module."""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors

from .base import BaseFigure


class TableFigure(BaseFigure):
    """Generate a heatmap table figure with color-coded cells."""

    def generate(self, df: pd.DataFrame, name_mappings: dict = None) -> None:
        """
        Generate and export the table figure.

        Args:
            df: DataFrame containing the performance metrics.
            name_mappings: Dictionary for mapping metric names to display names.
        """
        output_path = self.config["output"]
        font_size = self.config.get("font_size", 10)
        scale_x = self.config.get("scale_x", 1.2)
        scale_y = self.config.get("scale_y", 2)
        dpi = self.config.get("dpi", 300)
        table_columns = self.config.get("columns", [])

        # Resolve column names: map display names to actual DataFrame column names
        actual_columns = []
        if table_columns and name_mappings:
            # Create reverse mapping from display name to actual column name
            reverse_mapping = {v: k for k, v in name_mappings.items()}
            for col in table_columns:
                # Try the column as-is first (in case it's already the correct name)
                if col in df.columns:
                    actual_columns.append(col)
                # Then try to find it via reverse mapping
                elif col in reverse_mapping and reverse_mapping[col] in df.columns:
                    actual_columns.append(reverse_mapping[col])
                # If still not found, skip it
                else:
                    print(f"Warning: Column '{col}' not found in DataFrame, skipping")
        elif table_columns:
            actual_columns = table_columns
        else:
            actual_columns = list(df.columns)

        # Filter to only resolved columns
        if actual_columns:
            display_df = df[actual_columns].copy()
        else:
            display_df = df.copy()

        text_df = display_df.map(lambda x: f"{x:.4f}" if pd.notna(x) else "")

        fig_width = max(8, len(display_df.columns) * 2.5)
        fig_height = max(2.5, len(display_df.index) * 0.8)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis("off")

        table = ax.table(
            cellText=text_df.values,
            rowLabels=text_df.index,
            colLabels=text_df.columns,
            cellLoc="center",
            loc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(font_size)
        table.scale(scale_x, scale_y)

        cmap = plt.cm.RdYlGn
        norm = colors.Normalize(vmin=0, vmax=1)

        # Color data cells
        for row_idx in range(len(display_df.index)):
            for col_idx in range(len(display_df.columns)):
                value = display_df.iloc[row_idx, col_idx]
                cell = table[(row_idx + 1, col_idx)]

                if pd.notna(value):
                    cell.set_facecolor(cmap(norm(value)))
                else:
                    cell.set_facecolor("#DDDDDD")

        # Style headers
        for col_idx in range(len(display_df.columns)):
            table[(0, col_idx)].set_facecolor("#404040")
            table[(0, col_idx)].set_text_props(color="white", weight="bold")

        for row_idx in range(len(display_df.index)):
            table[(row_idx + 1, -1)].set_facecolor("#404040")
            table[(row_idx + 1, -1)].set_text_props(color="white", weight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()

        print(f"Saved table to: {output_path}")

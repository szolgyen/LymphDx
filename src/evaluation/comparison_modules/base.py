"""Abstract base class for figure generation."""

from abc import ABC, abstractmethod
import pandas as pd


class BaseFigure(ABC):
    """Abstract base class for generating comparison figures."""

    def __init__(self, config: dict):
        """
        Initialize figure with configuration.

        Args:
            config: Configuration dictionary containing figure-specific settings.
        """
        self.config = config

    @abstractmethod
    def generate(self, df: pd.DataFrame) -> None:
        """
        Generate the figure from a dataframe.

        Args:
            df: DataFrame containing the performance metrics.
        """
        pass

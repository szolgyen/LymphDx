"""Figure generation modules for performance comparison."""

from .table import TableFigure
from .grouped_bar import GroupedBarFigure
from .mcc_metrics import MCCMetricsFigure
from .accuracy_given_correct import AccuracyGivenCorrectFigure
from .error_analysis import ErrorAnalysisFigure
from .group_accuracy import GroupAccuracyFigure
from .group_breakdown_matrices import GroupBreakdownMatricesFigure

__all__ = [
    "TableFigure",
    "GroupedBarFigure",
    "MCCMetricsFigure",
    "AccuracyGivenCorrectFigure",
    "ErrorAnalysisFigure",
    "GroupAccuracyFigure",
    "GroupBreakdownMatricesFigure",
]

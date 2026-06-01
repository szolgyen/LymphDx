import argparse
from pathlib import Path

from pathology_llm.postprocessing.excel_merge import merge_predictions_with_input


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge predictions.jsonl into input.xlsx by case id and export an enriched Excel file"
        )
    )
    parser.add_argument(
        "--input-excel",
        default="configs/extraction/NM_Surgical_Path_LN_final.xlsx",
        help="Path to source Excel file with Number column",
    )
    parser.add_argument(
        "--predictions-jsonl",
        default="outputs/predictions/predictions.jsonl",
        help="Path to predictions JSONL file with case_id",
    )
    parser.add_argument(
        "--output-excel",
        default="outputs/excel/predictions_merged.xlsx",
        help="Output Excel path",
    )
    parser.add_argument(
        "--input-id-column",
        default="Number",
        help="ID column name in input Excel",
    )
    parser.add_argument(
        "--prediction-id-key",
        default="case_id",
        help="ID key in JSONL prediction records",
    )
    parser.add_argument(
        "--missing-value",
        default="NA",
        help="Value to use when input case has no matched prediction",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> int:
    args = parse_args()
    merge_predictions_with_input(
        input_excel=Path(args.input_excel),
        predictions_jsonl=Path(args.predictions_jsonl),
        output_excel=Path(args.output_excel),
        input_id_column=args.input_id_column,
        prediction_id_key=args.prediction_id_key,
        missing_value=args.missing_value,
    )
    print(f"Wrote merged Excel: {args.output_excel}")
    return 0

import argparse
from pathlib import Path

from pathology_llm.postprocessing.excel_merge import (
    merge_predictions_into_validation_template,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge predictions.jsonl into validation template Excel by case id and "
            "container label, filling predicted columns and generating match markers."
        )
    )
    parser.add_argument(
        "--input-excel",
        default="outputs/excel/NM_Surgical_Path_LN_validation_template.xlsx",
        help="Path to validation template Excel file",
    )
    parser.add_argument(
        "--predictions-jsonl",
        default="outputs/predictions/guidance/predictions.jsonl",
        help="Path to predictions JSONL file with case_id",
    )
    parser.add_argument(
        "--output-excel",
        default="outputs/excel/NM_Surgical_Path_LN_validation_filled.xlsx",
        help="Output Excel path",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main() -> int:
    args = parse_args()
    merge_predictions_into_validation_template(
        input_excel=Path(args.input_excel),
        predictions_jsonl=Path(args.predictions_jsonl),
        output_excel=Path(args.output_excel),
    )
    print(f"Wrote merged Excel: {args.output_excel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

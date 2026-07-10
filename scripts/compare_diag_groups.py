#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def normalize(value):
    """Normalize values for case-insensitive, whitespace-insensitive comparison."""
    if value is None or pd.isna(value):
        return None
    return str(value).strip().casefold()


def equal(gt, pred):
    """Return True if two values match after normalization."""
    gt_norm = normalize(gt)
    pred_norm = normalize(pred)

    if gt_norm is None or pred_norm is None:
        return False

    return gt_norm == pred_norm


def load_predictions(jsonl_path):
    """Load predictions keyed by case_id."""
    predictions = {}

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            case_id = record.get("case_id")
            if case_id is None:
                continue

            predictions[case_id] = {
                "diagnosis_group_1": record.get("diagnosis_group_1"),
                "diagnosis_group_2": record.get("diagnosis_group_2"),
                "diagnosis_group_3": record.get("diagnosis_group_3"),
            }

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Compare predicted diagnosis groups against ground truth."
    )

    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path to prediction JSONL file.",
    )

    parser.add_argument(
        "--excel",
        required=True,
        help="Path to Excel file containing the Results sheet.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output Excel file.",
    )

    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Process all Excel rows instead of filtering Container Duplicate == True.",
    )

    args = parser.parse_args()

    predictions = load_predictions(args.jsonl)

    df = pd.read_excel(args.excel, sheet_name="Results")

    # Optionally filter to duplicate container rows
    if not args.include_all:
        df = df[df["Container Duplicate"] == True].copy()

    output_rows = []

    for _, row in df.iterrows():
        case_id = row["Id"]

        prediction = predictions.get(case_id)

        pred_g1 = prediction["diagnosis_group_1"] if prediction else None
        pred_g2 = prediction["diagnosis_group_2"] if prediction else None
        pred_g3 = prediction["diagnosis_group_3"] if prediction else None

        output_rows.append(
            {
                "Id": case_id,
                "GT Report Diagnosis Group 1": row["GT Report Diagnosis Group 1"],
                "Predicted Report Diagnosis Group 1": pred_g1,
                "Correct Report Diagnosis Group 1": equal(
                    row["GT Report Diagnosis Group 1"], pred_g1
                ),
                "GT Report Diagnosis Group 2": row["GT Report Diagnosis Group 2"],
                "Predicted Report Diagnosis Group 2": pred_g2,
                "Correct Report Diagnosis Group 2": equal(
                    row["GT Report Diagnosis Group 2"], pred_g2
                ),
                "GT Report Diagnosis Group 3": row["GT Report Diagnosis Group 3"],
                "Predicted Report Diagnosis Group 3": pred_g3,
                "Correct Report Diagnosis Group 3": equal(
                    row["GT Report Diagnosis Group 3"], pred_g3
                ),
            }
        )

    output_df = pd.DataFrame(
        output_rows,
        columns=[
            "Id",
            "GT Report Diagnosis Group 1",
            "Predicted Report Diagnosis Group 1",
            "Correct Report Diagnosis Group 1",
            "GT Report Diagnosis Group 2",
            "Predicted Report Diagnosis Group 2",
            "Correct Report Diagnosis Group 2",
            "GT Report Diagnosis Group 3",
            "Predicted Report Diagnosis Group 3",
            "Correct Report Diagnosis Group 3",
        ],
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_df.to_excel(output_path, index=False)

    print(f"Wrote {len(output_df)} rows to {output_path}")


if __name__ == "__main__":
    main()

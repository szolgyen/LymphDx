import pandas as pd
from transformers import pipeline
import torch
import argparse


def load_deid_model():
    print("Loading de-identification model...")

    device = 0 if torch.cuda.is_available() else -1

    deid_pipeline = pipeline(
        "token-classification",
        model="obi/deid_roberta_i2b2",
        aggregation_strategy="simple",
        device=device,
    )
    return deid_pipeline


def read_excel_file(input_file, text_column):

    print("Reading input Excel file...")
    df = pd.read_excel(input_file, engine="openpyxl")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input file.")

    return df


def deidentify_text(text, deid_pipeline):
    """
    Replace detected PHI entities with tags.
    """
    if pd.isna(text):
        return ""

    entities = deid_pipeline(text)

    result = text

    for entity in sorted(entities, key=lambda x: x["start"], reverse=True):
        start = entity["start"]
        end = entity["end"]
        label = entity["entity_group"]

        tag = f"[{label}]"  # Example: [NAME], [DATE], etc.
        result = result[:start] + tag + result[end:]

    return result


def process_reports(df, deid_pipeline, text_column):

    print("Processing reports...")

    deidentified_reports = []

    for i, report in enumerate(df[text_column]):
        print(f"Processing report {i + 1}/{len(df)}")
        try:
            cleaned = deidentify_text(str(report), deid_pipeline)
        except Exception as e:
            print(f"Error processing report {i}: {e}")
            cleaned = ""
        deidentified_reports.append(cleaned)

    return deidentified_reports


def save_output(df, deidentified_reports, output_file):
    df["Deidentified_Reports"] = deidentified_reports

    print("Saving output Excel file...")
    df.to_excel(output_file, index=False, engine="openpyxl")

    print("Done! Output saved to:", output_file)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "De-identify pathology reports in an Excel file using a pre-trained model, replacing PHI with tags."
        )
    )
    parser.add_argument(
        "--input-excel",
        default="NM_Surgical_Path_LN_validation.xlsx",
        help="Path to Excel file with reports having PHI",
    )
    parser.add_argument(
        "--output-excel",
        default="configs/extraction/deidentified_reports.xlsx",
        help="Path to deidentified reports Excel file",
    )
    parser.add_argument(
        "--text-column",
        default="Report",
        help="Name of the column containing the reports",
    )
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def main():
    args = parse_args()

    input_file = args.input_excel
    output_file = args.output_excel
    text_column = args.text_column

    deid_pipeline = load_deid_model()

    df = read_excel_file(input_file, text_column)
    deidentified_reports = process_reports(df, deid_pipeline, text_column)
    save_output(df, deidentified_reports, output_file)


if __name__ == "__main__":
    raise SystemExit(main())

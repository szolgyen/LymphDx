import json
import logging
from pathlib import Path
import pandas as pd

from preprocessing.data_parsing import (
    ParsedReport,
    load_reports_from_excel,
)


logger = logging.getLogger(__name__)


def load_reports(path: str) -> list[ParsedReport]:
    source_path = Path(path)
    if source_path.suffix.lower() == ".xlsx":
        reports = load_reports_from_excel(source_path)
        logger.info("Loaded %d reports from %s", len(reports), path)
        return reports

    lines = source_path.read_text(encoding="utf-8").splitlines()
    reports = [
        ParsedReport(case_id=None, text=line.strip()) for line in lines if line.strip()
    ]
    if not reports:
        raise ValueError(f"No reports found in {path}")
    logger.info("Loaded %d reports from %s", len(reports), path)
    return reports


def load_diagnosis_terms(path: str | Path) -> set[str]:
    """Load constrained diagnosis terms from an Excel file.

    Expected format: a column named 'Code' containing one diagnosis term
    per row. Empty values are ignored.
    """
    df = pd.read_excel(path, engine="openpyxl")

    if "Diagnosis" not in df.columns:
        raise ValueError(f"Column 'Diagnosis' not found in {path}")

    terms = {
        str(value).strip()
        for value in df["Diagnosis"]
        if pd.notna(value) and str(value).strip()
    }

    if not terms:
        raise ValueError(f"No diagnosis terms loaded from {path}")

    logger.info("Loaded %d diagnosis terms from %s", len(terms), path)
    return terms


def write_outputs(output_dir: str, outputs: list[dict]) -> None:
    prepare_output_store(output_dir)
    for idx, item in enumerate(outputs, start=1):
        write_output_record(output_dir, idx, item)
    logger.info("Wrote %d outputs to %s", len(outputs), output_dir)


def prepare_output_store(output_dir: str, timestamp: str | None = None) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for case_file in out_path.glob("case_*.json"):
        case_file.unlink()
    for case_prompt in out_path.glob("case_*.txt"):
        case_prompt.unlink()

    predictions_filename = (
        f"predictions_{timestamp}.jsonl" if timestamp else "predictions.jsonl"
    )
    jsonl_path = out_path / predictions_filename
    jsonl_path.write_text("", encoding="utf-8")

    broken_predictions_filename = (
        f"predictions_broken_{timestamp}.jsonl"
        if timestamp
        else "predictions_broken.jsonl"
    )
    broken_jsonl_path = out_path / broken_predictions_filename
    broken_jsonl_path.write_text("", encoding="utf-8")


def write_output_record(
    output_dir: str, report_index: int, output: dict, timestamp: str | None = None
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / f"case_{report_index:04d}.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    predictions_filename = (
        f"predictions_{timestamp}.jsonl" if timestamp else "predictions.jsonl"
    )
    with (out_path / predictions_filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(output) + "\n")


def write_prompt_record(output_dir: str, report_index: int, prompt: str) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / f"case_{report_index:04d}.txt").write_text(
        prompt,
        encoding="utf-8",
    )


def write_broken_extraction_record(
    output_dir: str,
    report_index: int,
    raw_output: str | None,
    error_message: str,
    timestamp: str | None = None,
) -> None:
    """Write broken/unparseable extraction to a separate file for debugging."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    broken_record = {
        "report_index": report_index,
        "error": error_message,
        "raw_output": raw_output,
    }

    broken_predictions_filename = (
        f"predictions_broken_{timestamp}.jsonl"
        if timestamp
        else "predictions_broken.jsonl"
    )
    with (out_path / broken_predictions_filename).open("a", encoding="utf-8") as f:
        f.write(json.dumps(broken_record) + "\n")

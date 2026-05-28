import json
import logging
from pathlib import Path

from pathology_llm.preprocessing.data_parsing import (
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
    """Load constrained diagnosis terms from a text file.

    Expected format: one diagnosis term per line; empty lines and lines
    starting with '#' are ignored.
    """
    terms: set[str] = set()
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        terms.add(line)

    if not terms:
        raise ValueError(f"No diagnosis terms loaded from {path}")

    logger.info("Loaded %d diagnosis terms from %s", len(terms), path)
    return terms


def write_outputs(output_dir: str, outputs: list[dict]) -> None:
    prepare_output_store(output_dir)
    for idx, item in enumerate(outputs, start=1):
        write_output_record(output_dir, idx, item)
    logger.info("Wrote %d outputs to %s", len(outputs), output_dir)


def prepare_output_store(output_dir: str) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for case_file in out_path.glob("case_*.json"):
        case_file.unlink()

    jsonl_path = out_path / "predictions.jsonl"
    jsonl_path.write_text("", encoding="utf-8")


def write_output_record(output_dir: str, report_index: int, output: dict) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / f"case_{report_index:04d}.json").write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    with (out_path / "predictions.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(output) + "\n")

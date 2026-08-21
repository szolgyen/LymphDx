import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd

from utils.logging_config import configure_logging

logger = logging.getLogger(__name__)


def build_lookup(ontology_path):
    df = pd.read_excel(ontology_path)
    logger.info("Loaded ontology from %s", ontology_path)

    if "Code" not in df.columns or "Diagnosis" not in df.columns:
        logger.error("Ontology must contain 'Code' and 'Diagnosis' columns")
        raise ValueError("Ontology must contain 'Code' and 'Diagnosis' columns")

    lookup = {}

    for _, row in df.iterrows():
        code = row["Code"]
        diagnosis = row["Diagnosis"]

        if pd.isna(code) or pd.isna(diagnosis):
            continue

        key = str(diagnosis).strip().lower()
        lookup[key] = code

    return lookup


def map_reports(reports_path, lookup, output_path):
    df = pd.read_excel(reports_path)
    logger.info("Loaded reports from %s", reports_path)

    col = "GT Report Diagnosis"

    if col not in df.columns:
        logger.error("Missing column: %s", col)
        raise ValueError(f"Missing column: {col}")

    def map_code(x):
        if pd.isna(x):
            return None
        return lookup.get(str(x).strip().lower())

    df["Code"] = df[col].apply(map_code)

    df.to_excel(output_path, index=False)
    logger.info("Wrote mapped reports to %s", output_path)


def script_main() -> int:
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = Path("outputs/add_codes") / f"add_codes_{timestamp}.log"
        configure_logging("INFO", log_file=str(log_file))

        parser = argparse.ArgumentParser()
        parser.add_argument("--ontology-file", required=True)
        parser.add_argument("--reports-file", required=True)
        parser.add_argument("--output-file", required=True)

        args = parser.parse_args()

        logger.info("Starting diagnosis code mapping")
        lookup = build_lookup(args.ontology_file)
        map_reports(args.reports_file, lookup, args.output_file)
        logger.info("Diagnosis code mapping completed successfully")
        return 0
    except Exception as exc:
        logger.exception("Diagnosis code mapping failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(script_main())

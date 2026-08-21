"""IO utilities for evaluation module.

Handles configuration loading and data file reading/writing.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file does not exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        logger.error("Configuration file not found: %s", config_path)
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file) as f:
        config = yaml.safe_load(f)

    logger.info("Loaded configuration from %s", config_path)
    return config


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load newline-delimited JSON records from file.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON records.
    """
    records: list[dict[str, Any]] = []

    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    logger.info("Loaded %d records from %s", len(records), path)
    return records


def load_ground_truth_excel(path: Path, sheet_name: str) -> any:
    """Load ground-truth annotations from Excel file.

    Args:
        path: Path to the Excel file.
        sheet_name: Name of the sheet to load.

    Returns:
        DataFrame with ground-truth annotations.
    """
    import pandas as pd

    return pd.read_excel(path, sheet_name=sheet_name)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON data to file with stable indentation.

    Args:
        data: Dictionary to write.
        path: Output file path.
    """
    with path.open("w") as f:
        json.dump(data, f, indent=2)




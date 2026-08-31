"""Prediction data extractors for accessing and transforming prediction records.

Provides functions to extract diagnosis information from prediction records with clear,
descriptive names following domain terminology.
"""

from typing import Any

import pandas as pd


def extract_valid_primary_diagnoses(record: dict[str, Any]) -> dict[str, Any]:
    """Extract the dictionary of top-k valid primary diagnoses from a prediction record.

    Args:
        record: Single prediction record.

    Returns:
        Dictionary containing ranked primary diagnoses.
    """
    return record.get("valid_primary_diagnoses", {})


def extract_top_1_primary_diagnosis(record: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the top-1 ranked valid primary diagnosis from a prediction record.

    Returns the prediction dictionary for the top-ranked diagnosis:
    {
        "valid_primary_diagnosis_code": str,
        "valid_primary_diagnosis_name": str,
        "valid_primary_diagnosis_score": float,
        "valid_primary_diagnosis_group_1": str,
        "valid_primary_diagnosis_group_2": str,
        "valid_primary_diagnosis_group_3": str
    }

    Args:
        record: Single prediction record.

    Returns:
        Top-1 diagnosis dictionary or None if not found.
    """
    return extract_valid_primary_diagnoses(record).get("top_1")


def extract_top_k_primary_diagnosis_groups(
    record: dict[str, Any], k: int, group_key: str
) -> list[str] | None:
    """Extract diagnosis group categories for top-k ranked predictions.

    Args:
        record: Single prediction record.
        k: Number of top predictions to extract (e.g., 3, 5).
        group_key: Group key like "group_1", "group_2", or "group_3".

    Returns:
        List of group categories for top-k predictions, or None if key is invalid.
    """
    # Normalize group key: "group_1" -> "valid_primary_diagnosis_group_1"
    column_name = f"valid_primary_diagnosis_{group_key}"

    diagnoses = extract_valid_primary_diagnoses(record)
    groups = []

    for rank in range(1, k + 1):
        item = diagnoses.get(f"top_{rank}")
        if item:
            group_value = item.get(column_name)
            groups.append(group_value)

    return groups if groups else None


def extract_top_k_primary_diagnosis_codes(record: dict[str, Any], k: int) -> list[int]:
    """Extract diagnosis codes for top-k ranked predictions.

    Args:
        record: Single prediction record.
        k: Number of top predictions to extract.

    Returns:
        List of diagnosis codes for top-k ranked predictions.
    """
    diagnoses = extract_valid_primary_diagnoses(record)
    codes: list[int] = []

    for rank in range(1, k + 1):
        item = diagnoses.get(f"top_{rank}")
        if item:
            code = item.get("valid_primary_diagnosis_code")
            if code is not None:
                codes.append(code)

    return codes


def extract_container_diagnosis(record: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the container diagnosis from a prediction record.

    Returns the container dictionary containing valid_diagnoses.

    Args:
        record: Single prediction record.

    Returns:
        Container diagnosis dictionary or None if not found.
    """
    # Check for new nested structure first
    if "container" in record:
        container = record.get("container")
        if isinstance(container, dict):
            return container
    # Fallback to containers (plural)
    elif "containers" in record:
        containers = record.get("containers")
        if isinstance(containers, list) and len(containers) > 0:
            return containers[0]
    return None


def extract_top_1_container_diagnosis(record: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the top-1 ranked container diagnosis from a prediction record.

    Args:
        record: Single prediction record.

    Returns:
        Top-1 container diagnosis dictionary or None if not found.
    """
    container = extract_container_diagnosis(record)
    if not container:
        return None

    valid_diagnoses = container.get("valid_diagnoses", {})
    if isinstance(valid_diagnoses, dict):
        return valid_diagnoses.get("top_1")
    return None


def extract_top_k_container_diagnosis_codes(
    record: dict[str, Any], k: int
) -> list[int]:
    """Extract container diagnosis codes for top-k ranked predictions.

    Args:
        record: Single prediction record.
        k: Number of top predictions to extract.

    Returns:
        List of diagnosis codes for top-k container predictions.
    """
    container = extract_container_diagnosis(record)
    if not container:
        return []

    valid_diagnoses = container.get("valid_diagnoses", {})
    if not isinstance(valid_diagnoses, dict):
        return []

    codes: list[int] = []
    for rank in range(1, k + 1):
        item = valid_diagnoses.get(f"top_{rank}")
        if item and isinstance(item, dict):
            code = item.get("valid_diagnosis_code")
            if code is not None:
                codes.append(code)

    return codes


def extract_boolean_field(value: any) -> bool:
    """Convert various value types to boolean.

    Handles strings ("true", "yes", "1"), booleans, numpy booleans, and NaN values.

    Args:
        value: Value to convert (can be str, bool, numpy bool, or NaN).

    Returns:
        Boolean value.

    Raises:
        ValueError: If value cannot be meaningfully converted to boolean.
    """
    import numpy as np
    import logging

    logger = logging.getLogger(__name__)

    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    # Handle numpy numeric types (e.g., numpy.float64, numpy.int64)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)

    if not isinstance(value, str):
        logger.error("Expected a string or boolean, got %s", type(value))
        raise ValueError(f"Expected a string or boolean, got {type(value)}")

    value_lower = value.strip().lower()

    if value_lower in {"true", "1", "yes"}:
        return True
    elif value_lower in {"false", "0", "no"}:
        return False
    else:
        logger.error("Cannot convert string to boolean: %s", value)
        raise ValueError(f"Cannot convert string to boolean: {value}")


def build_prediction_lookup_map(
    records: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Build a lookup map from case_id to prediction record.

    Args:
        records: List of prediction records.

    Returns:
        Dictionary mapping case_id (int) to prediction record.
    """
    return {int(record["case_id"]): record for record in records}

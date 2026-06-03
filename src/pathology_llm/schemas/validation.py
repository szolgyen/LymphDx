import json
from collections.abc import Iterable
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError

from pathology_llm.schemas.pathology import PathologyExtractionV2


class SchemaValidationError(Exception):
    pass


class DiagnosisConstraintError(SchemaValidationError):
    pass


_DIAGNOSIS_CONSTRAINT_META_KEY = "x-reportllm-diagnosis-constrained"


def _flatten_constrained_values(value: Any, field_path: str) -> list[tuple[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        value_clean = value.strip()
        return [(field_path, value_clean)] if value_clean else []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        flattened: list[tuple[str, str]] = []
        for idx, item in enumerate(value):
            flattened.extend(_flatten_constrained_values(item, f"{field_path}[{idx}]") )
        return flattened
    return []


def _iter_constrained_values(
    extraction: BaseModel,
    prefix: str = "",
) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for field_name, field_info in extraction.__class__.model_fields.items():
        field_path = f"{prefix}.{field_name}" if prefix else field_name
        value = getattr(extraction, field_name, None)

        extras = field_info.json_schema_extra
        if isinstance(extras, dict) and extras.get(_DIAGNOSIS_CONSTRAINT_META_KEY):
            found.extend(_flatten_constrained_values(value, field_path))

        if isinstance(value, BaseModel):
            found.extend(_iter_constrained_values(value, prefix=field_path))
            continue

        if isinstance(value, list):
            for idx, item in enumerate(value):
                if isinstance(item, BaseModel):
                    found.extend(
                        _iter_constrained_values(item, prefix=f"{field_path}[{idx}]")
                    )

    return found


def _enforce_diagnosis_constraints(
    extraction: BaseModel,
    allowed_diagnoses: set[str],
) -> None:
    invalid_values = [
        (field_path, term)
        for field_path, term in _iter_constrained_values(extraction)
        if term not in allowed_diagnoses
    ]
    if invalid_values:
        raise DiagnosisConstraintError(
            "diagnosis-constrained fields contain values outside constrained diagnosis set: "
            f"{invalid_values!r}"
        )


def _normalize_differential_consistency(extraction: PathologyExtractionV2) -> None:
    # Keep only non-empty differential entries and remove any duplicate of primary diagnosis.
    primary = extraction.primary_diagnosis
    deduped: list[str] = []
    seen: set[str] = set()
    for term in extraction.differential_diagnoses:
        term_clean = term.strip()
        if not term_clean:
            continue
        if primary and term_clean == primary:
            continue
        if term_clean in seen:
            continue
        seen.add(term_clean)
        deduped.append(term_clean)

    extraction.differential_diagnoses = deduped
    extraction.has_differential_diagnosis = bool(deduped)


def _normalize_if_supported(extraction: BaseModel) -> None:
    if not isinstance(extraction, PathologyExtractionV2):
        return
    _normalize_differential_consistency(extraction)


def validate_pathology_output(
    raw: str | dict,
    allowed_diagnoses: set[str] | None = None,
) -> PathologyExtractionV2:
    """
    Validates model output against PathologyExtractionV2 schema.

    Input:
        raw: JSON string or dict from LLM
        allowed_diagnoses: optional constrained set for primary/secondary diagnosis terms

    Output:
        PathologyExtractionV2 (validated)

    Raises:
        SchemaValidationError if invalid
    """

    try:
        # normalize input
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        # validate via Pydantic
        extraction = PathologyExtractionV2.model_validate(data)

        _normalize_if_supported(extraction)

        # strict constrained decoding contract: no ontology mapping fallback
        if allowed_diagnoses is not None:
            _enforce_diagnosis_constraints(extraction, allowed_diagnoses)

        return extraction

    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}")

    except ValidationError as e:
        raise SchemaValidationError(f"Schema mismatch: {e}")


def validate_output(
    raw: str | dict,
    schema_model: type[BaseModel],
    allowed_diagnoses: set[str] | None = None,
) -> BaseModel:
    try:
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        extraction = schema_model.model_validate(data)
        _normalize_if_supported(extraction)

        if allowed_diagnoses is not None:
            _enforce_diagnosis_constraints(extraction, allowed_diagnoses)

        return extraction
    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}")
    except ValidationError as e:
        raise SchemaValidationError(f"Schema mismatch: {e}")

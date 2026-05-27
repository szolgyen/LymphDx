import json
from pydantic import ValidationError

from pathology_llm.schemas.pathology import PathologyExtraction


class SchemaValidationError(Exception):
    pass


class DiagnosisConstraintError(SchemaValidationError):
    pass


def _enforce_diagnosis_constraints(
    extraction: PathologyExtraction,
    allowed_diagnoses: set[str],
) -> None:
    if (
        extraction.primary_diagnosis
        and extraction.primary_diagnosis not in allowed_diagnoses
    ):
        raise DiagnosisConstraintError(
            "primary_diagnosis is outside constrained diagnosis set: "
            f"{extraction.primary_diagnosis!r}"
        )

    invalid_secondary = [
        term
        for term in extraction.differential_diagnoses
        if term not in allowed_diagnoses
    ]
    if invalid_secondary:
        raise DiagnosisConstraintError(
            "differential_diagnoses contains values outside constrained diagnosis set: "
            f"{invalid_secondary!r}"
        )


def _normalize_differential_consistency(extraction: PathologyExtraction) -> None:
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


def validate_pathology_output(
    raw: str | dict,
    allowed_diagnoses: set[str] | None = None,
) -> PathologyExtraction:
    """
    Validates model output against PathologyExtraction schema.

    Input:
        raw: JSON string or dict from LLM
        allowed_diagnoses: optional constrained set for primary/secondary diagnosis terms

    Output:
        PathologyExtraction (validated)

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
        extraction = PathologyExtraction.model_validate(data)

        _normalize_differential_consistency(extraction)

        # strict constrained decoding contract: no ontology mapping fallback
        if allowed_diagnoses is not None:
            _enforce_diagnosis_constraints(extraction, allowed_diagnoses)

        return extraction

    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}")

    except ValidationError as e:
        raise SchemaValidationError(f"Schema mismatch: {e}")

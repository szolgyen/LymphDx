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
        extraction.diagnosis_primary
        and extraction.diagnosis_primary not in allowed_diagnoses
    ):
        raise DiagnosisConstraintError(
            "diagnosis_primary is outside constrained diagnosis set: "
            f"{extraction.diagnosis_primary!r}"
        )

    invalid_secondary = [
        term for term in extraction.diagnosis_secondary if term not in allowed_diagnoses
    ]
    if invalid_secondary:
        raise DiagnosisConstraintError(
            "diagnosis_secondary contains values outside constrained diagnosis set: "
            f"{invalid_secondary!r}"
        )


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

        # strict constrained decoding contract: no ontology mapping fallback
        if allowed_diagnoses is not None:
            _enforce_diagnosis_constraints(extraction, allowed_diagnoses)

        return extraction

    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}")

    except ValidationError as e:
        raise SchemaValidationError(f"Schema mismatch: {e}")

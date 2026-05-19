import pytest

from pathology_llm.schemas.validation import (
    DiagnosisConstraintError,
    validate_pathology_output,
)


def test_validation_accepts_allowed_diagnosis_terms() -> None:
    raw = {
        "schema_version": "v1",
        "diagnosis_primary": "Adenocarcinoma",
        "diagnosis_secondary": ["Reactive lymphoid hyperplasia"],
        "specimen": "Colon biopsy",
        "is_lymph_node": False,
        "anatomic_location": "Colon",
        "biomarkers": [{"name": "KRAS", "value": "mutated"}],
        "confidence": 0.82,
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    obj = validate_pathology_output(raw, allowed_diagnoses=allowed)

    assert obj.diagnosis_primary == "Adenocarcinoma"
    assert obj.diagnosis_secondary == ["Reactive lymphoid hyperplasia"]


def test_validation_rejects_disallowed_primary_diagnosis() -> None:
    raw = {
        "schema_version": "v1",
        "diagnosis_primary": "Invasive component",
        "diagnosis_secondary": [],
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    with pytest.raises(DiagnosisConstraintError):
        validate_pathology_output(raw, allowed_diagnoses=allowed)


def test_validation_rejects_disallowed_secondary_diagnosis() -> None:
    raw = {
        "schema_version": "v1",
        "diagnosis_primary": "Adenocarcinoma",
        "diagnosis_secondary": ["Invasive component"],
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    with pytest.raises(DiagnosisConstraintError):
        validate_pathology_output(raw, allowed_diagnoses=allowed)

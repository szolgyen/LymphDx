import pytest
from pydantic import BaseModel, Field

from schemas.registry import get_schema_model
from schemas.validation import (
    DiagnosisConstraintError,
    SchemaValidationError,
    validate_output,
)


def test_validation_accepts_allowed_diagnosis_terms() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": True,
        "differential_diagnoses": ["Reactive lymphoid hyperplasia"],
        "specimen": "Colon biopsy",
        "is_lymph_node": False,
        "is_definitive": True,
        "has_prior_malignancy": None,
        "has_concurrent_malignancy": None,
        "anatomic_location": "Colon",
        "biomarkers": [{"name": "KRAS", "value": "mutated"}],
        "confidence": 0.82,
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    obj = validate_output(raw, schema_model=schema_v2, allowed_diagnoses=allowed)

    assert obj.primary_diagnosis == "Adenocarcinoma"
    assert obj.differential_diagnoses == ["Reactive lymphoid hyperplasia"]


def test_validation_rejects_disallowed_primary_diagnosis() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Invasive component",
        "has_differential_diagnosis": False,
        "differential_diagnoses": [],
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    with pytest.raises(DiagnosisConstraintError):
        validate_output(raw, schema_model=schema_v2, allowed_diagnoses=allowed)


def test_validation_rejects_disallowed_differential_diagnosis() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": True,
        "differential_diagnoses": ["Invasive component"],
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    with pytest.raises(DiagnosisConstraintError):
        validate_output(raw, schema_model=schema_v2, allowed_diagnoses=allowed)


def test_validation_rejects_primary_inside_differentials() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": True,
        "differential_diagnoses": ["Adenocarcinoma"],
    }

    obj = validate_output(raw, schema_model=schema_v2)

    assert obj.differential_diagnoses == []
    assert obj.has_differential_diagnosis is False


def test_validation_rejects_false_flag_with_non_empty_differentials() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": False,
        "differential_diagnoses": ["Reactive lymphoid hyperplasia"],
    }

    obj = validate_output(raw, schema_model=schema_v2)

    assert obj.differential_diagnoses == ["Reactive lymphoid hyperplasia"]
    assert obj.has_differential_diagnosis is True


def test_validation_rejects_true_flag_with_empty_differentials() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": True,
        "differential_diagnoses": [],
    }

    obj = validate_output(raw, schema_model=schema_v2)

    assert obj.differential_diagnoses == []
    assert obj.has_differential_diagnosis is False


def test_validation_rejects_null_flag_with_non_empty_differentials() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": None,
        "differential_diagnoses": ["Reactive lymphoid hyperplasia"],
    }

    obj = validate_output(raw, schema_model=schema_v2)

    assert obj.differential_diagnoses == ["Reactive lymphoid hyperplasia"]
    assert obj.has_differential_diagnosis is True


def test_validation_rejects_unsupported_schema_version() -> None:
    schema_v2 = get_schema_model("v2")
    raw = {
        "schema_version": "v1",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": False,
        "differential_diagnoses": [],
    }

    with pytest.raises(SchemaValidationError, match="Schema mismatch"):
        validate_output(raw, schema_model=schema_v2)


def test_validate_output_applies_constraints_for_v3_primary_diagnosis() -> None:
    raw = {
        "schema_version": "v3",
        "primary_diagnosis": "Disallowed diagnosis",
        "container": None,
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    from schemas.pathology import PathologyExtractionV3

    with pytest.raises(DiagnosisConstraintError):
        validate_output(
            raw, schema_model=PathologyExtractionV3, allowed_diagnoses=allowed
        )


def test_validate_output_does_not_constrain_v3_container_diagnosis() -> None:
    raw = {
        "schema_version": "v3",
        "primary_diagnosis": "Adenocarcinoma",
        "container": {
            "label": "A",
            "source": "Lung",
            "diagnosis": "Disallowed diagnosis",
        },
    }
    allowed = {"Adenocarcinoma", "Reactive lymphoid hyperplasia"}

    from schemas.pathology import PathologyExtractionV3

    obj = validate_output(
        raw, schema_model=PathologyExtractionV3, allowed_diagnoses=allowed
    )
    assert obj.primary_diagnosis == "Adenocarcinoma"
    assert obj.container is not None
    assert obj.container.diagnosis == "Disallowed diagnosis"


def test_validate_output_enforces_constraints_for_any_new_schema_with_metadata() -> (
    None
):
    class _SchemaWithConstrainedField(BaseModel):
        schema_version: str = "vx"
        diagnosis_like_field: str = Field(
            ..., json_schema_extra={"x-reportllm-diagnosis-constrained": True}
        )

    raw = {"schema_version": "vx", "diagnosis_like_field": "Unknown"}
    allowed = {"Known"}

    with pytest.raises(DiagnosisConstraintError):
        validate_output(
            raw, schema_model=_SchemaWithConstrainedField, allowed_diagnoses=allowed
        )

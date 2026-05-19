import pytest

from pathology_llm.inference.adapters.hf import HFAdapter
from pathology_llm.schemas.validation import (
    DiagnosisConstraintError,
    SchemaValidationError,
)


def _valid_payload(primary: str = "Adenocarcinoma") -> str:
    return f"""
    {{
        "schema_version": "v1",
        "diagnosis_primary": "{primary}",
        "diagnosis_secondary": [],
        "description": "Moderately differentiated tumor",
        "interpretation_status": "present",
        "specimen": "Colon biopsy",
        "is_lymph_node": false,
        "anatomic_location": "Colon",
        "container": null,
        "biomarkers": [
            {{"name": "KRAS", "value": "mutated"}}
        ],
        "confidence": 0.86
    }}
    """


def test_hf_adapter_parse_accepts_markdown_fenced_json():
    adapter = HFAdapter(
        model="fake-model",
        decoder="none",
        allowed_diagnoses={"Adenocarcinoma"},
    )
    raw = f"```json\n{_valid_payload()}\n```"

    extraction = adapter.parse(raw)

    assert extraction.diagnosis_primary == "Adenocarcinoma"


def test_hf_adapter_parse_rejects_disallowed_diagnosis():
    adapter = HFAdapter(
        model="fake-model",
        decoder="none",
        allowed_diagnoses={"DLBCL"},
    )

    with pytest.raises(DiagnosisConstraintError):
        adapter.parse(_valid_payload(primary="Adenocarcinoma"))


def test_hf_adapter_rejects_unsupported_decoder():
    with pytest.raises(ValueError):
        HFAdapter(model="fake-model", decoder="outlines")


def test_hf_adapter_parse_rejects_non_json():
    adapter = HFAdapter(model="fake-model", decoder="none")

    with pytest.raises(SchemaValidationError):
        adapter.parse("this is not json")

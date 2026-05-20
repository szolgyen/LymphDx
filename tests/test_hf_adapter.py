import pytest

from pathology_llm.inference.adapters.hf import HFAdapter
from pathology_llm.inference.decoders.hf_guidance import HFGuidanceDecoder
from pathology_llm.inference.decoders.outlines import OutlinesDecoder
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
        HFAdapter(model="fake-model", decoder="sglang")


def test_hf_adapter_parse_rejects_non_json():
    adapter = HFAdapter(model="fake-model", decoder="none")

    with pytest.raises(SchemaValidationError):
        adapter.parse("this is not json")


def test_hf_guidance_decoder_requires_runtime():
    decoder = HFGuidanceDecoder(
        allowed_diagnoses={"Adenocarcinoma"},
        runtime_available=False,
    )

    with pytest.raises(RuntimeError, match="requires the 'guidance' package"):
        decoder.validate_ready()


def test_hf_guidance_decoder_requires_allowed_diagnoses():
    decoder = HFGuidanceDecoder(runtime_available=True)

    with pytest.raises(ValueError, match="requires non-empty allowed_diagnoses"):
        decoder.validate_ready()


def test_hf_guidance_decoder_strict_prompt_contains_allowed_terms():
    decoder = HFGuidanceDecoder(
        allowed_diagnoses={"Adenocarcinoma", "DLBCL"},
        runtime_available=True,
    )

    prompt = decoder.prepare_prompt("Extract the report", tokenizer=None)

    assert "STRICT DECODER MODE" in prompt
    assert "- Adenocarcinoma" in prompt
    assert "- DLBCL" in prompt


def test_hf_guidance_decoder_overrides_sampling_kwargs():
    decoder = HFGuidanceDecoder(
        allowed_diagnoses={"Adenocarcinoma"},
        runtime_available=True,
    )

    kwargs = decoder.get_generation_kwargs(tokenizer=None)

    assert kwargs["do_sample"] is False
    assert kwargs["temperature"] is None


def test_hf_guidance_decoder_schema_constrains_diagnoses():
    decoder = HFGuidanceDecoder(
        allowed_diagnoses={"Adenocarcinoma", "DLBCL"},
        runtime_available=True,
    )

    schema = decoder._build_schema()

    primary_any_of = schema["properties"]["diagnosis_primary"]["anyOf"]
    primary_enum = next(item["enum"] for item in primary_any_of if "enum" in item)
    secondary_enum = schema["properties"]["diagnosis_secondary"]["items"]["enum"]
    assert primary_enum == ["Adenocarcinoma", "DLBCL"]
    assert secondary_enum == ["Adenocarcinoma", "DLBCL"]


def test_hf_adapter_uses_decoder_owned_generate_path(monkeypatch):
    adapter = HFAdapter(
        model="fake-model",
        decoder="none",
        allowed_diagnoses={"Adenocarcinoma"},
    )

    class _DecoderStub:
        def validate_ready(self):
            return None

        def generate(self, **kwargs):
            return _valid_payload(primary="Adenocarcinoma")

        def prepare_prompt(self, prompt, tokenizer):
            raise AssertionError(
                "prepare_prompt should not be used when decoder generates"
            )

        def get_generation_kwargs(self, tokenizer):
            return {}

    adapter._decoder = _DecoderStub()
    adapter._tokenizer = object()
    adapter._model = object()

    raw = adapter.generate("ignored prompt")

    assert "Adenocarcinoma" in raw


def test_outlines_decoder_requires_runtime():
    decoder = OutlinesDecoder(
        backend="hf",
        allowed_diagnoses={"Adenocarcinoma"},
        runtime_available=False,
    )

    with pytest.raises(RuntimeError, match="requires the 'outlines' package"):
        decoder.validate_ready()


def test_outlines_decoder_requires_allowed_diagnoses():
    decoder = OutlinesDecoder(backend="hf", runtime_available=True)

    with pytest.raises(ValueError, match="requires non-empty allowed_diagnoses"):
        decoder.validate_ready()


def test_outlines_decoder_not_implemented_for_unsupported_backend():
    decoder = OutlinesDecoder(
        backend="ollama",
        allowed_diagnoses={"Adenocarcinoma"},
        runtime_available=True,
    )

    with pytest.raises(NotImplementedError, match="backend='ollama'"):
        decoder.validate_ready()


def test_outlines_decoder_schema_constrains_diagnoses():
    decoder = OutlinesDecoder(
        backend="hf",
        allowed_diagnoses={"Adenocarcinoma", "DLBCL"},
        runtime_available=True,
    )

    schema = decoder._build_schema()

    primary_any_of = schema["properties"]["diagnosis_primary"]["anyOf"]
    primary_enum = next(item["enum"] for item in primary_any_of if "enum" in item)
    secondary_enum = schema["properties"]["diagnosis_secondary"]["items"]["enum"]
    assert primary_enum == ["Adenocarcinoma", "DLBCL"]
    assert secondary_enum == ["Adenocarcinoma", "DLBCL"]

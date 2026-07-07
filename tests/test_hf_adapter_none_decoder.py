from inference.adapters.hf import HFAdapter
from inference.decoders.none import NoneDecoder


def test_hf_adapter_none_decoder_allows_unconstrained_primary_diagnosis() -> None:
    adapter = HFAdapter(
        model="fake-model",
        decoder="none",
        allowed_diagnoses={"DLBCL"},
    )

    raw = """
    {
        "schema_version": "v2",
        "primary_diagnosis": "Adenocarcinoma",
        "has_differential_diagnosis": false,
        "differential_diagnoses": [],
        "specimen": null,
        "is_lymph_node": null,
        "is_definitive": null,
        "has_prior_malignancy": null,
        "has_concurrent_malignancy": null,
        "anatomic_location": null,
        "container": null,
        "biomarkers": [],
        "confidence": null
    }
    """

    extraction = adapter.parse(raw)

    assert extraction.primary_diagnosis == "Adenocarcinoma"


def test_hf_adapter_none_decoder_extracts_json_from_noisy_prefix() -> None:
    adapter = HFAdapter(
        model="fake-model",
        decoder="none",
        allowed_diagnoses={"DLBCL"},
    )

    raw = """
<unused94>thought
The user wants JSON only.
{
    "schema_version": "v2",
    "primary_diagnosis": "Adenocarcinoma",
    "has_differential_diagnosis": false,
    "differential_diagnoses": [],
    "specimen": null,
    "is_lymph_node": null,
    "is_definitive": null,
    "has_prior_malignancy": null,
    "has_concurrent_malignancy": null,
    "anatomic_location": null,
    "container": null,
    "biomarkers": [],
    "confidence": null
}
"""

    extraction = adapter.parse(raw)

    assert extraction.primary_diagnosis == "Adenocarcinoma"


def test_none_decoder_owns_json_prompt_guard() -> None:
    decoder = NoneDecoder(backend="hf", runtime_available=False)

    formatted = decoder.prepare_prompt("Extract JSON", tokenizer=None)

    assert "STRICT JSON MODE" in formatted
    assert "ALLOWED DIAGNOSES" not in formatted


def test_none_decoder_schema_is_json_constrained_but_not_diagnosis_constrained() -> (
    None
):
    decoder = NoneDecoder(backend="hf", runtime_available=False)

    schema = decoder._build_schema()

    primary_any_of = schema["properties"]["primary_diagnosis"]["anyOf"]
    assert all("enum" not in item for item in primary_any_of if isinstance(item, dict))
    assert "enum" not in schema["properties"]["differential_diagnoses"]["items"]


def test_none_decoder_returns_none_when_guidance_unavailable() -> None:
    decoder = NoneDecoder(backend="hf", runtime_available=False)

    result = decoder.generate(
        model=object(),
        tokenizer=object(),
        prompt="Extract JSON",
        max_new_tokens=64,
        temperature=0.0,
    )

    assert result is None

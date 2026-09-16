from pydantic import BaseModel

from inference.adapters.vllm import VLLMAdapter


class _Schema(BaseModel):
    primary_diagnosis: str


def _adapter(decoder: str = "none") -> VLLMAdapter:
    return VLLMAdapter(
        model="fake/model",
        decoder=decoder,
        allowed_diagnoses={"Adenocarcinoma"},
        schema_model=_Schema,
    )


def test_vllm_adapter_parses_plain_json():
    extraction = _adapter().parse('{"primary_diagnosis": "Adenocarcinoma"}')

    assert extraction.primary_diagnosis == "Adenocarcinoma"


def test_vllm_adapter_strips_reasoning_block():
    raw = '<unused94>thought\nweighing options<unused95>{"primary_diagnosis": "Adenocarcinoma"}'

    assert _adapter().parse(raw).primary_diagnosis == "Adenocarcinoma"


def test_vllm_adapter_strips_markdown_fence():
    raw = '```json\n{"primary_diagnosis": "Adenocarcinoma"}\n```'

    assert _adapter().parse(raw).primary_diagnosis == "Adenocarcinoma"


def test_vllm_adapter_extracts_json_from_noisy_prefix():
    raw = 'Here is the result:\n{"primary_diagnosis": "Adenocarcinoma"}\nThanks.'

    assert _adapter().parse(raw).primary_diagnosis == "Adenocarcinoma"


def test_vllm_adapter_omits_schema_when_decoder_is_none():
    assert _adapter("none")._build_extra_body() == {}


def test_vllm_adapter_sends_schema_when_decoder_requested():
    body = _adapter("outlines")._build_extra_body()

    assert body["response_format"]["type"] == "json_schema"

from types import ModuleType, SimpleNamespace
import sys

from pathology_llm.inference.adapters.vllm import VLLMAdapter
from pathology_llm.inference.decoders.outlines import OutlinesDecoder


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return messages[0]["content"]


class _FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_tokenizer(self):
        return _FakeTokenizer()

    def generate(self, prompts, sampling_params):
        assert len(prompts) == 1
        assert sampling_params.max_tokens == 512
        return [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text=(
                            '{"schema_version":"v1","diagnosis_primary":"Adenocarcinoma",'
                            '"diagnosis_secondary":[],"description":null,'
                            '"interpretation_status":null,"specimen":null,'
                            '"is_lymph_node":null,"anatomic_location":null,'
                            '"container":null,"biomarkers":[],"confidence":0.9}'
                        )
                    )
                ]
            )
        ]


class _FakeSamplingParams:
    def __init__(self, **kwargs):
        self.max_tokens = kwargs["max_tokens"]
        self.temperature = kwargs["temperature"]


class _FakeCuda:
    @staticmethod
    def is_available():
        return False


def test_vllm_adapter_none_decoder_extracts_valid_payload(monkeypatch):
    fake_vllm = ModuleType("vllm")
    fake_vllm.LLM = _FakeLLM
    fake_vllm.SamplingParams = _FakeSamplingParams
    fake_torch = ModuleType("torch")
    fake_torch.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    adapter = VLLMAdapter(
        model="fake/model",
        decoder="none",
        allowed_diagnoses={"Adenocarcinoma"},
    )

    extraction = adapter.extract("ignored prompt")

    assert adapter._llm.kwargs == {
        "model": "fake/model",
        "device": "cpu",
        "dtype": "float32",
        "enforce_eager": True,
        "tensor_parallel_size": 1,
        "trust_remote_code": True,
    }
    assert extraction.diagnosis_primary == "Adenocarcinoma"
    assert extraction.confidence == 0.9


def test_outlines_decoder_supports_vllm_backend(monkeypatch):
    calls = {}

    fake_outlines = ModuleType("outlines")
    fake_models = ModuleType("outlines.models")
    fake_generate = ModuleType("outlines.generate")
    fake_samplers = ModuleType("outlines.samplers")

    def fake_vllm(model_name, **kwargs):
        calls["model_name"] = model_name
        calls["model_kwargs"] = kwargs
        return "outlines-vllm-model"

    def fake_json(model, schema_object, sampler, whitespace_pattern=None):
        calls["json_model"] = model
        calls["schema"] = schema_object
        calls["whitespace_pattern"] = whitespace_pattern

        def _generator(prompt, max_tokens):
            calls["prompt"] = prompt
            calls["max_tokens"] = max_tokens
            return {
                "schema_version": "v1",
                "diagnosis_primary": "Adenocarcinoma",
                "diagnosis_secondary": [],
                "description": None,
                "interpretation_status": None,
                "specimen": None,
                "is_lymph_node": None,
                "anatomic_location": None,
                "container": None,
                "biomarkers": [],
                "confidence": None,
            }

        return _generator

    def fake_greedy():
        return "greedy"

    fake_models.vllm = fake_vllm
    fake_generate.json = fake_json
    fake_samplers.greedy = fake_greedy
    fake_outlines.models = fake_models
    fake_outlines.generate = fake_generate

    monkeypatch.setitem(sys.modules, "outlines", fake_outlines)
    monkeypatch.setitem(sys.modules, "outlines.models", fake_models)
    monkeypatch.setitem(sys.modules, "outlines.generate", fake_generate)
    monkeypatch.setitem(sys.modules, "outlines.samplers", fake_samplers)

    decoder = OutlinesDecoder(
        backend="vllm",
        allowed_diagnoses={"Adenocarcinoma"},
        runtime_available=True,
    )

    raw = decoder.generate(
        model={"model_name": "fake/model", "model_params": {"tensor_parallel_size": 1}},
        tokenizer=_FakeTokenizer(),
        prompt="ignored prompt",
        max_new_tokens=64,
        temperature=0.0,
    )

    assert '"diagnosis_primary": "Adenocarcinoma"' in raw
    assert calls["model_name"] == "fake/model"
    assert calls["model_kwargs"] == {"tensor_parallel_size": 1}
    assert calls["max_tokens"] == 64
    assert calls["whitespace_pattern"] == r""


def test_vllm_adapter_passes_cpu_model_params_to_outlines(monkeypatch):
    fake_torch = ModuleType("torch")
    fake_torch.cuda = _FakeCuda()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    adapter = VLLMAdapter(
        model="fake/model",
        decoder="outlines",
        allowed_diagnoses={"Adenocarcinoma"},
    )

    captured = {}

    def fake_generate(**kwargs):
        captured.update(kwargs)
        return '{"schema_version":"v1","diagnosis_primary":"Adenocarcinoma","diagnosis_secondary":[],"description":null,"interpretation_status":null,"specimen":null,"is_lymph_node":null,"anatomic_location":null,"container":null,"biomarkers":[],"confidence":null}'

    adapter._tokenizer = _FakeTokenizer()
    adapter._llm = object()
    adapter._decoder.validate_ready = lambda: None
    adapter._decoder.generate = fake_generate

    adapter.generate("ignored prompt")

    assert captured["model"] == {
        "model_name": "fake/model",
        "model_params": {
            "device": "cpu",
            "dtype": "float32",
            "enforce_eager": True,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
        },
    }

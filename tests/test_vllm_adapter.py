import pytest

from pathology_llm.inference.adapters.vllm import VLLMAdapter


def test_vllm_adapter_is_placeholder():
    adapter = VLLMAdapter(
        model="fake/model",
        decoder="none",
        allowed_diagnoses={"Adenocarcinoma"},
    )

    with pytest.raises(NotImplementedError, match="backend='vllm'"):
        adapter.generate("ignored prompt")

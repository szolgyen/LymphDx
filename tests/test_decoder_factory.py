import pytest

from pathology_llm.inference.decoders.factory import create_decoder


def test_create_decoder_returns_outlines_placeholder():
    decoder = create_decoder(backend="vllm", decoder_name="outlines")

    with pytest.raises(NotImplementedError, match="decoder='outlines'"):
        decoder.validate_ready()


def test_create_decoder_returns_sglang_placeholder():
    decoder = create_decoder(backend="sglang", decoder_name="sglang")

    with pytest.raises(NotImplementedError, match="decoder='sglang'"):
        decoder.validate_ready()

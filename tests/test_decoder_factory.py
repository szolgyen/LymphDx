import pytest

from inference.decoders.factory import create_decoder


def test_create_decoder_returns_outlines_not_implemented_for_vllm():
    decoder = create_decoder(
        backend="vllm",
        decoder_name="outlines",
        allowed_diagnoses={"Adenocarcinoma"},
    )

    decoder.validate_ready()


def test_create_decoder_returns_outlines_for_hf():
    decoder = create_decoder(
        backend="hf",
        decoder_name="outlines",
        allowed_diagnoses={"Adenocarcinoma"},
    )

    decoder.validate_ready()


def test_create_decoder_returns_sglang_placeholder():
    decoder = create_decoder(backend="sglang", decoder_name="sglang")

    with pytest.raises(NotImplementedError, match="decoder='sglang'"):
        decoder.validate_ready()

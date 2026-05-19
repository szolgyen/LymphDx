import logging

from pathology_llm.inference.adapters.base import BaseModelAdapter
from pathology_llm.inference.adapters.dummy import DummyAdapter
from pathology_llm.inference.adapters.hf import HFAdapter
from pathology_llm.inference.adapters.ollama import OllamaAdapter
from pathology_llm.inference.adapters.sglang import SGLangAdapter
from pathology_llm.inference.adapters.vllm import VLLMAdapter


logger = logging.getLogger(__name__)


SUPPORTED_BACKENDS = {"dummy", "hf", "vllm", "sglang", "ollama"}
SUPPORTED_DECODERS = {"auto", "none", "sglang", "guidance", "outlines"}


def _resolve_decoder(backend: str, decoder: str) -> str:
    decoder_name = decoder.strip().lower()
    if decoder_name not in SUPPORTED_DECODERS:
        raise ValueError(
            f"Unsupported decoder '{decoder}'. Allowed: {sorted(SUPPORTED_DECODERS)}"
        )

    if decoder_name == "auto":
        defaults = {
            "dummy": "none",
            "hf": "guidance",
            "vllm": "outlines",
            "sglang": "sglang",
            "ollama": "outlines",
        }
        return defaults[backend]

    if backend == "dummy" and decoder_name != "none":
        raise ValueError("dummy backend only supports decoder='none'")

    if backend == "sglang" and decoder_name not in {"sglang", "none"}:
        raise ValueError("sglang backend supports decoders: sglang, none")

    return decoder_name


def create_adapter(
    backend: str,
    model: str,
    decoder: str = "auto",
    allowed_diagnoses: set[str] | None = None,
) -> BaseModelAdapter:
    backend_name = backend.strip().lower()
    if backend_name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Allowed: {sorted(SUPPORTED_BACKENDS)}"
        )

    resolved_decoder = _resolve_decoder(backend_name, decoder)
    logger.info(
        "Creating adapter backend=%s model=%s decoder=%s",
        backend_name,
        model,
        resolved_decoder,
    )

    if backend_name == "dummy":
        return DummyAdapter(allowed_diagnoses=allowed_diagnoses)

    if backend_name == "hf":
        return HFAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
        )

    if backend_name == "vllm":
        return VLLMAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
        )

    if backend_name == "sglang":
        return SGLangAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
        )

    if backend_name == "ollama":
        return OllamaAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
        )

    # Defensive fallback; should be unreachable due to SUPPORTED_BACKENDS validation.
    raise ValueError(f"Unhandled backend '{backend_name}'")

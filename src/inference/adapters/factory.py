import logging

from pydantic import BaseModel

from inference.adapters.base import BaseModelAdapter

from inference.adapters.hf import HFAdapter
from inference.adapters.ollama import OllamaAdapter
from inference.adapters.sglang import SGLangAdapter
from inference.adapters.vllm import VLLMAdapter


logger = logging.getLogger(__name__)


SUPPORTED_BACKENDS = {"hf", "vllm", "sglang", "ollama"}
SUPPORTED_DECODERS = {"none", "guidance", "outlines", "xgrammar"}


def _resolve_decoder(backend: str, decoder: str) -> str:
    decoder_name = decoder.strip().lower()
    if decoder_name not in SUPPORTED_DECODERS:
        raise ValueError(
            f"Unsupported decoder '{decoder}'. Allowed: {sorted(SUPPORTED_DECODERS)}"
        )

    return decoder_name


def create_adapter(
    backend: str,
    model: str,
    decoder: str = "none",
    allowed_diagnoses: set[str] | None = None,
    schema_model: type[BaseModel] | None = None,
    server_base_url: str | None = None,
    max_new_tokens: int = 4096,
) -> BaseModelAdapter:
    backend_name = backend.strip().lower()
    if backend_name not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend '{backend}'. Allowed: {sorted(SUPPORTED_BACKENDS)}"
        )

    resolved_decoder = _resolve_decoder(backend_name, decoder)
    logger.info(
        "Creating adapter backend=%s model=%s decoder=%s max_new_tokens=%s",
        backend_name,
        model,
        resolved_decoder,
        max_new_tokens,
    )

    if backend_name == "hf":
        return HFAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
            schema_model=schema_model,
            max_new_tokens=max_new_tokens,
        )

    if backend_name == "vllm":
        vllm_kwargs = {"base_url": server_base_url} if server_base_url else {}
        return VLLMAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
            schema_model=schema_model,
            max_new_tokens=max_new_tokens,
            **vllm_kwargs,
        )

    if backend_name == "sglang":
        return SGLangAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
            schema_model=schema_model,
        )

    if backend_name == "ollama":
        return OllamaAdapter(
            model=model,
            decoder=resolved_decoder,
            allowed_diagnoses=allowed_diagnoses,
            schema_model=schema_model,
        )

    # Defensive fallback; should be unreachable due to SUPPORTED_BACKENDS validation.
    raise ValueError(f"Unhandled backend '{backend_name}'")

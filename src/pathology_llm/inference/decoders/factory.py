import logging
from typing import Any, Callable

from pathology_llm.inference.decoders.base import BaseDecoder
from pathology_llm.inference.decoders.hf_guidance import HFGuidanceDecoder
from pathology_llm.inference.decoders.none import NoneDecoder
from pathology_llm.inference.decoders.outlines import OutlinesDecoder
from pathology_llm.inference.decoders.sglang import SGLangDecoder


def create_decoder(
    backend: str,
    decoder_name: str,
    allowed_diagnoses: set[str] | None = None,
    prompt_formatter: Callable[[str, Any], str] | None = None,
    logger: logging.Logger | None = None,
) -> BaseDecoder:
    backend_name = backend.strip().lower()
    resolved_decoder = decoder_name.strip().lower()

    if resolved_decoder == "none":
        return NoneDecoder(prompt_formatter=prompt_formatter)

    if resolved_decoder == "guidance":
        if backend_name != "hf":
            raise ValueError(
                f"Unsupported decoder '{decoder_name}' for backend='{backend_name}'"
            )
        return HFGuidanceDecoder(
            allowed_diagnoses=allowed_diagnoses,
            prompt_formatter=prompt_formatter,
            logger=logger,
        )

    if resolved_decoder == "outlines":
        return OutlinesDecoder(
            backend=backend_name,
            prompt_formatter=prompt_formatter,
        )

    if resolved_decoder == "sglang":
        return SGLangDecoder(
            backend=backend_name,
            prompt_formatter=prompt_formatter,
        )

    raise ValueError(
        f"Unsupported decoder '{decoder_name}' for backend='{backend_name}'"
    )

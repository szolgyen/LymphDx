"""vLLM-specific decoder leveraging native constraint integration.

This decoder is designed exclusively for the vLLM backend. It uses vLLM's
native integration with constraint backends (xgrammar)
to enforce JSON schema constraints at the logit level during generation.

Requires vLLM server to be configured with a constraint backend:
  python -m vllm.entrypoints.openai.api_server \\
    --model google/medgemma-1.5-4b-it \\
    --guided-decoding-backend xgrammar
"""

import logging
from typing import Any, Callable

from pydantic import BaseModel

from inference.decoders.base import BaseDecoder
from inference.decoders.diagnosis_constraints import DiagnosisConstraintsMixin


class VLLMxgrammarDecoder(DiagnosisConstraintsMixin, BaseDecoder):
    """vLLM-native constraint decoder using response_format."""

    def __init__(
        self,
        allowed_diagnoses: set[str] | None = None,
        prompt_formatter: Callable[[str, Any], str] | None = None,
        logger: logging.Logger | None = None,
        schema_model: type[BaseModel] | None = None,
    ):
        super().__init__()
        self._allowed_diagnoses = allowed_diagnoses
        self._prompt_formatter = prompt_formatter
        self._logger = logger
        self._schema_model = schema_model

    def validate_ready(self) -> None:
        """Validate that schema is available for constraint enforcement."""
        if self._schema_model is None:
            raise ValueError(
                "decoder='vllm_xgrammar' requires a schema_model to be provided"
            )
        if self._allowed_diagnoses is None and self._logger:
            self._logger.warning(
                "decoder='vllm_xgrammar' with no diagnosis constraints - "
                "all diagnoses will be permitted"
            )

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        """Prepare prompt with optional diagnosis constraints suffix."""
        self.validate_ready()

        guarded_prompt = prompt + self.build_diagnosis_constraints_prompt_suffix()
        if self._prompt_formatter is None:
            return guarded_prompt
        return self._prompt_formatter(guarded_prompt, tokenizer)

    def _build_constrained_schema(self) -> dict[str, Any]:
        """Build schema dict with diagnosis constraints applied."""
        if self._schema_model is None:
            raise ValueError("schema_model is required to build constrained schema")

        # Get the JSON schema and apply diagnosis constraints
        base_schema_dict = self._schema_model.model_json_schema()
        constrained_schema_dict = self.apply_diagnosis_constraints_to_schema(base_schema_dict)

        # Return the constrained schema dict (ready to send to response_format)
        return constrained_schema_dict

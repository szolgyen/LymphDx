import json
import logging
from typing import Any, Callable

from pathology_llm.inference.decoders.base import BaseDecoder


class StrictJsonDecoder(BaseDecoder):
    """Shared strict JSON decoding helpers for backend-specific decoders."""

    def __init__(
        self,
        allowed_diagnoses: set[str] | None = None,
        prompt_formatter: Callable[[str, Any], str] | None = None,
        logger: logging.Logger | None = None,
    ):
        self._allowed_diagnoses = allowed_diagnoses
        self._prompt_formatter = prompt_formatter
        self._logger = logger

    def validate_ready(self) -> None:
        if not self._allowed_diagnoses:
            raise ValueError(
                f"decoder='{self.name}' requires non-empty allowed_diagnoses for strict constraints"
            )

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        self.validate_ready()

        diagnosis_terms = "\n".join(
            f"- {term}" for term in sorted(self._allowed_diagnoses or set())
        )
        strict_suffix = (
            f"\n\nSTRICT DECODER MODE ({self.name}):\n"
            "- You MUST output exactly one valid JSON object matching the requested schema.\n"
            "- diagnosis_primary must be selected from the allowed list below.\n"
            "- diagnosis_secondary terms must all be selected from the allowed list below.\n"
            "- Do not output any explanation or markdown.\n"
            "ALLOWED DIAGNOSES:\n"
            f"{diagnosis_terms}\n"
        )
        guarded_prompt = prompt + strict_suffix

        if self._logger is not None:
            self._logger.info(
                "Using strict %s decoder constraints with %d allowed diagnoses",
                self.name,
                len(self._allowed_diagnoses or set()),
            )

        if self._prompt_formatter is None:
            return guarded_prompt
        return self._prompt_formatter(guarded_prompt, tokenizer)

    def _build_schema(self) -> dict[str, Any]:
        allowed = sorted(self._allowed_diagnoses or set())
        if not allowed:
            raise ValueError(
                f"decoder='{self.name}' requires non-empty allowed_diagnoses for strict constraints"
            )

        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string", "enum": ["v1"]},
                "diagnosis_primary": {
                    "anyOf": [
                        {"type": "string", "enum": allowed},
                        {"type": "null"},
                    ]
                },
                "diagnosis_secondary": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string", "enum": allowed},
                },
                "description": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "interpretation_status": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "specimen": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "is_lymph_node": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "anatomic_location": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "container": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "biomarkers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "name": {"type": "string"},
                            "value": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "null"},
                                ]
                            },
                        },
                        "required": ["name", "value"],
                    },
                },
                "confidence": {
                    "anyOf": [
                        {"type": "number", "minimum": 0.0, "maximum": 1.0},
                        {"type": "null"},
                    ]
                },
            },
            "required": [
                "schema_version",
                "diagnosis_primary",
                "diagnosis_secondary",
                "description",
                "interpretation_status",
                "specimen",
                "is_lymph_node",
                "anatomic_location",
                "container",
                "biomarkers",
                "confidence",
            ],
        }

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

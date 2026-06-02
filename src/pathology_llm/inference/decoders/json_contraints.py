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
        return None

    @staticmethod
    def _build_json_output_guard() -> str:
        return (
            "\n\nSTRICT JSON MODE:\n"
            "- You MUST output exactly one valid JSON object matching the requested schema.\n"
            "- Do not output any explanation or markdown.\n"
        )

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        self.validate_ready()
        guarded_prompt = prompt + self._build_json_output_guard()

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
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "schema_version": {"type": "string", "enum": ["v2"]},
                "primary_diagnosis": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                },
                "differential_diagnoses": {
                    "type": "array",
                    "maxItems": 10,
                    "items": {"type": "string"},
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
                "is_definitive": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "has_differential_diagnosis": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "has_prior_malignancy": {
                    "anyOf": [
                        {"type": "boolean"},
                        {"type": "null"},
                    ]
                },
                "has_concurrent_malignancy": {
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
                "primary_diagnosis",
                "has_differential_diagnosis",
                "differential_diagnoses",
                "specimen",
                "is_lymph_node",
                "is_definitive",
                "has_prior_malignancy",
                "has_concurrent_malignancy",
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

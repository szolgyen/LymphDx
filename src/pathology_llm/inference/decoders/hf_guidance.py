import importlib.util
import json
import logging
import time
from typing import Any, Callable

from pathology_llm.inference.decoders.base import BaseDecoder


class HFGuidanceDecoder(BaseDecoder):
    """HF guidance decoder placeholder with explicit strategy boundary."""

    name = "guidance"

    def __init__(
        self,
        allowed_diagnoses: set[str] | None = None,
        prompt_formatter: Callable[[str, Any], str] | None = None,
        logger: logging.Logger | None = None,
        runtime_available: bool | None = None,
    ):
        self._allowed_diagnoses = allowed_diagnoses
        self._prompt_formatter = prompt_formatter
        self._logger = logger
        if runtime_available is None:
            self._runtime_available = importlib.util.find_spec("guidance") is not None
        else:
            self._runtime_available = runtime_available

    def validate_ready(self) -> None:
        if not self._runtime_available:
            raise RuntimeError(
                "decoder='guidance' requires the 'guidance' package, but it is not installed"
            )
        if not self._allowed_diagnoses:
            raise ValueError(
                "decoder='guidance' requires non-empty allowed_diagnoses for strict constraints"
            )

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        self.validate_ready()

        diagnosis_terms = "\n".join(
            f"- {term}" for term in sorted(self._allowed_diagnoses or set())
        )
        strict_suffix = (
            "\n\nSTRICT DECODER MODE (guidance):\n"
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
                "Using strict guidance decoder constraints with %d allowed diagnoses",
                len(self._allowed_diagnoses or set()),
            )

        if self._prompt_formatter is None:
            return guarded_prompt
        return self._prompt_formatter(guarded_prompt, tokenizer)

    def get_generation_kwargs(self, tokenizer: Any) -> dict[str, Any]:
        # Strict mode uses deterministic decoding and no sampling.
        return {
            "do_sample": False,
            "temperature": None,
        }

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str | None:
        self.validate_ready()

        import guidance
        import guidance.chat as guidance_chat

        prompt_for_model = self.prepare_prompt(prompt, tokenizer)
        schema = self._build_schema()

        if self._logger is not None:
            self._logger.info("Guidance decoding started")
        started = time.perf_counter()

        # Use explicit ChatML template to avoid noisy fallback warning logs.
        lm = guidance.models.Transformers(
            model=model,
            tokenizer=tokenizer,
            chat_template=guidance_chat.ChatMLTemplate,
            enable_monitoring=False,
        )
        guided_program = (
            prompt_for_model
            + "\n\nReturn exactly one JSON object that satisfies the schema."
            + guidance.json(
                name="extraction",
                schema=schema,
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
        )
        response = lm + guided_program
        payload = response["extraction"]

        if self._logger is not None:
            self._logger.info(
                "Guidance decoding finished in %.2fs",
                time.perf_counter() - started,
            )

        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    def _build_schema(self) -> dict[str, Any]:
        allowed = sorted(self._allowed_diagnoses or set())
        if not allowed:
            raise ValueError(
                "decoder='guidance' requires non-empty allowed_diagnoses for strict constraints"
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

import json
import logging
from typing import Any, Callable

from pydantic import BaseModel

from inference.decoders.base import BaseDecoder
from schemas.pathology import PathologyExtractionV2


class StrictJsonDecoder(BaseDecoder):
    """Shared strict JSON decoding helpers for backend-specific decoders."""

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
        self._schema_model = schema_model or PathologyExtractionV2

    def validate_ready(self) -> None:
        return None

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        self.validate_ready()
        guarded_prompt = prompt

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
        schema = self._schema_model.model_json_schema()
        self._remove_decoder_excluded_fields(schema)
        self._apply_strict_object_rules(schema)
        return schema

    @classmethod
    def _remove_decoder_excluded_fields(cls, node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            required = node.get("required")
            if isinstance(properties, dict):
                excluded_keys = [
                    key
                    for key, value in properties.items()
                    if isinstance(value, dict)
                    and value.get("x-reportllm-decoder-exclude") is True
                ]
                for key in excluded_keys:
                    properties.pop(key, None)
                if isinstance(required, list):
                    node["required"] = [
                        key for key in required if key not in excluded_keys
                    ]

            for value in node.values():
                cls._remove_decoder_excluded_fields(value)
            return

        if isinstance(node, list):
            for item in node:
                cls._remove_decoder_excluded_fields(item)

    @classmethod
    def _apply_strict_object_rules(cls, node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
                node["additionalProperties"] = False

            for value in node.values():
                cls._apply_strict_object_rules(value)
            return

        if isinstance(node, list):
            for item in node:
                cls._apply_strict_object_rules(item)

    @staticmethod
    def _json_dumps(payload: Any) -> str:
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

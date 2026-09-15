"""vLLM backend adapter.

Talks to a running vLLM OpenAI-compatible server rather than embedding the engine in
this process. That keeps model loading out of the pipeline run, lets several clients
share one GPU-resident model, and gives automatic prefix caching across calls.

Note on constrained decoding: vLLM 0.27 silently ignores the legacy ``guided_choice``
field, and reasoning models (e.g. medgemma-1.5) lose accuracy when their first token is
grammar-constrained. Schema enforcement is therefore applied via ``response_format`` only
when a decoder other than ``none`` is requested, and validation always runs afterwards.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from pydantic import BaseModel

from inference.adapters.base import BaseModelAdapter
from schemas.validation import SchemaValidationError, validate_output

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:8000/v1"

THINK_CLOSE = "<unused95>"


class VLLMAdapter(BaseModelAdapter):
    """vLLM backend adapter backed by a local OpenAI-compatible server."""

    backend_name = "vllm"

    def __init__(
        self,
        model: str,
        decoder: str = "none",
        allowed_diagnoses: Optional[set[str]] = None,
        schema_model: Optional[type[BaseModel]] = None,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "EMPTY",
        temperature: float = 0.0,
        max_new_tokens: int = 2048,
        timeout: float = 300.0,
    ):
        super().__init__(
            allowed_diagnoses=allowed_diagnoses,
            schema_model=schema_model,
        )

        self.model = model
        self.decoder_name = decoder.strip().lower()
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI

            logger.info("Connecting to vLLM server at %s", self.base_url)
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )

        return self._client

    def _build_extra_body(self) -> dict[str, Any]:
        if self.decoder_name == "none":
            return {}

        return {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": self.schema_model.__name__,
                    "schema": self.schema_model.model_json_schema(),
                },
            }
        }

    def generate(self, prompt: str) -> str:
        client = self._ensure_client()

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
            extra_body=self._build_extra_body(),
        )

        if response.choices[0].finish_reason == "length":
            logger.warning("vLLM output hit the token limit and may be truncated")

        return response.choices[0].message.content or ""

    def parse(self, raw: str) -> BaseModel:
        json_payload = self._extract_json_payload(raw)
        allowed_diagnoses = (
            None if self.decoder_name == "none" else self.allowed_diagnoses
        )

        return validate_output(
            json_payload,
            schema_model=self.schema_model,
            allowed_diagnoses=allowed_diagnoses,
        )

    @staticmethod
    def _extract_json_payload(raw: str) -> str:
        candidate = raw.strip()

        # Reasoning models emit a thought block before the answer.
        if THINK_CLOSE in candidate:
            candidate = candidate.rsplit(THINK_CLOSE, 1)[1].strip()

        if candidate.startswith("```"):
            lines = candidate.splitlines()

            if lines and lines[0].startswith("```"):
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            candidate = "\n".join(lines).strip()

        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")

        if start != -1 and end != -1 and end > start:
            nested = candidate[start : end + 1]

            try:
                json.loads(nested)
                return nested
            except json.JSONDecodeError:
                pass

        preview = candidate[:1024].replace("\n", "\\n")

        raise SchemaValidationError(
            "vLLM output is not valid JSON and no JSON object could be extracted. "
            f"Output preview: {preview!r}"
        )

import importlib.util
import json
import logging
import time
from typing import Any, Callable

from pathology_llm.inference.decoders.strict_json import StrictJsonDecoder


class OutlinesDecoder(StrictJsonDecoder):
    """Outlines decoder with HF support and placeholder behavior elsewhere."""

    name = "outlines"

    def __init__(
        self,
        backend: str,
        allowed_diagnoses: set[str] | None = None,
        prompt_formatter: Callable[[str, Any], str] | None = None,
        logger: logging.Logger | None = None,
        runtime_available: bool | None = None,
    ):
        self.backend = backend
        self._allowed_diagnoses = allowed_diagnoses
        self._prompt_formatter = prompt_formatter
        self._logger = logger
        if runtime_available is None:
            self._runtime_available = importlib.util.find_spec("outlines") is not None
        else:
            self._runtime_available = runtime_available
        self._generator = None
        self._generator_key: tuple[int, int] | None = None
        self._generator_backend: str | None = None

    def validate_ready(self) -> None:
        if self.backend != "hf":
            raise NotImplementedError(
                "Decoder integration is not implemented yet for "
                f"decoder='{self.name}', backend='{self.backend}'."
            )
        if not self._runtime_available:
            raise RuntimeError(
                "decoder='outlines' requires the 'outlines' package, but it is not installed"
            )
        super().validate_ready()

    def get_generation_kwargs(self, tokenizer: Any) -> dict[str, Any]:
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

        from outlines.generator import Generator, JsonSchema
        from outlines.models import from_transformers

        prompt_for_model = self.prepare_prompt(prompt, tokenizer)
        schema = self._build_schema()

        generator_key = (id(model), id(tokenizer))
        if self._generator is None or self._generator_key != generator_key:
            outlines_model = from_transformers(model, tokenizer)
            self._generator, self._generator_backend = self._build_generator(
                outlines_model=outlines_model,
                schema=schema,
                generator_cls=Generator,
                json_schema_cls=JsonSchema,
            )
            self._generator_key = generator_key

        if self._logger is not None:
            self._logger.info(
                "Outlines decoding started backend=%s",
                self._generator_backend,
            )
        started = time.perf_counter()

        payload = self._generator(
            prompt_for_model,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

        if self._logger is not None:
            self._logger.info(
                "Outlines decoding finished in %.2fs backend=%s",
                time.perf_counter() - started,
                self._generator_backend,
            )

        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    def _build_generator(
        self,
        outlines_model: Any,
        schema: dict[str, Any],
        generator_cls: Any,
        json_schema_cls: Any,
    ) -> tuple[Any, str]:
        backend_errors: list[str] = []
        for backend in ("llguidance", "xgrammar", "outlines_core"):
            try:
                generator = generator_cls(
                    outlines_model,
                    output_type=json_schema_cls(schema, whitespace_pattern=r""),
                    backend=backend,
                )
                if self._logger is not None:
                    self._logger.info("Using Outlines backend=%s", backend)
                return generator, backend
            except Exception as exc:
                backend_errors.append(f"{backend}: {exc}")

        details = "; ".join(backend_errors)
        raise RuntimeError(
            "decoder='outlines' could not initialize any supported backend. "
            f"Tried: {details}"
        )

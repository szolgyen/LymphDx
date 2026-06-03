import importlib.util
import json
import logging
import time
from typing import Any, Callable

from pydantic import BaseModel

from pathology_llm.inference.decoders.diagnosis_constraints import (
    DiagnosisConstraintsMixin,
)
from pathology_llm.inference.decoders.json_contraints import StrictJsonDecoder


class OutlinesDecoder(DiagnosisConstraintsMixin, StrictJsonDecoder):
    """Outlines decoder with HF and vLLM support."""

    name = "outlines"

    def __init__(
        self,
        backend: str,
        allowed_diagnoses: set[str] | None = None,
        prompt_formatter: Callable[[str, Any], str] | None = None,
        logger: logging.Logger | None = None,
        runtime_available: bool | None = None,
        schema_model: type[BaseModel] | None = None,
    ):
        self.backend = backend
        super().__init__(
            allowed_diagnoses=allowed_diagnoses,
            prompt_formatter=prompt_formatter,
            logger=logger,
            schema_model=schema_model,
        )
        if runtime_available is None:
            self._runtime_available = importlib.util.find_spec("outlines") is not None
        else:
            self._runtime_available = runtime_available
        self._generator = None
        self._generator_key: tuple[int, int] | None = None
        self._generator_backend: str | None = None

    def validate_ready(self) -> None:
        if self.backend not in {"hf", "vllm"}:
            raise NotImplementedError(
                "Decoder integration is not implemented yet for "
                f"decoder='{self.name}', backend='{self.backend}'."
            )
        if not self._runtime_available:
            raise RuntimeError(
                "decoder='outlines' requires the 'outlines' package, but it is not installed"
            )
        super().validate_ready()
        self.validate_diagnosis_constraints()

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        self.validate_ready()

        guarded_prompt = (
            prompt
            + self._build_json_output_guard()
            + self.build_diagnosis_constraints_prompt_suffix()
        )
        if self._logger is not None:
            self._logger.info(
                "Using strict %s decoder constraints with %d allowed diagnoses",
                self.name,
                len(self._allowed_diagnoses or set()),
            )

        if self._prompt_formatter is None:
            return guarded_prompt
        return self._prompt_formatter(guarded_prompt, tokenizer)

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

        if self.backend == "vllm":
            return self._generate_with_vllm(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
            )

        from outlines.generator import Generator, JsonSchema
        from outlines.models import from_transformers

        prompt_for_model = self.prepare_prompt(prompt, tokenizer)
        schema = self.apply_diagnosis_constraints_to_schema(self._build_schema())

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

    def _generate_with_vllm(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
    ) -> str:
        import outlines.generate
        import outlines.models
        from outlines.samplers import greedy

        prompt_for_model = self.prepare_prompt(prompt, tokenizer)
        schema = self.apply_diagnosis_constraints_to_schema(self._build_schema())
        model_name, model_params = self._resolve_vllm_model(model)
        generator_key = (
            self.backend,
            model_name,
            tuple(sorted(model_params.items())),
        )
        if self._generator is None or self._generator_key != generator_key:
            outlines_model = outlines.models.vllm(model_name, **model_params)
            self._generator = outlines.generate.json(
                outlines_model,
                schema,
                sampler=greedy(),
                whitespace_pattern=r"",
            )
            self._generator_backend = "outlines_vllm"
            self._generator_key = generator_key

        if self._logger is not None:
            self._logger.info("Outlines decoding started backend=%s", self.backend)
        started = time.perf_counter()
        payload = self._generator(prompt_for_model, max_tokens=max_new_tokens)
        if self._logger is not None:
            self._logger.info(
                "Outlines decoding finished in %.2fs backend=%s",
                time.perf_counter() - started,
                self.backend,
            )

        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    @staticmethod
    def _resolve_vllm_model(model: Any) -> tuple[str, dict[str, Any]]:
        if not isinstance(model, dict):
            raise TypeError(
                "Outlines vLLM decoder expects model metadata as a dict with model_name"
            )

        model_name = model.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("Outlines vLLM decoder requires a non-empty model_name")

        model_params = model.get("model_params") or {}
        if not isinstance(model_params, dict):
            raise TypeError("Outlines vLLM decoder model_params must be a dict")

        return model_name, model_params

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

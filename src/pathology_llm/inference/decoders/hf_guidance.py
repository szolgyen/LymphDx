import importlib.util
import logging
import time
from typing import Any, Callable

from pathology_llm.inference.decoders.diagnosis_constraints import (
    DiagnosisConstraintsMixin,
)
from pathology_llm.inference.decoders.json_contraints import StrictJsonDecoder


class HFGuidanceDecoder(DiagnosisConstraintsMixin, StrictJsonDecoder):
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
        schema = self.apply_diagnosis_constraints_to_schema(self._build_schema())

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

        return self._json_dumps(payload)

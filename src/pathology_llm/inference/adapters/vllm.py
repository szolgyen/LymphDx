import json
import logging
from typing import Any

from pathology_llm.inference.adapters.base import BaseModelAdapter
from pathology_llm.inference.decoders.base import BaseDecoder
from pathology_llm.inference.decoders.factory import create_decoder
from pathology_llm.schemas.validation import (
    SchemaValidationError,
    validate_pathology_output,
)


logger = logging.getLogger(__name__)


class VLLMAdapter(BaseModelAdapter):
    """vLLM backend adapter."""

    backend_name = "vllm"
    supported_decoders = {"none", "outlines"}

    def __init__(
        self,
        model: str,
        decoder: str,
        allowed_diagnoses: set[str] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        super().__init__(allowed_diagnoses=allowed_diagnoses)
        decoder_name = decoder.strip().lower()
        if decoder_name not in self.supported_decoders:
            raise ValueError(
                f"vLLM backend does not support decoder='{decoder_name}'. "
                f"Supported: {sorted(self.supported_decoders)}"
            )

        self.model_name = model
        self.decoder_name = decoder_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.trust_remote_code = trust_remote_code
        self.dtype = dtype
        self._runtime_device = self._detect_runtime_device()
        self._model_params = self._build_model_params()
        self._decoder: BaseDecoder = create_decoder(
            backend=self.backend_name,
            decoder_name=decoder_name,
            allowed_diagnoses=allowed_diagnoses,
            prompt_formatter=self._format_prompt_for_json,
            logger=logger,
        )
        self._llm = None
        self._tokenizer = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return

        try:
            from vllm import LLM
        except ImportError as exc:
            raise RuntimeError(
                "vLLM backend requires the 'vllm' package. "
                "Install optional deps, e.g. pip install -r requirements/vllm/requirements.txt"
            ) from exc

        logger.info(
            "Loading vLLM model=%s decoder=%s device=%s tensor_parallel_size=%s",
            self.model_name,
            self.decoder_name,
            self._runtime_device,
            self.tensor_parallel_size,
        )
        self._llm = LLM(model=self.model_name, **self._model_params)
        get_tokenizer = getattr(self._llm, "get_tokenizer", None)
        if callable(get_tokenizer):
            self._tokenizer = get_tokenizer()

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        self._decoder.validate_ready()

        if self.decoder_name == "outlines":
            return (
                self._decoder.generate(
                    model={
                        "model_name": self.model_name,
                        "model_params": self._model_params,
                    },
                    tokenizer=self._tokenizer,
                    prompt=prompt,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                )
                or ""
            )

        from vllm import SamplingParams

        prompt_for_model = self._decoder.prepare_prompt(prompt, self._tokenizer)
        sampling_params = SamplingParams(
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        outputs = self._llm.generate([prompt_for_model], sampling_params)
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM generation returned no outputs")
        return outputs[0].outputs[0].text.strip()

    @staticmethod
    def _detect_runtime_device() -> str:
        try:
            import torch
        except ImportError:
            return "cpu"

        cuda = getattr(torch, "cuda", None)
        if cuda is None:
            return "cpu"

        is_available = getattr(cuda, "is_available", None)
        if not callable(is_available):
            return "cpu"

        try:
            return "cuda" if is_available() else "cpu"
        except Exception:
            return "cpu"

    def _build_model_params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "trust_remote_code": self.trust_remote_code,
        }

        if self._runtime_device == "cpu":
            params.update(
                {
                    "device": "cpu",
                    "dtype": "float32" if self.dtype == "auto" else self.dtype,
                    "enforce_eager": True,
                    "tensor_parallel_size": 1,
                }
            )
            return params

        params.update(
            {
                "tensor_parallel_size": self.tensor_parallel_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "dtype": self.dtype,
            }
        )
        return params

    @staticmethod
    def _format_prompt_for_json(prompt: str, tokenizer: Any) -> str:
        json_guard = (
            "\n\nCRITICAL OUTPUT FORMAT:\n"
            "- Return EXACTLY one JSON object.\n"
            "- Do not include markdown, code fences, commentary, or trailing text.\n"
            "- Start with '{' and end with '}'.\n"
        )
        guarded_prompt = prompt + json_guard

        apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
        if callable(apply_chat_template):
            try:
                return apply_chat_template(
                    [{"role": "user", "content": guarded_prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                return guarded_prompt

        return guarded_prompt

    def parse(self, raw: str):
        json_payload = self._extract_json_payload(raw)
        return validate_pathology_output(
            json_payload,
            allowed_diagnoses=self.allowed_diagnoses,
        )

    @staticmethod
    def _extract_json_payload(raw: str) -> str | dict[str, Any]:
        candidate = raw.strip()

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
            nested_candidate = candidate[start : end + 1]
            try:
                json.loads(nested_candidate)
                return nested_candidate
            except json.JSONDecodeError:
                pass

        preview = candidate[:400].replace("\n", "\\n")
        raise SchemaValidationError(
            "vLLM output is not valid JSON and no JSON object could be extracted. "
            f"Output preview: {preview!r}"
        )

import json
import logging
from typing import Any

from pydantic import BaseModel

from inference.decoders.base import BaseDecoder
from inference.decoders.factory import create_decoder
from inference.adapters.base import BaseModelAdapter
from schemas.validation import (
    SchemaValidationError,
    validate_output,
)


logger = logging.getLogger(__name__)


class HFAdapter(BaseModelAdapter):
    """HuggingFace Transformers backend adapter."""

    backend_name = "hf"
    supported_decoders = {"none", "guidance", "outlines"}

    def __init__(
        self,
        model: str,
        decoder: str,
        allowed_diagnoses: set[str] | None = None,
        schema_model: type[BaseModel] | None = None,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        device_map: str = "auto",
    ):
        super().__init__(allowed_diagnoses=allowed_diagnoses, schema_model=schema_model)
        decoder_name = decoder.strip().lower()
        if decoder_name not in self.supported_decoders:
            raise ValueError(
                f"HF backend does not support decoder='{decoder_name}'. "
                f"Supported: {sorted(self.supported_decoders)}"
            )

        self.model_name = model
        self.decoder_name = decoder_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device_map = device_map
        self._decoder: BaseDecoder = create_decoder(
            backend=self.backend_name,
            decoder_name=decoder_name,
            allowed_diagnoses=allowed_diagnoses,
            prompt_formatter=self._format_prompt_for_json,
            logger=logger,
            schema_model=schema_model,
        )
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HF backend requires transformers/torch. "
                "Install optional deps, e.g. pip install -r requirements/hf/requirements.txt"
            ) from exc

        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        logger.info(
            "Loading HF model=%s decoder=%s device_map=%s",
            self.model_name,
            self.decoder_name,
            self.device_map,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            dtype=torch_dtype,
        )

        if (
            self._tokenizer.pad_token_id is None
            and self._tokenizer.eos_token_id is not None
        ):
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

    def generate(self, prompt: str) -> str:
        self._ensure_loaded()
        self._decoder.validate_ready()

        import torch

        tokenizer = self._tokenizer
        model = self._model

        decoder_output = self._decoder.generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )
        if decoder_output is not None:
            # Decoder handled generation; capture decorated prompt if available
            self._last_decorated_prompt = self._decoder.get_last_decorated_prompt()
            return decoder_output

        prompt_for_model = self._decoder.prepare_prompt(prompt, tokenizer)
        self._last_decorated_prompt = prompt_for_model
        encoded = tokenizer(prompt_for_model, return_tensors="pt")

        # With non-sharded models, move inputs to model device.
        model_device = getattr(model, "device", None)
        if model_device is not None and str(model_device) != "meta":
            encoded = {key: value.to(model_device) for key, value in encoded.items()}

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
            "temperature": self.temperature if self.temperature > 0 else None,
            "pad_token_id": tokenizer.pad_token_id,
        }
        generate_kwargs.update(self._decoder.get_generation_kwargs(tokenizer))
        generate_kwargs = {
            key: value for key, value in generate_kwargs.items() if value is not None
        }

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generate_kwargs)

        input_len = encoded["input_ids"].shape[1]
        generated_ids = output_ids[0][input_len:]
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    @staticmethod
    def _format_prompt_for_json(prompt: str, tokenizer: Any) -> str:
        """Encourage schema-shaped JSON-only outputs for base and chat models."""
        json_guard = (
            "\n\n\nCRITICAL OUTPUT FORMAT:\n"
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
                # Fall back to plain prompt if tokenizer chat template is unavailable.
                return guarded_prompt

        return guarded_prompt

    def parse(self, raw: str):
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
    def _extract_json_payload(raw: str) -> str | dict[str, Any]:
        candidate = raw.strip()

        # Handle markdown fenced responses.
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
            "HF output is not valid JSON and no JSON object could be extracted. "
            f"Output preview: {preview!r}"
        )

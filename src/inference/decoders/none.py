import importlib.util
from typing import Any, Callable

from pydantic import BaseModel

from inference.decoders.json_contraints import StrictJsonDecoder


class NoneDecoder(StrictJsonDecoder):
    """JSON-structured decoding without diagnosis-term constraints."""

    name = "none"

    def __init__(
        self,
        backend: str,
        prompt_formatter: Callable[[str, Any], str] | None = None,
        logger: Any = None,
        runtime_available: bool | None = None,
        schema_model: type[BaseModel] | None = None,
    ):
        super().__init__(
            allowed_diagnoses=None,
            prompt_formatter=prompt_formatter,
            logger=logger,
            schema_model=schema_model,
        )
        self.backend = backend
        if runtime_available is None:
            self._runtime_available = importlib.util.find_spec("guidance") is not None
        else:
            self._runtime_available = runtime_available

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str | None:
        # JSON-constrained generation is only available through Guidance on HF.
        if self.backend != "hf" or not self._runtime_available:
            return None

        import guidance
        import guidance.chat as guidance_chat

        prompt_for_model = self.prepare_prompt(prompt, tokenizer)
        schema = self._build_schema()

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

        # Store the decorated prompt without the schema JSON
        self._last_decorated_prompt = prompt_for_model

        return self._json_dumps(payload)

from typing import Any, Callable

from pathology_llm.inference.decoders.base import BaseDecoder


class PlaceholderDecoder(BaseDecoder):
    """Shared placeholder behavior for decoders not implemented yet."""

    name = "placeholder"

    def __init__(
        self,
        backend: str,
        prompt_formatter: Callable[[str, Any], str] | None = None,
    ):
        self.backend = backend
        self._prompt_formatter = prompt_formatter

    def validate_ready(self) -> None:
        raise NotImplementedError(
            "Decoder integration is not implemented yet for "
            f"decoder='{self.name}', backend='{self.backend}'."
        )

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        if self._prompt_formatter is None:
            return prompt
        return self._prompt_formatter(prompt, tokenizer)

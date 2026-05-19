from typing import Any, Callable

from pathology_llm.inference.decoders.base import BaseDecoder


class NoneDecoder(BaseDecoder):
    """Best-effort decoder without hard generation constraints."""

    name = "none"

    def __init__(
        self,
        prompt_formatter: Callable[[str, Any], str] | None = None,
    ):
        self._prompt_formatter = prompt_formatter

    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        if self._prompt_formatter is None:
            return prompt
        return self._prompt_formatter(prompt, tokenizer)

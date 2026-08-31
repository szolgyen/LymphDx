from abc import ABC, abstractmethod
from typing import Any


class BaseDecoder(ABC):
    """Decoder strategy interface used by backend adapters."""

    name = "base"

    def __init__(self):
        self._last_decorated_prompt: str | None = None

    @abstractmethod
    def prepare_prompt(self, prompt: str, tokenizer: Any) -> str:
        """Prepare a prompt before backend generation."""

    def get_last_decorated_prompt(self) -> str | None:
        """Return the last prepared prompt, if any."""
        return self._last_decorated_prompt

    def validate_ready(self) -> None:
        """Validate decoder runtime prerequisites before generation."""

    def get_generation_kwargs(self, tokenizer: Any) -> dict[str, Any]:
        """Return decoder-specific generation kwargs to merge into model.generate."""
        return {}

    def generate(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
    ) -> str | None:
        """Optional decoder-owned generation path. Return None to use adapter default generation."""
        return None

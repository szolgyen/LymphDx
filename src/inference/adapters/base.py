from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel


class BaseModelAdapter(ABC):
    def __init__(
        self,
        allowed_diagnoses: Optional[set[str]] = None,
        schema_model: Optional[type[BaseModel]] = None,
    ):
        if schema_model is None:
            raise ValueError("schema_model is required")
        self.allowed_diagnoses = allowed_diagnoses
        self.schema_model = schema_model

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return raw model output as string"""

    def extract(self, prompt: str) -> BaseModel:
        raw = self.generate(prompt)
        return self.parse(raw)

    @abstractmethod
    def parse(self, raw: str) -> BaseModel:
        """Convert raw output into validated schema"""

from abc import ABC, abstractmethod
from typing import Optional

from pathology_llm.schemas.pathology import PathologyExtraction


class BaseModelAdapter(ABC):
    def __init__(self, allowed_diagnoses: Optional[set[str]] = None):
        self.allowed_diagnoses = allowed_diagnoses

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return raw model output as string"""

    def extract(self, prompt: str) -> PathologyExtraction:
        raw = self.generate(prompt)
        return self.parse(raw)

    @abstractmethod
    def parse(self, raw: str) -> PathologyExtraction:
        """Convert raw output into validated schema"""

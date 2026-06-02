from copy import deepcopy
from typing import Any


class DiagnosisConstraintsMixin:
    """Adds diagnosis-list constraints on top of a base JSON schema/prompt."""

    _allowed_diagnoses: set[str] | None
    name: str

    def validate_diagnosis_constraints(self) -> None:
        if not self._allowed_diagnoses:
            raise ValueError(
                f"decoder='{self.name}' requires non-empty allowed_diagnoses for diagnosis constraints"
            )

    def build_diagnosis_constraints_prompt_suffix(self) -> str:
        self.validate_diagnosis_constraints()

        diagnosis_terms = "\n".join(
            f"- {term}" for term in sorted(self._allowed_diagnoses or set())
        )
        return (
            f"\n\nSTRICT DECODER MODE ({self.name}):\n"
            "- primary_diagnosis must be selected from the allowed list below.\n"
            "- differential_diagnoses terms must all be selected from the allowed list below.\n"
            "ALLOWED DIAGNOSES:\n"
            f"{diagnosis_terms}\n"
        )

    def apply_diagnosis_constraints_to_schema(
        self,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.validate_diagnosis_constraints()

        allowed = sorted(self._allowed_diagnoses or set())
        constrained = deepcopy(schema)
        properties = constrained.get("properties", {})

        primary_schema = properties.get("primary_diagnosis")
        if isinstance(primary_schema, dict) and "anyOf" in primary_schema:
            primary_schema["anyOf"] = [
                {"type": "string", "enum": allowed},
                {"type": "null"},
            ]

        differential_schema = properties.get("differential_diagnoses")
        if isinstance(differential_schema, dict):
            differential_schema["items"] = {"type": "string", "enum": allowed}

        return constrained

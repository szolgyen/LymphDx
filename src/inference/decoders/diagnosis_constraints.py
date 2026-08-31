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
            f"\n\n\nSTRICT DECODER MODE ({self.name}):\n"
            "- The primary_diagnosis must be selected from the allowed list below.\n"
            "- The containers[].diagnosis must be selected from the allowed list below.\n\n\n"
            "ALLOWED DIAGNOSES:\n"
            f"{diagnosis_terms}\n"
        )

    def _apply_diagnosis_constraints_to_node(
        self,
        node: Any,
        allowed: list[str],
    ) -> None:
        """Recursively apply diagnosis constraints to schema nodes.

        Looks for fields marked with x-reportllm-diagnosis-constrained and applies
        the allowed diagnosis enum constraint to them.
        """
        if not isinstance(node, dict):
            return

        # Check if this property is marked as diagnosis-constrained
        if node.get("x-reportllm-diagnosis-constrained") is True:
            if "anyOf" in node:
                # Replace anyOf with constrained enum and null option
                node["anyOf"] = [
                    {"type": "string", "enum": allowed},
                    {"type": "null"},
                ]
            elif "type" in node or "items" in node:
                # For array items, apply enum constraint directly
                if "items" in node:
                    node["items"] = {"type": "string", "enum": allowed}

        # Recursively process all nested structures
        for key, value in node.items():
            if isinstance(value, dict):
                self._apply_diagnosis_constraints_to_node(value, allowed)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        self._apply_diagnosis_constraints_to_node(item, allowed)

    def apply_diagnosis_constraints_to_schema(
        self,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        self.validate_diagnosis_constraints()

        allowed = sorted(self._allowed_diagnoses or set())
        constrained = deepcopy(schema)

        # Recursively apply constraints to all marked fields in properties
        properties = constrained.get("properties", {})
        for prop_value in properties.values():
            self._apply_diagnosis_constraints_to_node(prop_value, allowed)

        # Recursively apply constraints to all definitions ($defs)
        defs = constrained.get("$defs", {})
        for def_value in defs.values():
            self._apply_diagnosis_constraints_to_node(def_value, allowed)

        return constrained

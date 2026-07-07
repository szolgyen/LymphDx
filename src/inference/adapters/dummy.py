from inference.adapters.base import BaseModelAdapter
from schemas.validation import validate_output


class DummyAdapter(BaseModelAdapter):
    """Dummy adapter for testing pipeline without real model inference."""

    def __init__(
        self,
        allowed_diagnoses=None,
        schema_model=None,
    ):
        super().__init__(allowed_diagnoses=allowed_diagnoses, schema_model=schema_model)

    def generate(self, prompt: str) -> str:
        # Simulated LLM output (intentionally imperfect realism)
        return """
        {
            "schema_version": "v2",
            "diagnosis_primary": "Adenocarcinoma",
            "diagnosis_secondary": [],
            "description": "Moderately differentiated tumor",
            "interpretation_status": "present",
            "specimen": "Colon biopsy",
            "is_lymph_node": false,
            "is_definitive": true,
            "has_differential_diagnosis": false,
            "has_prior_malignancy": null,
            "has_concurrent_malignancy": null,
            "anatomic_location": "Colon",
            "container": null,
            "biomarkers": [
                {"name": "KRAS", "value": "mutated"}
            ],
            "confidence": 0.86
        }
        """

    def parse(self, raw: str):
        return validate_output(
            raw,
            schema_model=self.schema_model,
            allowed_diagnoses=self.allowed_diagnoses,
        )

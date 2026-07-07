from inference.adapters.base import BaseModelAdapter
from schemas.validation import validate_output


class PlaceholderAdapter(BaseModelAdapter):
    """Shared placeholder behavior for not-yet-implemented backends."""

    backend_name = "unknown"

    def __init__(
        self,
        model: str,
        decoder: str,
        allowed_diagnoses: set[str] | None = None,
        schema_model=None,
    ):
        super().__init__(allowed_diagnoses=allowed_diagnoses, schema_model=schema_model)
        self.model = model
        self.decoder = decoder

    def generate(self, prompt: str) -> str:
        raise NotImplementedError(
            "Backend integration is not implemented yet for "
            f"backend='{self.backend_name}', model='{self.model}', decoder='{self.decoder}'."
        )

    def parse(self, raw: str):
        return validate_output(
            raw,
            schema_model=self.schema_model,
            allowed_diagnoses=self.allowed_diagnoses,
        )

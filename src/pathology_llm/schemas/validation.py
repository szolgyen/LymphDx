import json
from pydantic import ValidationError

from pathology_llm.schemas.pathology import PathologyExtraction


class SchemaValidationError(Exception):
    pass


def validate_pathology_output(raw: str | dict) -> PathologyExtraction:
    """
    Validates model output against PathologyExtraction schema.

    Input:
        raw: JSON string or dict from LLM

    Output:
        PathologyExtraction (validated)

    Raises:
        SchemaValidationError if invalid
    """

    try:
        # normalize input
        if isinstance(raw, str):
            data = json.loads(raw)
        else:
            data = raw

        # validate via Pydantic
        return PathologyExtraction.model_validate(data)

    except json.JSONDecodeError as e:
        raise SchemaValidationError(f"Invalid JSON: {e}")

    except ValidationError as e:
        raise SchemaValidationError(f"Schema mismatch: {e}")
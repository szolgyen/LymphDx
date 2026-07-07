from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel

from schemas import pathology


_SCHEMA_NAME_RE = re.compile(r"^schema_(v[0-9][A-Za-z0-9_]*)$")


def _schema_attrs() -> dict[str, type[BaseModel]]:
    schemas: dict[str, type[BaseModel]] = {}
    for attr_name in dir(pathology):
        match = _SCHEMA_NAME_RE.match(attr_name)
        if not match:
            continue
        candidate: Any = getattr(pathology, attr_name)
        if isinstance(candidate, type) and issubclass(candidate, BaseModel):
            schemas[match.group(1)] = candidate
    return schemas


def get_schema_model(schema_key: str) -> type[BaseModel]:
    schemas = _schema_attrs()
    try:
        return schemas[schema_key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown schema '{schema_key}'. Available: {sorted(schemas)}"
        ) from exc


def get_prompt_template_path(schema_key: str) -> str:
    candidates = [
        Path("configs/prompts") / f"prompt_{schema_key}.txt",
        Path("configs/prompts") / f"extraction_{schema_key}.txt",
    ]
    for path in candidates:
        if path.exists():
            return str(path)
    raise FileNotFoundError(
        "No prompt template found for schema "
        f"'{schema_key}'. Expected one of: {[str(path) for path in candidates]}"
    )

import json
from pathlib import Path
from typing import Any


def normalize_case_id(value: Any) -> int | str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value.is_integer() else str(value)

    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    return text


def to_excel_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def load_prediction_records(
    predictions_jsonl: Path,
) -> tuple[dict[int | str, dict[str, Any]], list[str]]:
    prediction_map: dict[int | str, dict[str, Any]] = {}
    key_order: list[str] = []

    for line_number, raw_line in enumerate(
        predictions_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at line {line_number} in {predictions_jsonl}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"Expected JSON object at line {line_number} in {predictions_jsonl}"
            )

        case_id = normalize_case_id(record.get("case_id"))
        if case_id is None:
            continue

        for key in record:
            if key not in key_order:
                key_order.append(key)

        prediction_map[case_id] = record

    return prediction_map, key_order

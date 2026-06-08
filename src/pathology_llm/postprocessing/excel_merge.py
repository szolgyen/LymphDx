from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pathology_llm.postprocessing.predictions import (
    load_prediction_records,
    normalize_case_id,
    to_excel_value,
)


def iter_header_values(values: Iterable[Any]) -> list[str]:
    return [str(value).strip() if value is not None else "" for value in values]


def find_header_index(headers: list[str], target: str) -> int | None:
    if target in headers:
        return headers.index(target)

    target_folded = target.strip().casefold()
    for index, header in enumerate(headers):
        if header.strip().casefold() == target_folded:
            return index
    return None


def normalize_value_for_match(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return str(value).casefold()
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def normalize_containers(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, list):
        items = value
    else:
        items = [value]

    normalized: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            normalized.append(item)
            continue
        if item is None:
            continue
        normalized.append({"label": str(item)})

    return normalized


def resolve_container_label(container: dict[str, Any] | None) -> Any:
    if not container:
        return None
    label = container.get("label")
    return to_excel_value(label)


def resolve_container_field(container: dict[str, Any] | None, field: str) -> Any:
    if not container:
        return None
    return to_excel_value(container.get(field))


def select_container_for_row(
    record: dict[str, Any],
    excel_container_value: Any,
) -> dict[str, Any] | None:
    containers = normalize_containers(record.get("containers"))
    if not containers and "container" in record:
        containers = normalize_containers(record.get("container"))
    if not containers:
        return None

    container_by_label = {}
    for container in containers:
        label = container.get("label")
        if label is None:
            continue
        label_normalized = str(label).strip().casefold()
        if label_normalized:
            container_by_label[label_normalized] = container

    excel_label = (
        str(excel_container_value).strip().casefold() if excel_container_value else None
    )
    if excel_label and excel_label in container_by_label:
        return container_by_label[excel_label]

    return None


def resolve_fallback_container_label(record: dict[str, Any]) -> Any:
    containers = normalize_containers(record.get("containers"))
    if not containers and "container" in record:
        containers = normalize_containers(record.get("container"))

    for container in containers:
        label = resolve_container_label(container)
        if label not in {None, ""}:
            return label
    return None


def resolve_fallback_container_field(record: dict[str, Any], field: str) -> Any:
    containers = normalize_containers(record.get("containers"))
    if not containers and "container" in record:
        containers = normalize_containers(record.get("container"))

    for container in containers:
        value = resolve_container_field(container, field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def apply_match_highlighting_and_marker(
    sheet: Any,
    gt_col_index: int,
    predicted_col_index: int,
    match_marker_col_index: int,
) -> None:
    from openpyxl.styles import PatternFill  # type: ignore[import-not-found]

    match_fill = PatternFill(
        fill_type="solid",
        start_color="C6EFCE",
        end_color="C6EFCE",
    )

    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        gt_value = normalize_value_for_match(row[gt_col_index].value)
        predicted_value = normalize_value_for_match(row[predicted_col_index].value)

        if gt_value is None or predicted_value is None:
            continue

        if gt_value == predicted_value:
            row[predicted_col_index].fill = match_fill
            row[match_marker_col_index].value = "literal_match"


def merge_predictions_into_validation_template(
    input_excel: Path,
    predictions_jsonl: Path,
    output_excel: Path,
) -> None:
    try:
        from openpyxl import load_workbook  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Excel merge requires openpyxl. Install it with: pip install openpyxl"
        ) from exc

    if not input_excel.exists():
        raise FileNotFoundError(f"Template Excel file not found: {input_excel}")
    if not predictions_jsonl.exists():
        raise FileNotFoundError(
            f"Predictions JSONL file not found: {predictions_jsonl}"
        )

    prediction_map, _ = load_prediction_records(predictions_jsonl)

    workbook = load_workbook(filename=input_excel)
    sheet = workbook["Results"]

    if sheet.max_row < 1:
        raise ValueError(f"Excel file has no header row: {input_excel}")

    existing_headers = iter_header_values(
        next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
    )

    required_headers = [
        "Id",
        "Container Duplicate",
        "Report",
        "GT Report Diagnosis",
        "Predicted Report Diagnosis",
        "Predicted Report Diagnosis Match",
        "GT Container",
        "Predicted Container",
        "Predicted Container Match",
        "GT Has Differential Diagnosis",
        "Predicted Has Differential Diagnosis",
        "Predicted Has Differential Diagnosis Match",
        "GT Is Lymph Node",
        "Predicted Is Lymph Node",
        "Predicted Is Lymph Node Match",
        "GT Is Definitive",
        "Predicted Is Definitive",
        "Predicted Is Definitive Match",
        "GT Has Prior Malignancy",
        "Predicted Has Prior Malignancy",
        "Predicted Has Prior Malignancy Match",
        "GT Has Concurrent Malignancy",
        "Predicted Has Concurrent Malignancy",
        "Predicted Has Concurrent Malignancy Match",
        "GT Container Diagnosis",
        "Predicted Container Diagnosis",
    ]

    for header in required_headers:
        if header not in existing_headers:
            raise ValueError(
                f"Required header '{header}' not found in template. Found: {existing_headers}"
            )

    header_indexes = {h: i for i, h in enumerate(existing_headers)}

    # Read all existing rows - keep them as-is
    base_rows: list[list[Any]] = []
    for row in sheet.iter_rows(
        min_row=2,
        max_row=sheet.max_row,
        min_col=1,
        max_col=len(existing_headers),
        values_only=True,
    ):
        base_rows.append(list(row))

    # Group rows by case_id
    rows_by_case_id: dict[int | str | None, list[tuple[int, list[Any]]]] = {}
    for row_num_offset, base_row in enumerate(base_rows, start=2):
        row_id_value = base_row[header_indexes["Id"]]
        normalized_id = normalize_case_id(row_id_value)
        if normalized_id not in rows_by_case_id:
            rows_by_case_id[normalized_id] = []
        rows_by_case_id[normalized_id].append((row_num_offset, base_row))

    # For each case_id, fill rows with containers from JSONL in order
    for normalized_id, rows_for_case in rows_by_case_id.items():
        record = prediction_map.get(normalized_id)

        if record is None:
            continue

        containers = normalize_containers(record.get("containers"))

        # Fill as many rows as exist for this case_id with containers in order
        for row_offset, (row_num, base_row) in enumerate(rows_for_case):
            # Get container at this index (or None if we run out of containers)
            container = containers[row_offset] if row_offset < len(containers) else None

            # Fill Predicted Report Diagnosis (case-level)
            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Report Diagnosis"] + 1,
                value=to_excel_value(record.get("primary_diagnosis")),
            )

            # Fill Predicted Container (from container label)
            predicted_container = resolve_container_label(container)
            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Container"] + 1,
                value=predicted_container,
            )

            # Fill container-level fields from the matched container
            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Has Differential Diagnosis"] + 1,
                value=to_excel_value(record.get("has_differential_diagnosis")),
            )

            container_is_lymph = resolve_container_field(container, "is_lymph_node")
            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Is Lymph Node"] + 1,
                value=container_is_lymph,
            )

            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Is Definitive"] + 1,
                value=to_excel_value(record.get("is_definitive")),
            )

            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Has Prior Malignancy"] + 1,
                value=to_excel_value(record.get("has_prior_malignancy")),
            )

            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Has Concurrent Malignancy"] + 1,
                value=to_excel_value(record.get("has_concurrent_malignancy")),
            )

            container_diagnosis = resolve_container_field(container, "diagnosis")
            sheet.cell(
                row=row_num,
                column=header_indexes["Predicted Container Diagnosis"] + 1,
                value=container_diagnosis,
            )

    match_pairs = [
        (
            "GT Report Diagnosis",
            "Predicted Report Diagnosis",
            "Predicted Report Diagnosis Match",
        ),
        ("GT Container", "Predicted Container", "Predicted Container Match"),
        (
            "GT Has Differential Diagnosis",
            "Predicted Has Differential Diagnosis",
            "Predicted Has Differential Diagnosis Match",
        ),
        (
            "GT Is Lymph Node",
            "Predicted Is Lymph Node",
            "Predicted Is Lymph Node Match",
        ),
        (
            "GT Is Definitive",
            "Predicted Is Definitive",
            "Predicted Is Definitive Match",
        ),
        (
            "GT Has Prior Malignancy",
            "Predicted Has Prior Malignancy",
            "Predicted Has Prior Malignancy Match",
        ),
        (
            "GT Has Concurrent Malignancy",
            "Predicted Has Concurrent Malignancy",
            "Predicted Has Concurrent Malignancy Match",
        ),
    ]

    for gt_header, predicted_header, match_header in match_pairs:
        gt_col_index = header_indexes.get(gt_header)
        predicted_col_index = header_indexes.get(predicted_header)
        match_marker_col_index = header_indexes.get(match_header)

        if (
            gt_col_index is None
            or predicted_col_index is None
            or match_marker_col_index is None
        ):
            continue

        apply_match_highlighting_and_marker(
            sheet=sheet,
            gt_col_index=gt_col_index,
            predicted_col_index=predicted_col_index,
            match_marker_col_index=match_marker_col_index,
        )

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_excel)

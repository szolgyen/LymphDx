from collections.abc import Iterable
from pathlib import Path
from typing import Any

from pathology_llm.postprocessing.predictions import (
    load_prediction_records,
    normalize_case_id,
    to_excel_value,
)


def ensure_unique_column_name(existing_headers: set[str], name: str) -> str:
    if name not in existing_headers:
        return name

    index = 1
    candidate = f"pred_{name}"
    while candidate in existing_headers:
        index += 1
        candidate = f"pred_{name}_{index}"
    return candidate


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


def normalize_text_for_match(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()


def count_string_matches(
    sheet: Any,
    validation_col_index: int,
    primary_col_index: int,
) -> tuple[int, list[int]]:
    match_count = 0
    matched_rows: list[int] = []
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        validation_value = normalize_text_for_match(row[validation_col_index].value)
        primary_value = normalize_text_for_match(row[primary_col_index].value)
        if validation_value is not None and validation_value == primary_value:
            match_count += 1
            matched_rows.append(row[0].row)
    return match_count, matched_rows


def count_validation_matches_any_differential(
    sheet: Any,
    validation_col_index: int,
    number_col_index: int,
    differential_rows: list[tuple[Any, list[str]]],
) -> int:
    differentials_by_case: dict[int | str, set[str]] = {}
    for raw_case_id, differentials in differential_rows:
        normalized_case_id = normalize_case_id(raw_case_id)
        if normalized_case_id is None:
            continue

        case_differentials = differentials_by_case.setdefault(normalized_case_id, set())
        for diagnosis in differentials:
            normalized_diagnosis = normalize_text_for_match(diagnosis)
            if normalized_diagnosis is not None:
                case_differentials.add(normalized_diagnosis)

    match_count = 0
    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        normalized_case_id = normalize_case_id(row[number_col_index].value)
        validation_value = normalize_text_for_match(row[validation_col_index].value)
        if normalized_case_id is None or validation_value is None:
            continue

        case_differentials = differentials_by_case.get(normalized_case_id, set())
        if validation_value in case_differentials:
            match_count += 1

    return match_count


def highlight_primary_diagnosis_matches(
    sheet: Any,
    primary_col_index: int,
    matched_rows: list[int],
) -> None:
    from openpyxl.styles import PatternFill  # type: ignore[import-not-found]

    match_fill = PatternFill(
        fill_type="solid",
        start_color="C6EFCE",
        end_color="C6EFCE",
    )
    for row_number in matched_rows:
        cell = sheet.cell(row=row_number, column=primary_col_index + 1)
        cell.fill = match_fill


def write_match_summary_sheet(
    workbook: Any,
    primary_match_count: int,
    differential_match_count: int,
) -> None:
    summary_sheet_name = "match_summary"
    if summary_sheet_name in workbook.sheetnames:
        summary_sheet = workbook[summary_sheet_name]
    else:
        summary_sheet = workbook.create_sheet(title=summary_sheet_name)

    summary_sheet.cell(row=1, column=1, value="metric")
    summary_sheet.cell(row=1, column=2, value="value")
    summary_sheet.cell(
        row=2,
        column=1,
        value="Validation Diagnosis == primary_diagnosis string matches",
    )
    summary_sheet.cell(row=2, column=2, value=primary_match_count)
    summary_sheet.cell(
        row=3,
        column=1,
        value=(
            "Validation Diagnosis matches any differential diagnosis (joined by Number)"
        ),
    )
    summary_sheet.cell(row=3, column=2, value=differential_match_count)


def normalize_differential_diagnoses(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = str(item).strip()
        if not text:
            continue
        folded = text.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        normalized.append(text)
    return normalized


def write_differential_sheet(
    workbook: Any,
    case_id_header: str,
    rows: list[tuple[Any, list[str]]],
) -> None:
    sheet_name = "differential_diagnoses"
    if sheet_name in workbook.sheetnames:
        old_sheet = workbook[sheet_name]
        workbook.remove(old_sheet)
    sheet = workbook.create_sheet(title=sheet_name)

    max_differentials = 0
    for _, differentials in rows:
        if len(differentials) > max_differentials:
            max_differentials = len(differentials)

    sheet.cell(row=1, column=1, value=case_id_header)
    for differential_index in range(1, max_differentials + 1):
        sheet.cell(
            row=1,
            column=1 + differential_index,
            value=f"differential_diagnosis_{differential_index}",
        )

    for row_index, (case_id, differentials) in enumerate(rows, start=2):
        sheet.cell(row=row_index, column=1, value=case_id)
        for differential_index, diagnosis_text in enumerate(differentials, start=1):
            sheet.cell(
                row=row_index,
                column=1 + differential_index,
                value=diagnosis_text,
            )


def order_prediction_keys(
    prediction_keys: list[str],
    prediction_id_key: str,
) -> list[str]:
    ordered_keys = [
        key
        for key in prediction_keys
        if key not in {prediction_id_key, "differential_diagnoses"}
    ]
    if "schema_version" in ordered_keys:
        ordered_keys = [key for key in ordered_keys if key != "schema_version"]
        ordered_keys.append("schema_version")
    return ordered_keys


def build_excel_range(min_col: int, min_row: int, max_col: int, max_row: int) -> str:
    from openpyxl.utils.cell import get_column_letter  # type: ignore[import-not-found]

    return (
        f"{get_column_letter(min_col)}{min_row}:{get_column_letter(max_col)}{max_row}"
    )


def find_target_table(
    sheet: Any,
    input_id_index: int,
    existing_column_count: int,
) -> Any | None:
    from openpyxl.utils.cell import range_boundaries  # type: ignore[import-not-found]

    fallback_table = None
    input_id_column = input_id_index + 1

    for table in sheet.tables.values():
        min_col, min_row, max_col, _ = range_boundaries(table.ref)
        if min_row != 1:
            continue
        if not min_col <= input_id_column <= max_col:
            continue
        if max_col == existing_column_count:
            return table
        if fallback_table is None:
            fallback_table = table

    return fallback_table


def extend_table_columns(table: Any, added_headers: list[str]) -> None:
    from openpyxl.worksheet.table import TableColumn  # type: ignore[import-not-found]

    if not added_headers:
        return

    next_column_id = 1
    if table.tableColumns:
        next_column_id = max(column.id for column in table.tableColumns) + 1

    for header_name in added_headers:
        table.tableColumns.append(TableColumn(id=next_column_id, name=header_name))
        next_column_id += 1


def extend_table_ref(
    table: Any,
    new_last_col: int,
    new_last_row: int,
) -> None:
    from openpyxl.utils.cell import range_boundaries  # type: ignore[import-not-found]

    min_col, min_row, _, _ = range_boundaries(table.ref)
    table.ref = build_excel_range(min_col, min_row, new_last_col, new_last_row)
    if table.autoFilter is not None:
        table.autoFilter.ref = table.ref


def merge_predictions_with_input(
    input_excel: Path,
    predictions_jsonl: Path,
    output_excel: Path,
    input_id_column: str = "Number",
    prediction_id_key: str = "case_id",
    missing_value: str = "NA",
) -> None:
    try:
        from openpyxl import load_workbook  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "Excel merge requires openpyxl. Install it with: pip install openpyxl"
        ) from exc

    if not input_excel.exists():
        raise FileNotFoundError(f"Input Excel file not found: {input_excel}")
    if not predictions_jsonl.exists():
        raise FileNotFoundError(
            f"Predictions JSONL file not found: {predictions_jsonl}"
        )

    prediction_map, prediction_keys = load_prediction_records(predictions_jsonl)
    ordered_prediction_keys = order_prediction_keys(
        prediction_keys=prediction_keys,
        prediction_id_key=prediction_id_key,
    )

    workbook = load_workbook(filename=input_excel)
    sheet = workbook.active

    if sheet.max_row < 1:
        raise ValueError(f"Excel file has no header row: {input_excel}")

    existing_headers = iter_header_values(
        next(sheet.iter_rows(min_row=1, max_row=1, values_only=True))
    )
    if input_id_column not in existing_headers:
        raise ValueError(
            f"Excel file must contain '{input_id_column}' column. Found: {existing_headers}"
        )

    input_id_index = existing_headers.index(input_id_column)

    prediction_output_columns: list[tuple[str, str]] = []
    used_headers = set(existing_headers)
    for key in ordered_prediction_keys:
        header_name = ensure_unique_column_name(used_headers, key)
        used_headers.add(header_name)
        prediction_output_columns.append((key, header_name))

    output_columns = prediction_output_columns

    start_col = len(existing_headers) + 1
    for col_offset, (_, header_name) in enumerate(output_columns):
        sheet.cell(row=1, column=start_col + col_offset, value=header_name)

    differential_rows: list[tuple[Any, list[str]]] = []

    for row in sheet.iter_rows(min_row=2, max_row=sheet.max_row):
        excel_case_raw_value = row[input_id_index].value
        excel_case_id = normalize_case_id(row[input_id_index].value)
        record = prediction_map.get(excel_case_id)

        for col_offset, (key, _) in enumerate(prediction_output_columns):
            if record is None:
                out_value = missing_value
            else:
                out_value = to_excel_value(record.get(key))
            sheet.cell(row=row[0].row, column=start_col + col_offset, value=out_value)

        if record is None:
            differential_rows.append((excel_case_raw_value, []))
        else:
            differentials = normalize_differential_diagnoses(
                record.get("differential_diagnoses")
            )
            differential_rows.append((excel_case_raw_value, differentials))

    output_headers = [header for _, header in prediction_output_columns]
    primary_output_header = next(
        (
            output_header
            for key, output_header in prediction_output_columns
            if key == "primary_diagnosis"
        ),
        None,
    )
    validation_index = find_header_index(existing_headers, "Validation Diagnosis")
    primary_offset = (
        output_headers.index(primary_output_header)
        if primary_output_header in output_headers
        else None
    )
    primary_index = (
        start_col - 1 + primary_offset if primary_offset is not None else None
    )

    match_count = 0
    if validation_index is not None and primary_index is not None:
        match_count, matched_rows = count_string_matches(
            sheet=sheet,
            validation_col_index=validation_index,
            primary_col_index=primary_index,
        )
        highlight_primary_diagnosis_matches(
            sheet=sheet,
            primary_col_index=primary_index,
            matched_rows=matched_rows,
        )

    differential_match_count = 0
    if validation_index is not None:
        differential_match_count = count_validation_matches_any_differential(
            sheet=sheet,
            validation_col_index=validation_index,
            number_col_index=input_id_index,
            differential_rows=differential_rows,
        )

    write_match_summary_sheet(
        workbook,
        primary_match_count=match_count,
        differential_match_count=differential_match_count,
    )
    write_differential_sheet(
        workbook=workbook,
        case_id_header=input_id_column,
        rows=differential_rows,
    )

    target_table = find_target_table(
        sheet=sheet,
        input_id_index=input_id_index,
        existing_column_count=len(existing_headers),
    )
    if target_table is not None:
        added_headers = [header_name for _, header_name in output_columns]
        extend_table_columns(target_table, added_headers)
        extend_table_ref(
            table=target_table,
            new_last_col=start_col + len(output_columns) - 1,
            new_last_row=sheet.max_row,
        )

    output_excel.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_excel)

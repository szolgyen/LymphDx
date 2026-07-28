import pandas as pd
from transformers import pipeline
import torch
import argparse
import re
import unicodedata

from tqdm import tqdm


def load_deid_model():
    """
    Load the Hugging Face token-classification pipeline used for PHI tagging.

    Returns:
        transformers.Pipeline: Configured token-classification pipeline for
        de-identification.
    """
    print("Loading de-identification model...")

    device = 0 if torch.cuda.is_available() else -1

    deid_pipeline = pipeline(
        "token-classification",
        model="obi/deid_roberta_i2b2",
        aggregation_strategy="simple",
        device=device,
    )

    print("Model loaded successfully.")
    print(f"Model: {deid_pipeline.model.name_or_path}")
    print(f"Tokenizer: {deid_pipeline.tokenizer.name_or_path}")
    print(f"Max model tokens: {deid_pipeline.model.config.max_position_embeddings}")

    return deid_pipeline


def read_excel_file(input_file, text_column):
    """
    Read the input workbook and verify that the target text column exists.

    Args:
        input_file: Path to the Excel workbook that contains pathology reports.
        text_column: Name of the column that stores the report text.

    Returns:
        pandas.DataFrame: Loaded workbook contents.

    Raises:
        ValueError: If text_column is not present in the workbook.
    """

    print("Reading input Excel file...")
    df = pd.read_excel(input_file, engine="openpyxl")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in input file.")

    return df


def remove_overlapping_entities(entities):
    """
    Keep the strongest non-overlapping entity spans.

    Args:
        entities: List of entity dictionaries with start, end, entity_group,
        and score keys.

    Returns:
        list[dict]: Non-overlapping entities sorted by start offset.
    """
    entities = sorted(
        entities,
        key=lambda x: (
            x["score"],
            x["end"] - x["start"]
        ),
        reverse=True
    )

    kept = []

    for entity in entities:
        overlap = False

        for existing in kept:
            # Later replacement operates on character spans, so overlaps must be
            # resolved before any text substitution happens.
            if (
                entity["start"] < existing["end"]
                and entity["end"] > existing["start"]
            ):
                overlap = True
                break

        if not overlap:
            kept.append(entity)

    return sorted(
        kept,
        key=lambda x: x["start"]
    )


def merge_adjacent_entities(entities, max_gap=1):
    """
    Merge neighboring spans from the same entity class when separated by tiny gaps.

    Args:
        entities: List of entity dictionaries sorted or unsorted by offset.
        max_gap: Maximum number of characters allowed between spans for them to
        be merged.

    Returns:
        list[dict]: Entity spans with small same-label gaps collapsed.
    """
    entities = sorted(
        entities,
        key=lambda x: x["start"]
    )

    merged = []

    for entity in entities:
        if not merged:
            merged.append(entity)
            continue

        previous = merged[-1]
        gap = entity["start"] - previous["end"]

        if (
            entity["entity_group"] == previous["entity_group"]
            and gap <= max_gap
        ):
            previous["end"] = max(previous["end"], entity["end"])
            previous["score"] = max(
                previous.get("score", 0.0),
                entity.get("score", 0.0)
            )
        else:
            merged.append(entity)

    return merged


def standardize_report_text(text):
    """
    Normalize report text so token offsets are more stable across malformed input.

    Args:
        text: Raw report text or any value convertible to a string.

    Returns:
        str: Normalized report text.
    """
    standardized = str(text).lower()

    try:
        standardized = standardized.encode("latin1").decode("utf8")
    except Exception:
        pass

    standardized = unicodedata.normalize("NFKC", standardized)
    standardized = standardized.replace("\u00A0", " ")
    standardized = standardized.replace("Â", "")
    standardized = re.sub(r"[\x00-\x1F\x7F]", " ", standardized)
    standardized = re.sub(r"[\r\n\t]+", " ", standardized)
    standardized = re.sub(r"\s+", " ", standardized)

    return standardized.strip()


def expand_identifier_span(text, start, end, id_chars=set("0123456789-/#().:+")):
    """
    Expand ID-like entities to include adjacent identifier punctuation or digits.

    Args:
        text: Full report text containing the detected entity.
        start: Inclusive start offset of the detected entity.
        end: Exclusive end offset of the detected entity.
        id_chars: Characters that should be absorbed into identifier-like spans.

    Returns:
        tuple[int, int]: Expanded start and end offsets.
    """
    while start > 0 and text[start - 1] in id_chars:
        start -= 1

    while end < len(text) and text[end] in id_chars:
        end += 1

    return start, end


def deidentify_text(text, deid_pipeline, min_confidence_threshold=0.75):
    """
    Replace detected PHI spans in one report using token-based chunking.

    Uses tokenizer offsets to:
    - Respect model token limits
    - Maintain original document offsets
    - Add overlap between chunks
    - Avoid missing PHI at chunk boundaries

    Args:
        text: Raw report text to de-identify.
        deid_pipeline: Loaded Hugging Face token-classification pipeline.
        min_confidence_threshold: Minimum score required before a detected
        entity is replaced.

    Returns:
        str: De-identified report text with PHI spans replaced by labels.
    """

    if pd.isna(text):
        return ""

    text = standardize_report_text(text)

    tokenizer = deid_pipeline.tokenizer

    # Leave room for special tokens
    max_model_tokens = min(
        deid_pipeline.model.config.max_position_embeddings,
        512  # safety cap for BERT/RoBERTa variants
    )

    chunk_tokens = max_model_tokens - 64
    overlap_tokens = 128

    # Tokenize entire document without truncation
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )

    offsets = encoding["offset_mapping"]

    entities = []
    seen = set()

    # Overlap adjacent windows so entities near chunk boundaries are not missed.
    step = chunk_tokens - overlap_tokens

    for token_start_idx in range(0, len(offsets), step):

        token_end_idx = min(
            token_start_idx + chunk_tokens,
            len(offsets)
        )

        chunk_offsets = offsets[token_start_idx:token_end_idx]

        if not chunk_offsets:
            continue

        # Convert token positions back to original text positions
        char_start = chunk_offsets[0][0]
        char_end = chunk_offsets[-1][1]

        chunk_text = text[char_start:char_end]

        try:
            chunk_entities = deid_pipeline(chunk_text)

            for entity in chunk_entities:

                global_start = entity["start"] + char_start
                global_end = entity["end"] + char_start
                score = float(entity.get("score", 0.0))

                if score < min_confidence_threshold:
                    continue

                if entity["entity_group"] in {"ID", "PHONE"}:
                    global_start, global_end = expand_identifier_span(
                        text,
                        global_start,
                        global_end
                    )

                key = (
                    global_start,
                    global_end,
                    entity["entity_group"]
                )

                if key not in seen:
                    seen.add(key)

                    entities.append({
                        "start": global_start,
                        "end": global_end,
                        "entity_group": entity["entity_group"],
                        "score": score
                    })

        except Exception as e:
            print(
                f"Chunk error "
                f"(chars {char_start}-{char_end}): {e}"
            )

    # Replace from right-to-left so earlier replacements do not shift later spans.
    entities = remove_overlapping_entities(entities)
    entities = merge_adjacent_entities(entities)
    
    result = text

    for entity in sorted(
        entities,
        key=lambda x: x["start"],
        reverse=True
    ):

        start = entity["start"]
        end = entity["end"]
        label = entity["entity_group"]

        result = (
            result[:start]
            + f"[{label}({end - start}/{entity['score']:.2f})]"
            + result[end:]
        )
        
    return result


def process_reports(df, deid_pipeline, text_column, min_confidence_threshold=0.75):
    """
    Run de-identification across every report in the selected DataFrame column.

    Args:
        df: Input DataFrame containing report data.
        deid_pipeline: Loaded Hugging Face token-classification pipeline.
        text_column: Name of the column containing report text.
        min_confidence_threshold: Minimum score required before a detected
        entity is replaced.

    Returns:
        list[str]: De-identified report text for each input row.
    """

    print("Processing reports...")

    deidentified_reports = []

    for i, report in enumerate(tqdm(df[text_column], desc="Processing reports")):
        try:
            cleaned = deidentify_text(
                str(report),
                deid_pipeline,
                min_confidence_threshold=min_confidence_threshold
            )
        except Exception as e:
            print(f"Error processing report {i}: {e}")
            cleaned = ""
        deidentified_reports.append(cleaned)

    return deidentified_reports


def save_output(df, deidentified_reports, output_file):
    """
    Write the de-identified reports back to an Excel workbook.

    Args:
        df: Original DataFrame to augment with de-identified output.
        deidentified_reports: De-identified report text aligned to df rows.
        output_file: Destination path for the output Excel workbook.
    """
    df["Deidentified_Reports"] = deidentified_reports

    print("Saving output Excel file...")
    df.to_excel(output_file, index=False, engine="openpyxl")

    print("Done! Output saved to:", output_file)


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line interface for the standalone de-identification script.

    Returns:
        argparse.ArgumentParser: Parser configured with script options.
    """
    parser = argparse.ArgumentParser(
        description=(
            "De-identify pathology reports in an Excel file using a pre-trained model, replacing PHI with tags."
        )
    )
    parser.add_argument(
        "--input-excel",
        default="NM_Surgical_Path_LN_validation.xlsx",
        help="Path to Excel file with reports having PHI",
    )
    parser.add_argument(
        "--output-excel",
        default="configs/extraction/deidentified_reports.xlsx",
        help="Path to deidentified reports Excel file",
    )
    parser.add_argument(
        "--text-column",
        default="Report",
        help="Name of the column containing the reports",
    )
    parser.add_argument(
        "--min-confidence-threshold",
        type=float,
        default=0.7,
        help="Minimum entity confidence required before replacement",
    )
    return parser


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for standalone execution.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    return build_parser().parse_args()


def main():
    """
    Run the end-to-end Excel de-identification workflow.

    Returns:
        None: The script writes output to disk and exits through SystemExit.
    """
    args = parse_args()

    input_file = args.input_excel
    output_file = args.output_excel
    text_column = args.text_column
    min_confidence_threshold = args.min_confidence_threshold

    deid_pipeline = load_deid_model()

    df = read_excel_file(input_file, text_column)
    deidentified_reports = process_reports(
        df,
        deid_pipeline,
        text_column,
        min_confidence_threshold=min_confidence_threshold,
    )
    save_output(df, deidentified_reports, output_file)


if __name__ == "__main__":
    raise SystemExit(main())
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml

from extraction.pipeline import ExtractionPipeline
from inference.adapters.factory import create_adapter
from schemas.registry import get_prompt_template_path, get_schema_model
from utils.logging_config import configure_logging
from utils.utils import (
    load_reports,
    load_diagnosis_terms,
    prepare_output_store,
    write_output_record,
    write_prompt_record,
    write_broken_extraction_record,
)


logger = logging.getLogger(__name__)


CONFIG_DEFAULTS: dict[str, Any] = {
    "backend": "hf",
    "model": "google/medgemma-4b-it",
    "decoder": "none",
    "schema": "v5",
    "input_file": None,
    "diagnosis_dictionary": "inputs/LN_Dx_dictionary_codes_20260824.xlsx",
    "output_dir": "outputs/predictions",
    "log_level": "INFO",
}

CHOICES: dict[str, set[str]] = {
    "backend": {"hf", "vllm", "sglang", "ollama"},
    "decoder": {"none", "guidance", "outlines"},
    "log_level": {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pathology extraction pipeline from YAML config"
    )
    parser.add_argument(
        "--config",
        default="configs/llm_pipeline.yaml",
        help="Path to YAML config file containing all pipeline options",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict[str, Any]:

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError("Pipeline config must be a YAML mapping/object")

    unknown_keys = sorted(set(raw) - set(CONFIG_DEFAULTS))
    if unknown_keys:
        raise ValueError(f"Unknown config keys: {unknown_keys}")

    config = dict(CONFIG_DEFAULTS)
    config.update(raw)

    if not config["input_file"]:
        raise ValueError("Config key 'input_file' is required")

    for key, allowed in CHOICES.items():
        value = str(config[key]).strip()
        if value not in allowed:
            raise ValueError(
                f"Invalid value for '{key}': {value!r}. Allowed: {sorted(allowed)}"
            )
        config[key] = value

    return config


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    schema_model = get_schema_model(config["schema"])
    prompt_template_path = get_prompt_template_path(config["schema"])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_output_dir = Path(config["output_dir"]) / timestamp
    log_file = timestamped_output_dir / f"run_pipeline_{timestamp}.log"
    configure_logging(config["log_level"], log_file=str(log_file))

    try:
        logger.info("Logging to file %s", log_file)
        logger.info(
            "Starting pipeline backend=%s model=%s decoder=%s",
            config["backend"],
            config["model"],
            config["decoder"],
        )
        if config["decoder"] != "none":
            allowed_diagnoses = load_diagnosis_terms(config["diagnosis_dictionary"])
        else:
            allowed_diagnoses = None
        reports = load_reports(config["input_file"])
        adapter = create_adapter(
            backend=config["backend"],
            model=config["model"],
            decoder=config["decoder"],
            allowed_diagnoses=allowed_diagnoses,
            schema_model=schema_model,
        )
        pipeline = ExtractionPipeline(
            adapter=adapter,
            prompt_template_path=prompt_template_path,
            allowed_diagnoses=allowed_diagnoses,
            decoder_name=config["decoder"],
        )
        prepare_output_store(str(timestamped_output_dir), timestamp=timestamp)

        def _persist_output(report_index: int, extraction_output, prompt: str) -> None:
            write_output_record(
                output_dir=str(timestamped_output_dir),
                report_index=report_index,
                output=extraction_output.model_dump(),
                timestamp=timestamp,
            )
            write_prompt_record(
                output_dir=str(timestamped_output_dir),
                report_index=report_index,
                prompt=prompt,
            )

        def _persist_broken_output(
            report_index: int, raw_output: str | None, error_message: str
        ) -> None:
            write_broken_extraction_record(
                output_dir=str(timestamped_output_dir),
                report_index=report_index,
                raw_output=raw_output,
                error_message=error_message,
                timestamp=timestamp,
            )

        extraction_outputs = pipeline.extract_reports(
            reports,
            on_success=_persist_output,
            on_error=_persist_broken_output,
        )
        if reports and not extraction_outputs:
            logger.error(
                "Extraction produced zero valid outputs; see logs for per-report errors"
            )
            raise RuntimeError(
                "Extraction produced zero valid outputs; see logs for per-report errors"
            )
        logger.info(
            "Pipeline completed successfully reports=%d output_dir=%s (incremental writes enabled)",
            len(extraction_outputs),
            config["output_dir"],
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

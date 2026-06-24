import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml

from pathology_llm.extraction.pipeline import ExtractionPipeline
from pathology_llm.inference.adapters.factory import create_adapter
from pathology_llm.schemas.registry import get_prompt_template_path, get_schema_model
from pathology_llm.utils.logging_config import configure_logging
from pathology_llm.utils.utils import (
    load_reports,
    load_diagnosis_terms,
    prepare_output_store,
    write_output_record,
    write_prompt_record,
    write_broken_extraction_record,
)


logger = logging.getLogger(__name__)


CONFIG_DEFAULTS: dict[str, Any] = {
    "backend": "dummy",
    "model": "google/medgemma-4b-it",
    "decoder": "auto",
    "schema": "v2",
    "input_file": None,
    "diagnosis_terms_file": "configs/extraction/diagnosis_terms_v1.txt",
    "output_dir": "outputs/predictions",
    "log_level": "INFO",
}

CHOICES: dict[str, set[str]] = {
    "backend": {"dummy", "hf", "vllm", "sglang", "ollama"},
    "decoder": {"auto", "none", "sglang", "guidance", "outlines"},
    "log_level": {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pathology extraction pipeline from YAML config"
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline/run_pipeline.yaml",
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
    log_file = Path("outputs/logs") / f"run_pipeline_{timestamp}.log"
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
            allowed_diagnoses = load_diagnosis_terms(config["diagnosis_terms_file"])
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
            include_diagnosis_constraints=(config["decoder"] != "none"),
        )
        prepare_output_store(config["output_dir"])

        def _persist_output(report_index: int, extraction_output, prompt: str) -> None:
            write_output_record(
                output_dir=config["output_dir"],
                report_index=report_index,
                output=extraction_output.model_dump(),
            )
            write_prompt_record(
                output_dir=config["output_dir"],
                report_index=report_index,
                prompt=prompt,
            )

        def _persist_broken_output(
            report_index: int, raw_output: str | None, error_message: str
        ) -> None:
            write_broken_extraction_record(
                output_dir=config["output_dir"],
                report_index=report_index,
                raw_output=raw_output,
                error_message=error_message,
            )

        extraction_outputs = pipeline.extract_reports(
            reports,
            on_success=_persist_output,
            on_error=_persist_broken_output,
        )
        if reports and not extraction_outputs:
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

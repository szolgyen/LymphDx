import argparse
import logging
from datetime import datetime
from pathlib import Path

from pathology_llm.extraction.pipeline import ExtractionPipeline
from pathology_llm.inference.factory import create_adapter
from pathology_llm.utils.logging_config import configure_logging
from pathology_llm.utils.utils import load_reports, load_diagnosis_terms, write_outputs


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pathology extraction pipeline")
    parser.add_argument(
        "--backend",
        default="dummy",
        choices=["dummy", "hf", "vllm", "sglang", "ollama"],
        help="Inference backend",
    )
    parser.add_argument(
        "--model",
        default="google/medgemma-4b-it",
        help="Model identifier for selected backend",
    )
    parser.add_argument(
        "--decoder",
        default="auto",
        choices=["auto", "none", "sglang", "guidance", "outlines"],
        help="Constrained decoder implementation",
    )
    parser.add_argument(
        "--input-file",
        required=True,
        help="Path to .txt file with one report per line",
    )
    parser.add_argument(
        "--prompt-template",
        default="configs/prompts/extraction_v1.txt",
        help="Prompt template path",
    )
    parser.add_argument(
        "--diagnosis-terms-file",
        default="configs/extraction/diagnosis_terms_v1.txt",
        help="Text file with one allowed diagnosis per line",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/predictions",
        help="Directory for extracted JSON outputs",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Framework log level",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("outputs/logs") / f"run_pipeline_{timestamp}.log"
    configure_logging(args.log_level, log_file=str(log_file))

    try:
        logger.info("Logging to file %s", log_file)
        logger.info(
            "Starting pipeline backend=%s model=%s decoder=%s",
            args.backend,
            args.model,
            args.decoder,
        )
        allowed_diagnoses = load_diagnosis_terms(args.diagnosis_terms_file)
        reports = load_reports(args.input_file)
        adapter = create_adapter(
            backend=args.backend,
            model=args.model,
            decoder=args.decoder,
            allowed_diagnoses=allowed_diagnoses,
        )
        pipeline = ExtractionPipeline(
            adapter=adapter,
            prompt_template_path=args.prompt_template,
            allowed_diagnoses=allowed_diagnoses,
        )
        extraction_outputs = pipeline.extract_reports(reports)
        if reports and not extraction_outputs:
            raise RuntimeError(
                "Extraction produced zero valid outputs; see logs for per-report errors"
            )
        write_outputs(args.output_dir, [obj.model_dump() for obj in extraction_outputs])
        logger.info(
            "Pipeline completed successfully reports=%d output_dir=%s",
            len(extraction_outputs),
            args.output_dir,
        )
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

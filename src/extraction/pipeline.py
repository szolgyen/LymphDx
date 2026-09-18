import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable
from pydantic import BaseModel

from inference.adapters.base import BaseModelAdapter
from preprocessing.data_parsing import ParsedReport
from prompting.prompt_builder import build_extraction_prompt_from_template
from schemas.validation import SchemaValidationError


logger = logging.getLogger(__name__)


class ExtractionPipeline:
    def __init__(
        self,
        adapter: BaseModelAdapter,
        prompt_template_path: str,
        allowed_diagnoses: set[str],
        decoder_name: str = "none",
        concurrency: int = 1,
    ):
        self.adapter = adapter
        self.prompt_template_path = prompt_template_path
        self.allowed_diagnoses = allowed_diagnoses
        self.decoder_name = decoder_name
        self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")

        # Concurrency handling based on backend
        backend_name = getattr(adapter, "backend_name", "unknown")
        if backend_name == "hf" and concurrency > 1:
            logger.warning(
                "Disabling concurrency for HF adapter (setting to 1). "
                "For multi-report GPU inference, use vLLM backend."
            )
            self.concurrency = 1
        else:
            self.concurrency = max(1, concurrency)

    def extract_reports(
        self,
        reports: list[ParsedReport],
        on_success: Callable[[int, BaseModel, str, str], None] | None = None,
        on_error: Callable[[int, str | None, str], None] | None = None,
    ) -> list[BaseModel]:
        backend_name = getattr(self.adapter, "backend_name", "unknown")
        parallelization_strategy = (
            f"threading ({self.concurrency} workers)"
            if self.concurrency > 1
            else "sequential"
        )
        logger.info(
            "Extracting structured outputs for %d reports (backend=%s, parallelization=%s)",
            len(reports),
            backend_name,
            parallelization_strategy,
        )
        outputs: list[BaseModel] = []
        errors: dict[int, str] = {}

        def _process_report(report_data: tuple[int, ParsedReport]) -> tuple[int, BaseModel | None]:
            """Process a single report and return (index, output or None)."""
            idx, report = report_data
            try:
                logger.debug("Building prompt for report_index=%d", idx)
                prompt = build_extraction_prompt_from_template(
                    template=self.prompt_template,
                    input_text=report.text,
                )
                logger.debug("Running extraction for report_index=%d", idx)
                # Separate generate and parse to capture raw output on error
                raw = self.adapter.generate(prompt)
                # Use decorated prompt (with decoder constraints) if available, otherwise use original
                prompt_for_logging = self.adapter.get_last_decorated_prompt() or prompt
                try:
                    extraction = self.adapter.parse(raw)
                    extraction.case_id = report.case_id
                    if on_success is not None:
                        on_success(idx, extraction, prompt_for_logging, raw)
                    return (idx, extraction)
                except SchemaValidationError as parse_error:
                    # Schema parse error: capture raw output for debugging
                    if on_error is not None:
                        on_error(idx, raw, str(parse_error))
                    logger.error(
                        "Failed to parse output for report_index=%d: %s",
                        idx,
                        str(parse_error),
                        exc_info=True,
                    )
                    errors[idx] = str(parse_error)
                    return (idx, None)
            except Exception as e:
                logger.error(
                    "Failed to extract report_index=%d: %s",
                    idx,
                    str(e),
                    exc_info=True,
                )
                if on_error is not None:
                    on_error(idx, None, str(e))
                errors[idx] = str(e)
                return (idx, None)

        # Use ThreadPoolExecutor for concurrent processing
        if self.concurrency > 1:
            with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
                futures = {
                    executor.submit(_process_report, (idx, report)): idx
                    for idx, report in enumerate(reports, start=1)
                }
                for completed, future in enumerate(as_completed(futures), start=1):
                    try:
                        idx, extraction = future.result()
                        if extraction is not None:
                            outputs.append(extraction)
                    except Exception as e:
                        idx = futures[future]
                        logger.error("Unexpected error processing report_index=%d: %s", idx, e)
                        errors[idx] = str(e)

                    if completed % max(1, len(reports) // 10) == 0:
                        logger.info(
                            "Processed %d/%d reports",
                            completed,
                            len(reports),
                        )
        else:
            # Sequential processing for concurrency=1
            for idx, report in enumerate(reports, start=1):
                idx_result, extraction = _process_report((idx, report))
                if extraction is not None:
                    outputs.append(extraction)

        if errors:
            logger.warning(
                "Extraction completed with %d errors out of %d reports",
                len(errors),
                len(reports),
            )
        else:
            logger.info("All %d reports extracted successfully", len(reports))

        return outputs

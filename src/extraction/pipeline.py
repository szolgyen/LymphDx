import logging
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
    ):
        self.adapter = adapter
        self.prompt_template_path = prompt_template_path
        self.allowed_diagnoses = allowed_diagnoses
        self.decoder_name = decoder_name
        self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")

    def extract_reports(
        self,
        reports: list[ParsedReport],
        on_success: Callable[[int, BaseModel, str], None] | None = None,
        on_error: Callable[[int, str | None, str], None] | None = None,
    ) -> list[BaseModel]:
        logger.info("Extracting structured outputs for %d reports", len(reports))
        outputs: list[BaseModel] = []
        errors: dict[int, str] = {}

        for idx, report in enumerate(reports, start=1):
            try:
                logger.debug("Building prompt for report_index=%d", idx)
                prompt = build_extraction_prompt_from_template(
                    template=self.prompt_template,
                    input_text=report.text,
                    allowed_diagnoses=self.allowed_diagnoses,
                    include_diagnosis_constraints=False,
                )
                logger.debug("Running extraction for report_index=%d", idx)
                # Separate generate and parse to capture raw output on error
                raw = self.adapter.generate(prompt)
                # Use decorated prompt (with decoder constraints) if available, otherwise use original
                prompt_for_logging = self.adapter.get_last_decorated_prompt() or prompt
                try:
                    extraction = self.adapter.parse(raw)
                    extraction.case_id = report.case_id
                    outputs.append(extraction)
                    if on_success is not None:
                        on_success(idx, extraction, prompt_for_logging)
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
            except Exception as e:
                logger.error(
                    "Failed to extract report_index=%d: %s",
                    idx,
                    str(e),
                    exc_info=True,
                )
                errors[idx] = str(e)

        if errors:
            logger.warning(
                "Extraction completed with %d errors out of %d reports",
                len(errors),
                len(reports),
            )
        else:
            logger.info("All %d reports extracted successfully", len(reports))

        return outputs

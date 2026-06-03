import logging
from pathlib import Path
from typing import Callable
from pydantic import BaseModel

from pathology_llm.inference.adapters.base import BaseModelAdapter
from pathology_llm.preprocessing.data_parsing import ParsedReport
from pathology_llm.prompting.prompt_builder import build_extraction_prompt_from_template


logger = logging.getLogger(__name__)


class ExtractionPipeline:
    def __init__(
        self,
        adapter: BaseModelAdapter,
        prompt_template_path: str,
        allowed_diagnoses: set[str],
        include_diagnosis_constraints: bool = True,
    ):
        self.adapter = adapter
        self.prompt_template_path = prompt_template_path
        self.allowed_diagnoses = allowed_diagnoses
        self.include_diagnosis_constraints = include_diagnosis_constraints
        self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")

    def extract_reports(
        self,
        reports: list[ParsedReport],
        on_success: Callable[[int, BaseModel, str], None] | None = None,
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
                    include_diagnosis_constraints=self.include_diagnosis_constraints,
                )
                logger.debug("Running extraction for report_index=%d", idx)
                extraction = self.adapter.extract(prompt)
                extraction.case_id = report.case_id
                outputs.append(extraction)
                if on_success is not None:
                    on_success(idx, extraction, prompt)
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

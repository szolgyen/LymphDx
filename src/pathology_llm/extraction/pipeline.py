import logging
from pathlib import Path

from pathology_llm.inference.base import BaseModelAdapter
from pathology_llm.prompting.prompt_builder import build_extraction_prompt_from_template
from pathology_llm.schemas.pathology import PathologyExtraction


logger = logging.getLogger(__name__)


class ExtractionPipeline:
    def __init__(
        self,
        adapter: BaseModelAdapter,
        prompt_template_path: str,
        allowed_diagnoses: set[str],
    ):
        self.adapter = adapter
        self.prompt_template_path = prompt_template_path
        self.allowed_diagnoses = allowed_diagnoses
        self.prompt_template = Path(prompt_template_path).read_text(encoding="utf-8")

    def extract_reports(self, reports: list[str]) -> list[PathologyExtraction]:
        logger.info("Extracting structured outputs for %d reports", len(reports))
        outputs: list[PathologyExtraction] = []
        errors: dict[int, str] = {}

        for idx, report_text in enumerate(reports, start=1):
            try:
                logger.debug("Building prompt for report_index=%d", idx)
                prompt = build_extraction_prompt_from_template(
                    template=self.prompt_template,
                    input_text=report_text,
                    allowed_diagnoses=self.allowed_diagnoses,
                )
                logger.debug("Running extraction for report_index=%d", idx)
                outputs.append(self.adapter.extract(prompt))
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

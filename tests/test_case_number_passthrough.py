from pathlib import Path

from pathology_llm.extraction.pipeline import ExtractionPipeline
from pathology_llm.inference.adapters.base import BaseModelAdapter
from pathology_llm.preprocessing.data_parsing import ParsedReport
from pathology_llm.schemas.pathology import PathologyExtraction
from pathology_llm.utils.utils import load_reports


class _StubAdapter(BaseModelAdapter):
    def __init__(self):
        super().__init__(allowed_diagnoses={"Adenocarcinoma"})
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def parse(self, raw: str) -> PathologyExtraction:
        raise NotImplementedError

    def extract(self, prompt: str) -> PathologyExtraction:
        self.last_prompt = prompt
        return PathologyExtraction(primary_diagnosis="Adenocarcinoma")


def test_pipeline_attaches_case_number_without_prompt_injection(tmp_path: Path) -> None:
    template_path = tmp_path / "prompt.txt"
    template_path.write_text("Report: {input_text}", encoding="utf-8")

    adapter = _StubAdapter()
    pipeline = ExtractionPipeline(
        adapter=adapter,
        prompt_template_path=str(template_path),
        allowed_diagnoses={"Adenocarcinoma"},
    )

    reports = [ParsedReport(case_id=123, text="Final diagnosis text")]
    outputs = pipeline.extract_reports(reports)

    assert len(outputs) == 1
    assert outputs[0].case_id == 123
    assert outputs[0].primary_diagnosis == "Adenocarcinoma"
    assert adapter.last_prompt is not None
    assert "123" not in adapter.last_prompt
    assert "Final diagnosis text" in adapter.last_prompt


def test_load_reports_txt_sets_number_none(tmp_path: Path) -> None:
    input_file = tmp_path / "reports.txt"
    input_file.write_text("line one\n\nline two\n", encoding="utf-8")

    reports = load_reports(str(input_file))

    assert len(reports) == 2
    assert reports[0].case_id is None
    assert reports[0].text == "line one"
    assert reports[1].case_id is None
    assert reports[1].text == "line two"

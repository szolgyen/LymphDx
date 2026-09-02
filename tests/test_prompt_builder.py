from pathlib import Path

import pytest

from prompting.prompt_builder import (
    build_extraction_prompt,
    build_extraction_prompt_from_template,
)


def test_build_extraction_prompt_from_template_replaces_input_text() -> None:
    prompt = build_extraction_prompt_from_template(
        template="Report: {input_text}",
        input_text="Final diagnosis text",
    )

    assert prompt == "Report: Final diagnosis text"


def test_build_extraction_prompt_requires_input_text_placeholder(tmp_path: Path) -> None:
    template_path = tmp_path / "prompt.txt"
    template_path.write_text("Report", encoding="utf-8")

    with pytest.raises(ValueError, match="Prompt template must include '\\{input_text\\}' placeholder"):
        build_extraction_prompt(
            template_path=str(template_path),
            input_text="Final diagnosis text",
        )

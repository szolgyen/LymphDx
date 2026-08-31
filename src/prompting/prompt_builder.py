from pathlib import Path


def build_extraction_prompt_from_template(
    template: str,
    input_text: str,
    allowed_diagnoses: set[str],
    include_diagnosis_constraints: bool = True,
) -> str:
    if include_diagnosis_constraints and not allowed_diagnoses:
        raise ValueError("allowed_diagnoses cannot be empty")

    if "{input_text}" not in template:
        raise ValueError("Prompt template must include '{input_text}' placeholder")

    prompt = template.replace("{input_text}", input_text)

    return prompt


def build_extraction_prompt(
    template_path: str,
    input_text: str,
    allowed_diagnoses: set[str],
    include_diagnosis_constraints: bool = True,
) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    return build_extraction_prompt_from_template(
        template=template,
        input_text=input_text,
        allowed_diagnoses=allowed_diagnoses,
        include_diagnosis_constraints=include_diagnosis_constraints,
    )

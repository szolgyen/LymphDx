from pathlib import Path


def build_extraction_prompt_from_template(
    template: str,
    input_text: str,
    allowed_diagnoses: set[str],
) -> str:
    if not allowed_diagnoses:
        raise ValueError("allowed_diagnoses cannot be empty")

    if "{input_text}" not in template:
        raise ValueError("Prompt template must include '{input_text}' placeholder")

    diagnoses_block = "\n".join(f"- {item}" for item in sorted(allowed_diagnoses))

    constrained_suffix = (
        "\n\nDIAGNOSIS CONSTRAINTS:\n"
        "- diagnosis_primary and diagnosis_secondary values must be selected exactly "
        "from the following list.\n"
        "- Do not output synonyms, reformulations, or ontology mappings.\n"
        f"{diagnoses_block}\n"
    )

    return template.replace("{input_text}", input_text) + constrained_suffix


def build_extraction_prompt(
    template_path: str,
    input_text: str,
    allowed_diagnoses: set[str],
) -> str:
    template = Path(template_path).read_text(encoding="utf-8")
    return build_extraction_prompt_from_template(
        template=template,
        input_text=input_text,
        allowed_diagnoses=allowed_diagnoses,
    )

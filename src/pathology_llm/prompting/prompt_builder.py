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
    if not include_diagnosis_constraints:
        return prompt

    diagnoses_block = "\n".join(f"- {item}" for item in sorted(allowed_diagnoses))

    constraints_block = (
        "\n\nDIAGNOSIS CONSTRAINTS:\n"
        "Use only terms from the allowed diagnosis list below.\n"
        "Do not output synonyms, reformulations, or ontology mappings.\n"
        "\nALLOWED DIAGNOSIS TERMS:\n"
        f"{diagnoses_block}\n"
    )

    marker = "\nJSON OUTPUT:"
    if marker in prompt:
        head, tail = prompt.rsplit(marker, 1)
        return head + constraints_block + marker + tail

    return prompt + constraints_block


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

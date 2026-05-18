from pathology_llm.schemas.validation import validate_pathology_output

prompt = open("configs/prompts/extraction_v1.txt").read()

fake_llm_output = {
    "schema_version": "v1",
    "diagnosis_primary": "Adenocarcinoma",
    "diagnosis_secondary": [],
    "confidence": 0.9
}

validated = validate_pathology_output(fake_llm_output)
print(validated)
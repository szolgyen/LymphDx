from pathology_llm.schemas.validation import validate_pathology_output

raw = {
    "schema_version": "v1",
    "diagnosis_primary": "Adenocarcinoma",
    "diagnosis_secondary": ["Invasive component"],
    "specimen": "Colon biopsy",
    "is_lymph_node": False,
    "anatomic_location": "Colon",
    "biomarkers": [{"name": "KRAS", "value": "mutated"}],
    "confidence": 0.82
}

obj = validate_pathology_output(raw)
print(obj)
print(obj.model_dump())
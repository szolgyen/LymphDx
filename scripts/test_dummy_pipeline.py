from pathlib import Path
from pathology_llm.inference.dummy_adapter import DummyAdapter


def load_prompt():
    return Path("configs/prompts/extraction_v1.txt").read_text()


adapter = DummyAdapter()

prompt = load_prompt().replace("{input_text}", "Colon biopsy shows adenocarcinoma.")

result = adapter.extract(prompt)

print(result)
print(result.model_dump())
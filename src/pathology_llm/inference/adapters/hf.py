from pathology_llm.inference.adapters.base import PlaceholderAdapter


class HFAdapter(PlaceholderAdapter):
    """HuggingFace Transformers backend adapter (placeholder)."""

    backend_name = "hf"

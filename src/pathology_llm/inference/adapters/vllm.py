from pathology_llm.inference.adapters.placeholder import PlaceholderAdapter


class VLLMAdapter(PlaceholderAdapter):
    """vLLM backend adapter (placeholder)."""

    backend_name = "vllm"

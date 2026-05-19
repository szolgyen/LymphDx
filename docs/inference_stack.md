# Inference Stack

## Adapter Layer

Location: `src/pathology_llm/inference/adapters/`

Implemented:

- `dummy.py`: deterministic mock output for integration/testing.
- `hf.py`: HuggingFace runtime adapter with lazy model loading and parse validation.

Placeholders:

- `vllm.py`
- `sglang.py`
- `ollama.py`

Factory:

- `factory.py` resolves backend + decoder compatibility and constructs adapters.

## Decoder Layer

Location: `src/pathology_llm/inference/decoders/`

Implemented:

- `none.py`: best-effort generation path using prompt formatting only.
- `hf_guidance.py`: strict Guidance-backed JSON-schema constrained generation.

Placeholders:

- `outlines.py`
- `sglang.py`
- shared placeholder behavior in `placeholder.py`

Factory:

- `factory.py` dispatches to the appropriate decoder object.

## Current HF Decoder Modes

### `decoder=none`

- Uses normal model generation.
- Relies on prompt formatting and downstream validation.

### `decoder=guidance`

- Uses Guidance constrained generation path.
- Builds a strict JSON schema including diagnosis enum constraints.
- Enforces non-empty allowed diagnosis terms.

## Validation Contract

After decoding, all outputs still pass through `schemas/validation.py`:

- JSON parse validation.
- Pydantic schema validation.
- Diagnosis set enforcement for primary/secondary diagnoses.

This provides defense in depth even with constrained decoding enabled.

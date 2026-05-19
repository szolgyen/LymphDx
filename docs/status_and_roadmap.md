# Status and Roadmap

## Current Status

Implemented and working:

- End-to-end CLI pipeline from text reports to structured JSON outputs.
- Prompt templating with diagnosis constraints injection.
- Strict schema and diagnosis validation.
- Inference architecture split into parallel `adapters/` and `decoders/` packages.
- HF adapter with:
  - `decoder=none`
  - `decoder=guidance` (strict constrained JSON generation path)
- Decoder placeholders for `outlines` and `sglang`.
- Backend placeholders for `vllm`, `sglang`, and `ollama` adapters.
- Test suite passing.

## What Is Implemented Already (Detailed)

- Adapter factory compatibility logic and auto decoder mapping.
- Guidance constrained decoding integration for HF.
- Per-report pipeline error handling and non-zero failure when all reports fail.
- Reproducible environment lock and dependency tracking.

## What Is Ahead

### Priority 1: Performance and Throughput

- Reduce strict decoding latency (token limits, prompt/schema sizing, wrapper reuse optimizations).
- Add report-level performance metrics output.

### Priority 2: Additional Decoder Implementations

- Implement real `OutlinesDecoder`.
- Implement real `SGLangDecoder`.
- Expand decoder compatibility matrix by backend.

### Priority 3: Additional Backend Implementations

- Implement `VLLMAdapter` runtime integration.
- Implement `SGLangAdapter` runtime integration.
- Implement `OllamaAdapter` runtime integration.

### Priority 4: Quality and Evaluation

- Add dataset-level evaluation metrics under `src/pathology_llm/evaluation/`.
- Add regression fixtures for difficult pathology report patterns.

## Suggested Next Milestone

Implement real `OutlinesDecoder` for a second constrained decoding path and connect it to the first non-HF production backend (likely vLLM). This validates cross-backend decoder abstraction early.

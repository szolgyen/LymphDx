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
- Outlines decoder implementation is available.
- SGLang decoder remains a placeholder.
- Backend placeholders for `vllm`, `sglang`, and `ollama` adapters.
- Test suite passing.

## What Is Implemented Already (Detailed)

- Adapter factory compatibility logic and auto decoder mapping.
- Guidance constrained decoding integration for HF.
- Per-report pipeline error handling and non-zero failure when all reports fail.
- Reproducible environment lock and dependency tracking.

Current practical constraint from recent backend trials:

- Cross-backend portability is bounded by a compatibility matrix (model architecture support, CUDA/driver stack, engine-specific kernels), not just adapter wiring.
- As of current host tests, HF remains the reliable production path for MedGemma/OpenBioLLM.
- In the current repository state, non-HF backends are represented by placeholder adapters and remain implementation targets.

## What Is Ahead

### Priority 1: Performance and Throughput

- Reduce strict decoding latency (token limits, prompt/schema sizing, wrapper reuse optimizations).
- Add report-level performance metrics output.

### Priority 2: Additional Decoder Implementations

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

Implement the first non-HF production backend adapter (likely `vllm`), then pair it with the existing `outlines` decoder path and validate against a constrained compatibility matrix.

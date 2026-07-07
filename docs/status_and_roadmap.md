# Status and Roadmap

## Current Status

Implemented and working:

- End-to-end YAML config-driven pipeline from text reports to structured JSON outputs with timestamped results.
- Prompt templating with diagnosis constraints injection.
- Strict schema and diagnosis validation via duck-typed schema-agnostic validator.
- Inference architecture split into parallel `adapters/` and `decoders/` packages.
- HF adapter with:
  - `decoder=none`
  - `decoder=guidance` (strict constrained JSON generation path)
  - `decoder=outlines` (strict constrained decoding path)
- Outlines decoder implementation is available.
- SGLang decoder remains a placeholder.
- Backend placeholders for `vllm`, `sglang`, and `ollama` adapters.
- Test suite passing with schema-agnostic validation.
- Top-N ontology matching with nested prediction structures.

## What Is Implemented Already (Detailed)

- Adapter factory with explicit backend/decoder validation (no auto-resolution).
- Guidance and Outlines constrained decoding integration for HF.
- Per-report pipeline error handling and non-zero failure when all reports fail.
- Reproducible environment lock and dependency tracking with Python 3.12+ requirement.
- Timestamped output organization with YYYYMMDD_HHMMSS format.
- Schema-agnostic validation using duck-typing for cross-version compatibility.

Current practical constraint from recent backend trials:

- Cross-backend portability is bounded by a compatibility matrix (model architecture support, CUDA/driver stack, engine-specific kernels), not just adapter wiring.
- As of current host tests, HF remains the reliable production path for MedGemma/OpenBioLLM.
- In the current repository state, non-HF backends are represented by placeholder adapters and remain implementation targets.
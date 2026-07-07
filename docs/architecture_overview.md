# Architecture Overview

## Goal

Extract structured pathology JSON objects from free-text reports with strict schema validation and diagnosis constraints.

## Top-Level Design

The framework is organized by processing responsibilities:

- `scripts/`
  - Entrypoints and execution scripts.
- `src/extraction/`
  - Pipeline orchestration over report batches.
- `src/inference/adapters/`
  - Backend runtime integration (HF, placeholders).
- `src/inference/decoders/`
  - Decoding strategies (`none`, `guidance`, `outlines`, placeholders).
- `src/prompting/`
  - Prompt template construction and diagnosis constraints injection.
- `src/schemas/`
  - Structured schema and strict output validation.
- `src/utils/`
  - Logging and file I/O utilities.

## Core Principles

- Strict schema-first extraction.
- Diagnosis constraints are enforced via constrained decoding, not mapped post hoc.
- Adapter runtime concerns are separated from decoder strategy concerns.
- Placeholder scaffolding is explicit for unimplemented backends/decoders (currently `vllm`, `sglang`, `ollama` adapters and `sglang` decoder).

## Parallel Inference Structure

Inference follows a parallel package layout:

- `src/inference/adapters/`
  - `base.py`, `factory.py`, concrete adapters (hf, vllm, sglang, ollama).
- `src/inference/decoders/`
  - `base.py`, `factory.py`, concrete/placeholder decoders (none, guidance, outlines, sglang).

This keeps runtime backend logic and decoding policy cleanly separated.

# Architecture Overview

## Goal

Extract structured pathology JSON objects from free-text reports with strict schema validation.

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

- Schema-first extraction with enforced JSON validity via constrained decoding.
- Flexible diagnosis extraction supporting unconstrained (`none`) and constrained (`guidance`, `outlines`) decoding strategies.
- Post hoc ontology mapping support for unconstrained diagnosis outputs.
- Clear separation between adapter runtime logic and decoder strategy.
- Explicit placeholders for unimplemented backends/decoders (`vllm`, `sglang`, `ollama` adapters; `sglang` decoder).

## Parallel Inference Structure

Inference follows a parallel package layout:

- `src/inference/adapters/`
  - `base.py`, `factory.py`, concrete adapters (hf, vllm, sglang, ollama).
- `src/inference/decoders/`
  - `base.py`, `factory.py`, concrete/placeholder decoders (none, guidance, outlines, sglang).

This keeps runtime backend logic and decoding policy cleanly separated.

# Architecture Overview

## Goal

Extract structured pathology JSON objects from free-text reports with strict schema validation and diagnosis constraints.

## Top-Level Design

The framework is organized by processing responsibilities:

- `scripts/`
  - Entrypoints and execution scripts.
- `src/pathology_llm/extraction/`
  - Pipeline orchestration over report batches.
- `src/pathology_llm/inference/adapters/`
  - Backend runtime integration (HF, dummy, placeholders).
- `src/pathology_llm/inference/decoders/`
  - Decoding strategies (`none`, `guidance`, placeholders).
- `src/pathology_llm/prompting/`
  - Prompt template construction and diagnosis constraints injection.
- `src/pathology_llm/schemas/`
  - Structured schema and strict output validation.
- `src/pathology_llm/utils/`
  - Logging and file I/O utilities.

## Core Principles

- Strict schema-first extraction.
- Diagnosis constraints are enforced, not mapped post hoc.
- Adapter runtime concerns are separated from decoder strategy concerns.
- Placeholder scaffolding is explicit for unimplemented backends/decoders.

## Parallel Inference Structure

Inference now follows a parallel package layout:

- `src/pathology_llm/inference/adapters/`
  - `base.py`, `factory.py`, concrete adapters.
- `src/pathology_llm/inference/decoders/`
  - `base.py`, `factory.py`, concrete/placeholder decoders.

This keeps runtime backend logic and decoding policy cleanly separated.

# Hematopathology LLM

Structured pathology report extraction with modular LLM backends.

## Quick Start

For first-time setup from a fresh clone, use the installation guide:

- [docs/installation.md](docs/installation.md)

Shortest path:

```bash
make bootstrap
```

Then run the HF pipeline with the configured backend:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/heme-llm-hf/bin/python scripts/run_pipeline.py --config configs/pipeline/run_pipeline.yaml
```

Edit `configs/pipeline/run_pipeline.yaml` to customize model, decoder, input/output paths.
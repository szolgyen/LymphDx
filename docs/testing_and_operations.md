# Operations

## Common Runtime Commands

Edit `configs/pipeline/run_pipeline.yaml` to set backend, model, decoder, and file paths.

Then run:

```bash
CUDA_VISIBLE_DEVICES=0 .venvs/heme-llm-hf/bin/python scripts/run_pipeline.py --config configs/pipeline/run_pipeline.yaml
```

Example configurations:

**HF + Guidance:**

```yaml
backend: hf
model: google/medgemma-4b-it
decoder: guidance
input_file: configs/extraction/sample_reports.txt
diagnosis_terms_file: configs/extraction/diagnosis_terms_v1.txt
log_level: INFO
```

**HF + Outlines:**

```yaml
backend: hf
model: google/medgemma-4b-it
decoder: outlines
input_file: configs/extraction/sample_reports.txt
diagnosis_terms_file: configs/extraction/diagnosis_terms_v1.txt
log_level: INFO
```

## Observed HF Model/Decoder Matrix

Observed from local runs in this repository (May 2026):

| Model | `guidance` | `outlines` | Notes |
|---|---|---|---|
| `google/medgemma-4b-it` | Works | Works | Successful end-to-end extraction run observed. |
| `google/medgemma-1.5-4b-it` | Works | Works | `decoder=none` failed (non-JSON outputs), constrained decoders worked. |
| `aaditya/Llama3-OpenBioLLM-8B` | Works | Works | Successful constrained extraction observed for both decoders. |

Notes:

- This is an observed compatibility snapshot, not a strict support matrix.
- Results can vary by GPU topology, CUDA driver, and dependency versions.
- For HF backend runs, pinning to a single GPU (`CUDA_VISIBLE_DEVICES=0`) is the default documented path.

## Observed Backend Compatibility Issues (May 2026)

These failures were reproduced during local integration attempts on the current host stack
(NVIDIA L40S, driver 550.54.15, CUDA 12.4).

Context:

- These are historical external backend integration attempts.
- In the current repository state, non-HF adapters (`vllm`, `sglang`, `ollama`) are placeholders.

| Backend / Runtime | Model | Status | Observed Failure |
|---|---|---|---|
| SGLang (`sglang==0.5.9`) | `google/medgemma-4b-it` | Not working | Runtime segfaults during Triton/FlashInfer execution path (scheduler/worker crash, process exits). |
| vLLM (`vllm==0.21.0`) | `google/medgemma-4b-it` | Not working | CUDA driver/runtime mismatch on this host (`driver too old` for selected wheel stack). |
| vLLM (`vllm==0.6.6.post1` + cu124 pinned torch) | `google/medgemma-4b-it` | Not working | Model architecture not supported by this vLLM build (`Gemma3ForConditionalGeneration`). |
| vLLM (`vllm==0.6.6.post1` + cu124 pinned torch) | `google/gemma-2-2b-it` | Not working (in current custom image) | CUDA custom op mismatch (`_C::rotary_embedding`/operator backend errors), server exits before health ready. |
| vLLM (`vllm==0.6.6.post1` + cu124 pinned torch) | `aaditya/Llama3-OpenBioLLM-8B` | Not working (in current custom image) | Multiprocessing/runtime instability (`ZMQError`, then CUDA op errors such as `_C::rms_norm`) in this tested stack. |

Working baseline from these experiments:

- HF backend remains the stable path for MedGemma and OpenBioLLM on this host.

Operational recommendation:

- Treat backend/model support as an explicit compatibility matrix.
- Keep non-HF backend integrations as opt-in experimental until a tested matrix row is marked working on this host.

## Logging

- Console + file logging are enabled.
- Per-run log files: `outputs/logs/run_pipeline_YYYYMMDD_HHMMSS.log`
- Guidance mode logs start/finish timing per report.

## Environment Reproducibility

Tracked dependency sources:

- `pyproject.toml`
- `requirements/base.txt`
- `requirements/hf/requirements.txt`
- `requirements/vllm/requirements.txt`
- `requirements/sglang/requirements.txt`
- `requirements/ollama/requirements.txt`
- `uv.lock`
- `requirements/hf/lock.txt`
- `requirements/vllm/lock.txt`
- `requirements/sglang/lock.txt`
- `requirements/ollama/lock.txt`

Locking model:

- `uv.lock` is the single lock file for dependencies declared in `pyproject.toml`.
- Backend-specific environments use backend requirements files plus backend lock files.
- Regenerate backend locks with `uv pip compile requirements/<backend>/requirements.txt -o requirements/<backend>/lock.txt`.

Notes:

- CUDA-compatible Torch is pinned for the current environment constraints.
- `gpustat` is tracked for optional Guidance GPU monitoring metrics.

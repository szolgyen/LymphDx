#!/bin/bash
set -euo pipefail

# Check if the script is running in a HPC environment
if command -v module >/dev/null 2>&1; then
    echo "Loading CUDA module..."
    module load cuda/12.4.0-gcc-12.4.0
fi

echo "Activating virtual environment..."
source .venvs/heme-llm-vllm/bin/activate

MODEL=$(awk '/^model:/ {print $2; exit}' configs/llm_pipeline.yaml)

echo "Starting vLLM API server with model: $MODEL"
exec python -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --gpu-memory-utilization 0.9
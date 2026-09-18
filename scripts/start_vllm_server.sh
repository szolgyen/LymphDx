#!/bin/bash
set -euo pipefail

# Check if the script is running in a HPC environment
if command -v module >/dev/null 2>&1; then
    echo "Loading CUDA module..."
    module load cuda/12.4.0-gcc-12.4.0
fi

echo "Activating virtual environment..."
source .venvs/heme-llm-vllm/bin/activate

# Read configuration from llm_pipeline.yaml
CONFIG_FILE="${1:-configs/llm_pipeline.yaml}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

MODEL=$(awk '/^model:/ {print $2; exit}' "$CONFIG_FILE")
BACKEND=$(awk '/^backend:/ {print $2; exit}' "$CONFIG_FILE")
DECODER=$(awk '/^decoder:/ {print $2; exit}' "$CONFIG_FILE")

echo "Configuration detected:"
echo "  Backend: $BACKEND"
echo "  Decoder: $DECODER"
echo "  Model: $MODEL"
echo ""

# Build vLLM command
VLLM_CMD="vllm serve $MODEL \
    --gpu-memory-utilization 0.9"

# Add constraint backend depending on the decoder configuration
if [ "$DECODER" = "xgrammar" ]; then
    echo "Starting vLLM with kernel-optimized schema constraint backend (xgrammar)..."
    VLLM_CMD="$VLLM_CMD --structured-outputs-config.backend xgrammar"
fi

echo "Running: $VLLM_CMD"
echo ""
exec $VLLM_CMD
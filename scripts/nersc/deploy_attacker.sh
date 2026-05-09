#!/usr/bin/env bash
set -euo pipefail

# Default model if not provided
MODEL_NAME="${1:-Qwen/Qwen2.5-32B-Instruct}"

eval "$(conda shell.bash hook)"
conda activate vllm

export HF_HOME="$SCRATCH/LLMs"

ip=$(hostname -I | awk '{print $1}')
echo "Server IP address: $ip"

NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

echo "Using model: $MODEL_NAME"
echo "Using $NUM_GPUS GPUs"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --tensor-parallel-size "$NUM_GPUS" \
  --max-model-len 32768 \
  --port 30000
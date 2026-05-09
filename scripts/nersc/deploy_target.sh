#!/usr/bin/env bash
set -euo pipefail

# Require model name as first argument
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

MODEL_NAME="$1"

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
  --max-model-len 8192 \
  --port 30000
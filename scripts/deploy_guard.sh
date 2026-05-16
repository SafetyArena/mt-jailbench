#!/bin/bash
set -a

MODEL_PATH="GraySwanAI/Llama-3-8B-Instruct-RR" # meta-llama/Llama-Guard-3-8B | GraySwanAI/Llama-3-8B-Instruct-RR | Qwen/Qwen3Guard-Gen-8B | allenai/wildguard
PORT="8080"
NODE="0.0.0.0"
MODEL_NAME="guard"
BACKGROUND=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -m, --model MODEL_PATH          Model path/tag (default: $MODEL_PATH)"
    echo "  --served-model-name NAME        Served model name (default: $MODEL_NAME)"
    echo "  --port PORT                     Server port (default: $PORT)"
    echo "  --background                    Start server in background (default)"
    echo ""
    echo "Examples:"
    echo "  bash $0 --port 8080"
    echo "  bash $0 -m meta-llama/Llama-Guard-3-8B --served-model-name guard --port 8080"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        --background)
            BACKGROUND=true
            shift
            ;;
        --served-model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --node)
            NODE="$2"
            shift 2
            ;;
        --no-wait)
            NO_WAIT=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

SERVER_URL="http://${NODE}:${PORT}"

mkdir -p "logs"

if curl -s "$SERVER_URL/health" > /dev/null 2>&1 || curl -s "$SERVER_URL/v1/models" > /dev/null 2>&1; then
    echo "✓ vLLM guard server is already running on $NODE:$PORT"
    echo "  Logs: logs/guard_server.log"
    exit 0
fi

# Tensor parallel must match visible devices (CUDA_VISIBLE_DEVICES), not the whole node.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    _cvd="${CUDA_VISIBLE_DEVICES// /}"
    _commas="${_cvd//[^,]/}"
    NUM_GPUS=$(( ${#_commas} + 1 ))
elif command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=1
fi
echo "Detected $NUM_GPUS visible GPU(s), setting tensor parallelism to $NUM_GPUS"

# Run vLLM server
echo "Starting vLLM guard server with model: $MODEL_PATH"
echo "  Host: $NODE"
echo "  Port: $PORT"
echo "  Served model name: $MODEL_NAME"
echo "  TP: $NUM_GPUS"

VLLM_ARGS=(
  serve "$MODEL_PATH"
  --host "$NODE"
  --port "$PORT"
  --served-model-name "$MODEL_NAME"
  --trust-remote-code
  --dtype "bfloat16"
  --tensor-parallel-size "$NUM_GPUS"
  --seed "0"
  --disable-uvicorn-access-log
)

if [ "$BACKGROUND" = true ]; then
  echo "Starting in background on $SERVER_URL"
  echo "  Logs: logs/guard_server.log"

  pkill -f "vllm serve .*--port ${PORT}" 2>/dev/null || true
  pkill -f "uvicorn.*:${PORT}" 2>/dev/null || true
  sleep 2

  nohup vllm "${VLLM_ARGS[@]}" > "logs/guard_server.log" 2>&1 &
  SERVER_PID="$!"
  echo "  PID: $SERVER_PID"

  if [ -z "${NO_WAIT:-}" ]; then
    echo "Waiting for server to initialize (this may take several minutes)..."
    WAITED=0
    while true; do
      if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
        echo "❌ Server process died. Check logs:"
        echo "  tail -40 logs/guard_server.log"
        exit 1
      fi

      if curl -s "$SERVER_URL/health" > /dev/null 2>&1 || curl -s "$SERVER_URL/v1/models" > /dev/null 2>&1; then
        echo ""
        echo "✓ Server is ready and responding! (waited ${WAITED}s)"
        break
      fi

      if [ $((WAITED % 10)) -eq 0 ] && [ "$WAITED" -gt 0 ]; then
        echo -n " (${WAITED}s)"
      else
        echo -n "."
      fi

      sleep 2
      WAITED=$((WAITED + 2))
    done
    echo ""
  else
    echo "Skipping wait (--no-wait)"
  fi
else
  vllm "${VLLM_ARGS[@]}"
fi
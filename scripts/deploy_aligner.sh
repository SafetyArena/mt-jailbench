#!/bin/bash
set -a

MODEL_PATH="aligner/aligner-7b-v1.0"
PORT="8081"
NODE="0.0.0.0"
MODEL_NAME="aligner"
BACKGROUND=true

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "  -m, --model MODEL_PATH          Model path/tag (default: $MODEL_PATH)"
    echo "  --served-model-name NAME        Served model name (default: $MODEL_NAME)"
    echo "  --port PORT                     Server port (default: $PORT)"
    echo "  --node NODE                     Bind address (default: $NODE)"
    echo "  --background                    Start in background (default)"
    echo "  --no-wait                       Do not wait for server readiness"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_PATH="$2"
            shift 2
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
        --background)
            BACKGROUND=true
            shift
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
mkdir -p logs

if curl -s "$SERVER_URL/health" >/dev/null 2>&1 || curl -s "$SERVER_URL/v1/models" >/dev/null 2>&1; then
    echo "Aligner server is already running on $NODE:$PORT"
    exit 0
fi

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    visible_devices="${CUDA_VISIBLE_DEVICES// /}"
    commas="${visible_devices//[^,]/}"
    NUM_GPUS=$(( ${#commas} + 1 ))
elif command -v nvidia-smi >/dev/null 2>&1; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=1
fi

VLLM_ARGS=(
    serve "$MODEL_PATH"
    --host "$NODE"
    --port "$PORT"
    --served-model-name "$MODEL_NAME"
    --trust-remote-code
    --dtype bfloat16
    --tensor-parallel-size "$NUM_GPUS"
    --seed 0
    --disable-uvicorn-access-log
)

if [[ "$BACKGROUND" != true ]]; then
    vllm "${VLLM_ARGS[@]}"
    exit $?
fi

nohup vllm "${VLLM_ARGS[@]}" >logs/aligner_server.log 2>&1 &
SERVER_PID=$!
echo "Started aligner server with PID $SERVER_PID; logs: logs/aligner_server.log"

if [[ -n "${NO_WAIT:-}" ]]; then
    exit 0
fi

while true; do
    if ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
        echo "Aligner server exited; inspect logs/aligner_server.log"
        exit 1
    fi
    if curl -s "$SERVER_URL/health" >/dev/null 2>&1 || curl -s "$SERVER_URL/v1/models" >/dev/null 2>&1; then
        echo "Aligner server is ready on $NODE:$PORT"
        exit 0
    fi
    sleep 2
done

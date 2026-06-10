#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_PATH="$SCRIPT_DIR/model_int8.onnx"

if [ -f "$MODEL_PATH" ]; then
    echo "Model already exists at $MODEL_PATH"
    exit 0
fi

URL="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_int8.onnx"
echo "Downloading nomic-embed-text-v1.5 INT8 ONNX model (137MB)..."
curl -L -o "$MODEL_PATH" "$URL"
echo "Downloaded to $MODEL_PATH"
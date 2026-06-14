#!/bin/bash

HOST="localhost"
PORT="12434"
MODEL="mlx-community/gemma-4-12B-it-6bit"

echo "=========================================================="
echo " Starting MLX-VLM Server"
echo " Model: $MODEL"
echo " URL:   http://$HOST:$PORT"
echo "=========================================================="

SCRIPT_DIR="/Users/asantiola/workspaces/playground-ai-ml/mlx"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"
SERVER=mlx_vlm.server
MODEL="mlx-community/gemma-4-12B-it-6bit"

python -m "$SERVER" --host "$HOST"  --port "$PORT" --model "$MODEL"

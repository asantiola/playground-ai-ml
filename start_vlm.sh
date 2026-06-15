#!/bin/bash

HOST="localhost"
PORT="12434"
SERVER=mlx_vlm.server
MODEL="mlx-community/gemma-4-12B-it-6bit"

echo "=========================================================="
echo " Starting $SERVER"
echo " Model: $MODEL"
echo " URL:   http://$HOST:$PORT"
echo "=========================================================="

SCRIPT_DIR="/Users/asantiola/workspaces/playground-ai-ml/mlx"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"

source "$VENV_DIR/bin/activate"
python -m "$SERVER" --host "$HOST"  --port "$PORT" --model "$MODEL"

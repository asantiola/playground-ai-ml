#!/bin/bash

HOST="localhost"
PORT="12434"
SERVER="mlx_lm server"
MODEL="mlx-community/phi-4-6bit"
MAX_TOKENS=8192

echo "=========================================================="
echo " Starting $SERVER"
echo " Model: $MODEL"
echo " URL:   http://$HOST:$PORT"
echo "=========================================================="

VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"

source "$VENV_DIR/bin/activate"
python -m $SERVER --host "$HOST" --port "$PORT" --model "$MODEL" --max-tokens $MAX_TOKENS

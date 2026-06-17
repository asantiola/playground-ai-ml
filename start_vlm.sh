#!/bin/bash

HOST="localhost"
PORT="12434"
SERVER=mlx_vlm.server
MAX_KV_SIZE=8192

echo "=========================================================="
echo " Starting $SERVER"
echo " URL:   http://$HOST:$PORT"
echo "=========================================================="

VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"

source "$VENV_DIR/bin/activate"
python -m "$SERVER" --host "$HOST" --port "$PORT" --max-kv-size $MAX_KV_SIZE

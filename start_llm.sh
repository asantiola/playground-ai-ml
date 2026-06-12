#!/bin/bash

# --- Configuration ---
HOST="localhost"
PORT="12434"
MODEL="mlx-community/gemma-4-12B-it-6bit"

echo "=========================================================="
echo " Starting MLX-VLM Server"
echo " Model: $MODEL"
echo " URL:   http://$HOST:$PORT"
echo "=========================================================="

# Run the server. Using 'exec' ensures that stopping the script (Ctrl+C) 
# cleanly tears down the underlying Python process instantly.
exec python -m mlx_vlm.server \
    --host "$HOST" \
    --port "$PORT" \
    --model "$MODEL"

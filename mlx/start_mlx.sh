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
PIDFILE="$SCRIPT_DIR/mlx_vlm.pid"
LOGFILE="$SCRIPT_DIR/mlx_vlm.log"
SERVER=mlx_vlm.server
MODEL="mlx-community/gemma-4-12B-it-6bit"

# 1. Check if the server is already running
if [ -f "$PIDFILE" ]; then
    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "mlx_vlm.server is already running with PID $PID."
        exit 0
    else
        rm "$PIDFILE"
    fi
fi

# 2. Activate your virtual environment
source "$VENV_DIR/bin/activate"

# 3. Start the server in the background and log output to your script directory
nohup "$SERVER" --host "$HOST"  --port "$PORT" --model "$MODEL" > "$LOGFILE" 2>&1 &

# 4. Capture and save the PID
echo $! > "$PIDFILE"

echo "mlx_vlm.server started with PID $(cat $PIDFILE)."

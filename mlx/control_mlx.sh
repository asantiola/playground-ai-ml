#!/bin/bash

SCRIPT_DIR="/Users/asantiola/workspaces/playground-ai-ml/mlx"
VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"
PIDFILE="$SCRIPT_DIR/mlx_vlm.pid"
LOGFILE="$SCRIPT_DIR/mlx_vlm.log"

SERVER=mlx_vlm.server
HOST="localhost"
PORT="12434"
MODEL="mlx-community/gemma-4-12B-it-6bit"

# Function to check if running
is_running() {
    [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null
}

case "$1" in
    status)
        if is_running; then
            echo "RUNNING"
        else
            echo "STOPPED"
        fi
        ;;
    start)
        if is_running; then
            echo "Already running."
        else
            source "$VENV_DIR/bin/activate"
            nohup "$SERVER" --host "$HOST"  --port "$PORT" --model "$MODEL" > "$LOGFILE" 2>&1 &
            echo $! > "$PIDFILE"
            echo "Started."
        fi
        ;;
    stop)
        if is_running; then
            PID=$(cat "$PIDFILE")
            kill "$PID"
            rm "$PIDFILE"
            echo "Stopped."
        else
            echo "Not running."
            [ -f "$PIDFILE" ] && rm "$PIDFILE"
        fi
        ;;
esac

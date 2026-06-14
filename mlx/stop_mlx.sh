#!/bin/bash

# Define paths based on your setup
SCRIPT_DIR="/Users/asantiola/workspaces/playground-ai-ml/mlx"
PIDFILE="$SCRIPT_DIR/mlx_vlm.pid"

# 1. Check if the PID file exists
if [ ! -f "$PIDFILE" ]; then
    echo "No PID file found in $SCRIPT_DIR. Server is likely not running."
    exit 0
fi

# 2. Read the PID
PID=$(cat "$PIDFILE")

# 3. Verify the process is running and kill it
if kill -0 "$PID" 2>/dev/null; then
    kill "$PID"
    echo "Stopped mlx_vlm.server (PID: $PID)."
else
    echo "Process $PID wasn't running, cleaning up stale PID file."
fi

# 4. Remove the PID file
rm "$PIDFILE"

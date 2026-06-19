#!/bin/bash

HOST="localhost"
PORT="12434"
SERVER="mlx_lm server"
MODEL="mlx-community/phi-4-6bit"
MAX_TOKENS=8192

PARMS="-m $SERVER"
PARMS+=" --host $HOST"
PARMS+=" --port $PORT"
PARMS+=" --max-tokens $MAX_TOKENS"
PARMS+=" --model $MODEL"

echo "=========================================================="
echo " Starting $SERVER"
echo " Model: $MODEL"
echo " URL:   http://$HOST:$PORT"
echo " CMD: python $PARMS"
echo "=========================================================="

VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"

source "$VENV_DIR/bin/activate"
python $PARMS

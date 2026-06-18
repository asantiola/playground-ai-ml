#!/bin/bash

ARG1=$1

HOST="localhost"
PORT="12434"
SERVER=mlx_vlm.server
MODEL="mlx-community/gemma-4-12B-it-6bit"
MAX_KV_SIZE=16384
DRAFT_MODEL=""
MTP=""

PARMS="-m $SERVER"
PARMS+=" --host $HOST"
PARMS+=" --port $PORT"
PARMS+=" --max-kv-size $MAX_KV_SIZE"
PARMS+=" --model $MODEL"
if [ $ARG1 == "mtp" ]
then
    DRAFT_MODEL="mlx-community/gemma-4-12B-it-assistant-6bit"
    MTP=$ARG1
    PARMS+=" --draft-model $DRAFT_MODEL"
    PARMS+=" --draft-kind mtp"
fi

echo "=========================================================="
echo " Starting $SERVER"
echo " URL: http://$HOST:$PORT"
echo " CMD: python $PARMS"
echo "=========================================================="

VENV_DIR="/Users/asantiola/workspaces/playground-ai-ml/.venv"

source "$VENV_DIR/bin/activate"
python $PARMS

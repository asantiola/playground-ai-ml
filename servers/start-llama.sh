#!/bin/bash

HOST="localhost"
PORT="12434"
SERVER="llama-server"
MODEL="microsoft/phi-4-gguf:IQ4_NL"
# MODEL="Qwen/Qwen2.5-Coder-14B-Instruct-GGUF:Q4_K_M"
CONTEXT_SIZE=16384
CACHE_TYPE_K=q4_0
CACHE_TYPE_V=q4_0
FLASH_ATTN=on

SERVER="llama-server"
PARMS="--parallel 1"

if [ -n "$MODEL" ]
then
    PARMS+=" -hf $MODEL"
fi

if [ -n "$HOST" ]
then
    PARMS+=" --host $HOST"
fi

if [ -n "$PORT" ]
then
    PARMS+=" --port $PORT"
fi

if [ -n "$FLASH_ATTN" ]
then
    PARMS+=" --flash-attn $FLASH_ATTN -t 10 -b 2048 -ub 512"
fi

if [ -n "$CONTEXT_SIZE" ]
then
    PARMS+=" --ctx-size $CONTEXT_SIZE"
fi

if [ -n "$CACHE_TYPE_K" ]
then
    PARMS+=" --cache-type-k $CACHE_TYPE_K"
fi

if [ -n "$CACHE_TYPE_V" ]
then
    PARMS+=" --cache-type-v $CACHE_TYPE_V"
fi

# if [ $ARG1 == "mtp" ]
# then
#     PARMS+=" --spec-type draft-mtp --spec-draft-n-max 3"
# fi

echo "=========================================================="
echo " Starting $SERVER"
echo " URL: http://$HOST:$PORT"
echo " CMD: $SERVER $PARMS"
echo "=========================================================="

$SERVER $PARMS

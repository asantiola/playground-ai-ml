#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
cd "$WORKDIR"

MODEL=${1:-"unsloth/gemma-4-12B-it-qat-GGUF:UD-Q4_K_XL"}

set -x
llama-server -hf $MODEL --host localhost --port 12434 --ctx-size 16384 --gpu-layers all --reasoning off

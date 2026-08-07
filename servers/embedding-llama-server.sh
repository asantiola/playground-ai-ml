#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
cd "$WORKDIR"

MODEL=${1:-"unsloth/embeddinggemma-300m-GGUF:Q8_0"}

set -x
llama-server -hf $MODEL --host localhost --port 12435 --gpu-layers all --embeddings --pooling mean

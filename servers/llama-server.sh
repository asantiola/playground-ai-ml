#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
cd "$WORKDIR"

MODEL=${1:-"Qwen/Qwen2.5-Coder-14B-Instruct-GGUF:Q5_0"}

set -x
llama-server -hf $MODEL --host localhost --port 12434

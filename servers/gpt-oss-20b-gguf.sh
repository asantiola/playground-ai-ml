#!/bin/sh

WORKDIR="/Users/asantiola/workspaces/playground-ai-ml/"
cd "$WORKDIR"

set -x
llama-server -hf ggml-org/gpt-oss-20b-GGUF --host localhost --port 12434 -c 0 --jinja

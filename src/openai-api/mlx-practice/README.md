# Practice using mlx-lm python package
- having issues with prompts?

### Install
```
pip install --upgrade mlx-vlm mlx-lm mlx
```

### Uninstall
```
pip uninstall mlx-vlm mlx-lm mlx
```

### sample CLI
- These works:
```
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "What is the capital of France?"
mlx_lm.generate --model mlx-community/gpt-oss-20b-MXFP4-Q4 --prompt "What is the capital of France?"
mlx_vlm.generate --model --model mlx-community/gemma-4-e4b-it-8bit --prompt "What is the capital of France?"
```

### starting the server
- These works but watch memory pressure for 20b model
```
mlx_lm.server --help
mlx_lm.server  --host localhost --port 12434 --model mlx-community/gpt-oss-20b-MXFP4-Q4
mlx_lm.server  --host localhost --port 12434 --model mlx-community/phi-4-6bit
mlx_vlm.server --host localhost --port 12434 --model mlx-community/gemma-4-e4b-it-8bit
```

### Tried DMR + VLM-METAL + MLX?
- Model below works (mlx-community/Llama-3.2-1B-Instruct-4bit)
- take note /Users/asantiola/.docker/model-runner/vllm-metal/bin/python3 used ~15Gb memory!
```
docker model pull hf.co/mlx-community/Llama-3.2-1B-Instruct-4bit
```
- Tested ai/smollm2-vllm:135M, was also using python3 ~15Gb memory!
- config
```
llm = ChatOpenAI(
    model="hf.co/mlx-community/Llama-3.2-1B-Instruct-4bit",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)
```

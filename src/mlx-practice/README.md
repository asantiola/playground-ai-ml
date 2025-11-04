# Practice using mlx-lm python package
- having issues with prompts?

### Install
```
pip install mlx-lm
```

### sample CLI
```
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "What is the capital of France?"
mlx_lm.generate --model mlx-community/gpt-oss-20b-MXFP4-Q4 --prompt "What is the capital of France?"
```

### starting the server
```
mlx_lm.server --help
mlx_lm.server --port 12434
```

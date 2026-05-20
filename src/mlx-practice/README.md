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
```
mlx_lm.generate --model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit --prompt "What is the capital of France?"
mlx_lm.generate --model mlx-community/gpt-oss-20b-MXFP4-Q4 --prompt "What is the capital of France?"
mlx_vlm.generate --model --model mlx-community/gemma-4-e4b-it-4bit --prompt "What is the capital of France?"
```

### starting the server
```
mlx_lm.server --help
mlx_lm.server --port 12434 --model mlx-community/gpt-oss-20b-MXFP4-Q4
mlx_vlm.server --port 12434 --model mlx-community/gemma-4-e4b-it-4bit
```

### Tried DMR + VLM-METAL + MLX?
- not using for now (20 May 2026)
- config
```
llm = ChatOpenAI(
    model="huggingface.co/mlx-community/gemma-4-e4b-it-4bit",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)
```
- output
```
openai.InternalServerError: unable to load runner: error waiting for runner to be ready: vllm-metal terminated unexpectedly: vllm-metal failed: (APIServer pid=80518)     raise RuntimeError(
(APIServer pid=80518) RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {'EngineCore': 1}
uper().__init__(
(APIServer pid=80518)   File "/Users/asantiola/.docker/model-runner/vllm-metal/lib/python3.12/site-packages/vllm/v1/engine/core_client.py", line 535, in __init__
(APIServer pid=80518)     with launch_core_engines(
(APIServer pid=80518)          ^^^^^^^^^^^^^^^^^^^^
(APIServer pid=80518)   File "/Users/asantiola/.docker/model-runner/vllm-metal/lib/python3.12/contextlib.py", line 144, in __exit__
(APIServer pid=80518)     next(self.gen)
(APIServer pid=80518)   File "/Users/asantiola/.docker/model-runner/vllm-metal/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 998, in launch_core_engines
(APIServer pid=80518)     wait_for_engine_startup(
(APIServer pid=80518)   File "/Users/asantiola/.docker/model-runner/vllm-metal/lib/python3.12/site-packages/vllm/v1/engine/utils.py", line 1057, in wait_for_engine_startup
```

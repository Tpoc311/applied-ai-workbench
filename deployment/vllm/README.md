# vLLM

## Pull

```bash
docker pull vllm/vllm-openai:v0.28.0
```

## Run

```bash
docker run -d \
  --name vllm \
  --gpus all \
  --ipc=host \
  -p 8100:8000 \
  -v hf-cache:/root/.cache/huggingface \
  -v vllm-cache:/root/.cache/vllm \
  vllm/vllm-openai:v0.28.0 \
  --model Qwen/Qwen3-0.6B \
  --gpu-memory-utilization 0.95 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --enforce-eager
```

## Test

```bash
curl http://localhost:8100/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {
        "role": "user",
        "content": "Say hello in one sentence."
      }
    ],
    "max_tokens": 32
  }'
```

## Stop / remove

```bash
docker stop vllm
docker rm vllm
```

Or:

```bash
docker rm -f vllm
```

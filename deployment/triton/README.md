# NVIDIA Triton Inference Server

## Pull

```bash
docker pull --platform linux/amd64 nvcr.io/nvidia/tritonserver:26.08-py3
```

## Run

Run from the repository root:

```bash
docker run -d \
  --name triton \
  --gpus all \
  -p 8000:8000 \
  -p 8001:8001 \
  -p 8002:8002 \
  -v "$(pwd)/deployment/triton/model_repository:/models:ro" \
  nvcr.io/nvidia/tritonserver:26.08-py3 \
  tritonserver --model-repository=/models
```

## Test

```bash
curl -i http://localhost:8000/v2/health/ready
```

## Stop / remove

```bash
docker stop triton
docker rm triton
```

Or:

```bash
docker rm -f triton
```

## Save / load

Save the `linux/amd64` image:

```bash
docker save \
  --platform linux/amd64 \
  -o tritonserver-26.08-amd64.tar \
  nvcr.io/nvidia/tritonserver:26.08-py3
```

Load it on another machine:

```bash
docker load -i tritonserver-26.08-amd64.tar
```

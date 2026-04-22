# Pytorch workspace

A docker workspace for using PyTorch to learn and solve various problems.

## Start/stop service

```bash
docker compose -f pytorch/compose.yaml up -d --build
docker compose -f pytorch/compose.yaml exec pytorch-workspace bash
```

```bash
docker compose -f pytorch/compose.yaml down
```

# Pytorch workspace

A docker workspace for using PyTorch to learn and solve various problems.

## Structure

```text
pytorch/
├── docker/                # dockerfile and requirements
├── image_classification/  # image classification experiments and training pipelines
├── compose.yaml           # Docker Compose configuration
└── README.md              # PyTorch workspace overview
```

## Available tasks

* [Image classification](./image_classification)
* [Object detection](./object_detection)

More tasks will be added later as the workspace grows.

## Start/stop service

```bash
docker compose -f pytorch/compose.yaml up -d --build
docker compose -f pytorch/compose.yaml exec pytorch-workspace bash
```

```bash
docker compose -f pytorch/compose.yaml down
```

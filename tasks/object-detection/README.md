# Object detection

## Start/stop service

```bash
docker compose -f tasks/object-detection/compose.yaml up -d --build
docker compose -f tasks/object-detection/compose.yaml exec workspace bash
```

```bash
docker compose -f tasks/object-detection/compose.yaml down
```

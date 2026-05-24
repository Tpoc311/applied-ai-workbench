# Local MLflow setup

## Deploy Postgres

```bash
docker run -d \
  --name mlflow-postgres \
  --restart unless-stopped \
  -e POSTGRES_USER=mlflow \
  -e POSTGRES_PASSWORD=<PASSWORD> \
  -e POSTGRES_DB=mlflow \
  -p "127.0.0.1:5432:5432" \
  -v /disk/volumes/mlflow/postgres:/var/lib/postgresql \
  postgres:18.4
```

## Install dependencies

```bash
pip3 install psycopg2-binary mlflow==3.12.0
```

## Configure autorun

```bash
sudo nano /etc/systemd/system/mlflow.service
```

Insert there the config with specified credentials and URLs:

```text
[Unit]
Description=MLflow Tracking Server
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
User=me
WorkingDirectory=/home/me
Restart=always
RestartSec=10

EnvironmentFile=/etc/env/mlflow.env

ExecStart=/home/me/.venv/main/bin/mlflow server \
  --host ${MLFLOW_HOST} \
  --port ${MLFLOW_PORT} \
  --allowed-hosts ${MLFLOW_ALLOWED_HOSTS} \
  --backend-store-uri ${MLFLOW_BACKEND_STORE_URI} \
  --artifacts-destination ${MLFLOW_ARTIFACTS_DESTINATION} \
  --cors-allowed-origins ${MLFLOW_CORS_ORIGINS}

[Install]
WantedBy=multi-user.target
```

Start MLflow daemon:

```bash
sudo systemctl daemon-reload
sudo systemctl enable mlflow
sudo systemctl start mlflow
```

Check status:

```bash
sudo systemctl status mlflow
```
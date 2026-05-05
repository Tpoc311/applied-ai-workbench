# Local DVC setup

## Install DVC

```bash
pip3 install dvc==3.67.1
```

## Configure remote

```bash
dvc remote modify --local applied-ai-workbench url ssh://<ADDRESS>:<PATH>
```

## Add user:

```bash
dvc remote modify --local applied-ai-workbench user <LOGIN>
dvc remote modify --local applied-ai-workbench password <PASSWORD>
```

## Clean removed files from remote

```bash
dvc gc --workspace --cloud -r applied-ai-workbench
```

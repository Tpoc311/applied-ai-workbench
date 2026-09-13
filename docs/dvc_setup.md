# DVC setup

## Remote setup

Use this setup when working from another machine.

### Install DVC

For remote setup, DVC with SSH support is required:

```bash
pip3 install "dvc[ssh]==3.67.1"
```

SSH config must be configured and the SSH connection must work:

```bash
ssh <SSH_HOST>
```

### Configure DVC remote

```bash
dvc remote modify --local applied-ai-workbench url ssh://<SSH_HOST>/<PATH>
```

Example:

```bash
dvc remote modify --local applied-ai-workbench url ssh://my-c1/disk/volumes/dvc/applied-ai-workbench
```

## Local setup

Use this setup when working directly on the machine with DVC storage.

### Install DVC

```bash
pip3 install dvc==3.67.1
```

### Configure DVC remote

```bash
dvc remote modify --local applied-ai-workbench url <PATH>
```

Example:

```bash
dvc remote modify --local applied-ai-workbench url /disk/volumes/dvc/applied-ai-workbench
```

## Clean removed files from remote

```bash
dvc gc --workspace --cloud -r applied-ai-workbench
```

# CVEvolve Remote Deployment (Podman)

## 1. Transfer files to remote machine

```bash
REMOTE=user@remote-host

# Pre-built bin images (971 MB)
scp /home/takaji/xrd_5x5_bins.h5 $REMOTE:~/xrd_5x5_bins.h5

# CVEvolve source + 5x5 config (small)
scp -r CVEvolve $REMOTE:~/CVEvolve
scp -r cvevolve_5x5 $REMOTE:~/cvevolve_5x5
```

## 2. Update paths for the remote machine

Three types of paths need updating:

**config.yaml** — change workspace paths:
```yaml
workspace:
  root_dir: /app/cvevolve_5x5/sessions
  data_dir: /app/cvevolve_5x5/test_data
  holdout_data_dir: /app/cvevolve_5x5/holdout_data
```

**prompt.md, holdout_test_prompt.md, test_data/baseline.py** — change HDF5 path:
```
/home/takaji/xrd_5x5_bins.h5  →  /data/xrd_5x5_bins.h5
```

Quick sed on the remote machine:
```bash
cd ~/cvevolve_5x5
sed -i 's|/home/takaji/xrd_5x5_bins.h5|/data/xrd_5x5_bins.h5|g' prompt.md holdout_test_prompt.md test_data/baseline.py
sed -i 's|root_dir: .*|root_dir: /app/cvevolve_5x5/sessions|' config.yaml
sed -i 's|data_dir: .*|data_dir: /app/cvevolve_5x5/test_data|' config.yaml
sed -i 's|holdout_data_dir: .*|holdout_data_dir: /app/cvevolve_5x5/holdout_data|' config.yaml
```

## 3. Build the container

Create `~/Containerfile`:
```dockerfile
FROM python:3.12-slim

RUN pip install uv

COPY CVEvolve /app/CVEvolve
COPY cvevolve_5x5 /app/cvevolve_5x5

WORKDIR /app/CVEvolve
RUN uv venv && uv sync

ENTRYPOINT ["uv", "run", "cvevolve", "run", \
    "--config", "/app/cvevolve_5x5/config.yaml", \
    "--prompt", "/app/cvevolve_5x5/prompt.md", \
    "--holdout-test-prompt", "/app/cvevolve_5x5/holdout_test_prompt.md"]
```

Build:
```bash
cd ~
podman build -t cvevolve-5x5 -f Containerfile .
```

## 4. Run

```bash
podman run --rm -it \
  -v ~/xrd_5x5_bins.h5:/data/xrd_5x5_bins.h5:ro \
  -v ~/cvevolve_5x5/sessions:/app/cvevolve_5x5/sessions \
  -e ARGO_API_KEY=trobson \
  cvevolve-5x5
```

The sessions volume mount persists results back to the host.

## 5. Resume a session

```bash
podman run --rm -it \
  -v ~/xrd_5x5_bins.h5:/data/xrd_5x5_bins.h5:ro \
  -v ~/cvevolve_5x5/sessions:/app/cvevolve_5x5/sessions \
  -e ARGO_API_KEY=trobson \
  --entrypoint uv \
  cvevolve-5x5 run cvevolve resume \
  --session /app/cvevolve_5x5/sessions/hotspot_5x5_binned
```

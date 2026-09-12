---
sidebar_position: 4
title: "Install Guide: Docker Compose"
description: "Executable Claude Code installation guide for Docker Compose deployment"
---

# Install Guide: Docker Compose

:::info This document is written as executable instructions for Claude Code
Read the entire **Before We Begin** section first and collect the user's answers. Then follow each section in order. Every check must pass (or be acknowledged) before moving to the next section.
:::

:::tip Human readers
This guide is formatted for an AI coding agent to execute interactively — paste it into Claude Code (or a similar agent) and it will run the install for you, asking the questions below along the way. Prefer to install by hand? Start with the [Installation overview](/getting-started/installation) instead.
:::

## Before We Begin

Present each question to the user and record their answers. Use the recorded answers throughout all subsequent steps.

---

**Q1 — Run Mode**

> "Are you running me directly on the target machine, or from a workstation that will SSH into the target machine?"

- **`local`** — Claude Code is running on the machine where miStudio will be installed. Use direct shell commands.
- **`remote`** — Claude Code is running on a workstation. Ask: *"What is the SSH user and hostname or IP of the target machine? (e.g. `sean@192.168.1.100`)"* Record as `SSH_TARGET`. Prefix all target-machine commands with `ssh $SSH_TARGET "..."`.

---

**Q2 — Missing Prerequisites**

> "If I find a required tool or driver is missing, should I attempt to install it automatically (requires sudo), or report what's missing and stop so you can handle it?"

- **`auto`** — Attempt installation automatically via apt/curl where possible.
- **`diagnose`** — Report the issue with fix instructions and stop.

---

**Q3 — Secrets**

> "Should I generate secure random values for the database password and SECRET_KEY, or will you provide them?"

- **`generate`** — Claude Code generates values using `openssl rand`.
- **`provide`** — Ask the user for each value before proceeding.

---

Record answers as `RUN_MODE`, `PREREQ_MODE`, `SECRETS_MODE`. Confirm with the user before proceeding.

---

## Pre-Flight Checks

Run all checks before any installation steps. For each result:
- **PASS** — continue silently
- **WARN** — print the warning and ask the user whether to continue
- **FAIL (auto)** — attempt the documented fix, then re-check; if still failing, stop and report
- **FAIL (diagnose)** — print the issue and fix instructions, then stop

### Hardware

**GPU present**
```bash
lspci | grep -i nvidia
```
- PASS: at least one result
- FAIL: "No NVIDIA GPU detected. miStudio requires a CUDA-capable GPU for SAE training and feature extraction."

**NVIDIA driver**
```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```
- PASS: returns GPU name and driver version
- FAIL auto: Print "Driver installation requires a reboot and cannot be automated safely. Install with: `sudo apt install nvidia-driver-535` then reboot." Stop.
- FAIL diagnose: Same message. Stop.

**VRAM**
```bash
nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits
```
- PASS ≥ 16384 MB: optimal
- WARN 8192–16383 MB: "8–15GB VRAM detected. Suitable for small models (≤3B). Large models or wide SAEs (131k features) will OOM."
- FAIL < 8192 MB: "Less than 8GB VRAM. miStudio requires at least 8GB for minimal functionality."

**Disk space**
```bash
df -BG / | tail -1 | awk '{print $4}' | tr -d 'G'
```
- PASS ≥ 100 GB free
- WARN 50–99 GB: "Limited disk space. Models and datasets will fill this quickly. Proceed with caution."
- FAIL < 50 GB: "Less than 50GB free. Provision more disk space before installing."

### Software

**OS**
```bash
. /etc/os-release && echo "$ID $VERSION_ID"
```
- PASS: Ubuntu 20.04+ or Debian 11+
- WARN: other Linux — "Untested OS. Proceeding may require manual adjustments."
- FAIL: macOS or Windows — "miStudio requires a Linux host with NVIDIA GPU support."

**Docker Engine**
```bash
docker version --format '{{.Server.Version}}' 2>/dev/null || echo "NOT_FOUND"
```
- PASS: version 20.10 or higher
- FAIL auto:
  ```bash
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker $USER
  # Note: user must log out and back in for group change to take effect
  newgrp docker
  ```
- FAIL diagnose: "Install Docker Engine: https://docs.docker.com/engine/install/ubuntu/"

**Docker Compose v2**
```bash
docker compose version 2>/dev/null || echo "NOT_FOUND"
```
- PASS: `v2.x` (the `docker compose` subcommand works)
- FAIL auto: `sudo apt install docker-compose-plugin`
- FAIL diagnose: "Install Docker Compose v2: `sudo apt install docker-compose-plugin`"

**Docker daemon running**
```bash
docker info > /dev/null 2>&1 && echo "RUNNING" || echo "NOT_RUNNING"
```
- PASS: `RUNNING`
- FAIL auto: `sudo systemctl start docker && sudo systemctl enable docker`
- FAIL diagnose: "Start Docker: `sudo systemctl start docker`"

**NVIDIA Container Toolkit**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi 2>&1 | grep -c "Driver Version" || echo "0"
```
- PASS: returns `1`
- FAIL auto:
  ```bash
  distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt update && sudo apt install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
  ```
- FAIL diagnose: "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"

**Git**
```bash
git --version 2>/dev/null || echo "NOT_FOUND"
```
- PASS: any version
- FAIL auto: `sudo apt install -y git`
- FAIL diagnose: "Install git: `sudo apt install git`"

### Ports

Check that required ports are free:
```bash
ss -tlnp | grep -E ':80 |:3000 |:8000 |:5432 |:5433 |:6379 |:11434 |:3001 '
```
- PASS: no output
- WARN: for each occupied port, identify the holding process and print:
  `"Port XXXX is in use by [process]. Stop it or miStudio's [service] will fail to start."`
  Ask the user to resolve before continuing.

### Network

**Internet access (Docker Hub)**
```bash
curl -sf --max-time 10 https://hub.docker.com > /dev/null && echo "OK" || echo "FAIL"
```
- PASS: `OK`
- FAIL: "Cannot reach Docker Hub. Check internet connectivity and firewall rules. miStudio images must be pulled on first run."

---

## Configuration

### Domain Name

Ask the user:
> "What hostname should miStudio be accessible at? Press Enter to use the default: `mistudio.hitsai.local`"

Record as `DOMAIN`. Default: `mistudio.hitsai.local`.

Add to `/etc/hosts` if not already present:
```bash
grep -q "$DOMAIN" /etc/hosts || echo "127.0.0.1  $DOMAIN" | sudo tee -a /etc/hosts
```

Confirm: `"miStudio will be accessible at http://$DOMAIN"`

### Secrets

**If SECRETS_MODE=generate:**
```bash
POSTGRES_PASSWORD=$(openssl rand -hex 16)
SECRET_KEY=$(openssl rand -hex 32)
```
Print both values and instruct the user to save them:
> "Generated credentials — save these now:
> POSTGRES_PASSWORD: `$POSTGRES_PASSWORD`
> SECRET_KEY: `$SECRET_KEY`"

**If SECRETS_MODE=provide:**
Ask the user:
- "What should the PostgreSQL password be?" → `POSTGRES_PASSWORD`
- "What should the SECRET_KEY be? (used for AES-256 encryption of stored API keys — use a long random string)" → `SECRET_KEY`

### Optional API Keys

Ask the user:
> "Do you have a HuggingFace token? (needed for gated models like Gemma). Press Enter to skip."
Record as `HF_TOKEN`. Default: empty.

> "Do you have an OpenAI API key? (needed for GPT-based auto-labeling). Press Enter to skip."
Record as `OPENAI_API_KEY`. Default: empty.

---

## Installation

### Step 1 — Clone the repository

```bash
git clone https://github.com/Onegaishimas/miStudio.git
cd miStudio
```

### Step 2 — Create the `.env` file

```bash
cp .env.example .env
```

Write the collected values into `.env`:
```bash
cat > .env << EOF
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DB=mistudio
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO
SECRET_KEY=$SECRET_KEY
HF_TOKEN=$HF_TOKEN
OPENAI_API_KEY=$OPENAI_API_KEY
EOF
```

### Step 3 — Update domain references

If `DOMAIN` differs from `mistudio.hitsai.local`, update nginx config and compose env vars:
```bash
# Update nginx config
sed -i "s/dkr-mistudio\.hitsai\.local/$DOMAIN/g" nginx/nginx.docker.conf

# Update backend environment in docker-compose.yml
sed -i "s|http://dev-mistudio\.hitsai\.local|http://$DOMAIN|g" docker-compose.yml
```

### Step 4 — Pull images

```bash
docker compose pull
```

This pulls `hitsai/mistudio-backend`, `hitsai/mistudio-frontend`, `postgres`, `redis`, `nginx`, and the Neuronpedia webapp. Expect several minutes on first run.

### Step 5 — Start all services

```bash
docker compose up -d
```

Watch for startup errors:
```bash
docker compose ps
docker compose logs --tail=20
```

All services should reach `healthy` or `running` status. The backend runs Alembic migrations automatically on first start — this may take up to 60 seconds.

### Step 6 — Wait for backend ready

Poll until the API responds (up to 3 minutes):
```bash
echo "Waiting for backend..."
for i in $(seq 1 36); do
  curl -sf http://localhost:8000/api/v1/system/health > /dev/null && echo "Backend ready after ${i}0s" && break
  echo "  Attempt $i/36..."
  sleep 5
done
```

---

## Post-Install Verification

Run each check and report results:

```bash
# All containers running
docker compose ps

# Backend API health
curl -s http://$DOMAIN/api/v1/system/health

# Frontend reachable
curl -sf http://$DOMAIN > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"

# GPU accessible inside backend container
docker compose exec backend nvidia-smi

# Database migrations applied
docker compose exec backend python -c "from src.core.database import engine; print('DB: OK')"
```

Print access summary:
```
✓ miStudio is running at: http://$DOMAIN
✓ API docs:               http://localhost:8000/api/docs
✓ Backend direct:         http://localhost:8000
✓ Frontend direct:        http://localhost:3000
```

---

## Troubleshooting Quick Reference

| Symptom | Check | Fix |
|---------|-------|-----|
| Backend container exits immediately | `docker compose logs backend` | Usually DB not ready — check postgres health |
| `nvidia-smi` fails inside container | `docker run --gpus all nvidia/cuda:12.1.0-base nvidia-smi` | NVIDIA Container Toolkit not configured |
| Port 80 already in use | `ss -tlnp \| grep :80` | Stop conflicting service or change `NGINX_HTTP_PORT` in `.env` |
| Frontend loads but API calls fail | `docker compose logs nginx` | Check nginx proxy_pass config |
| Celery worker OOM on training | `docker compose logs celery-worker` | Reduce batch size in training config, or use a smaller model |
| DB migration fails on startup | `docker compose logs backend \| grep alembic` | `docker compose exec backend alembic upgrade head` |
| Image pull fails | `docker pull hitsai/mistudio-backend:latest` | Check internet access and Docker Hub status |

---
sidebar_position: 2
title: "System Requirements & Installation"
description: "Hardware requirements and software installation guide"
---

# System Requirements & Installation

## Hardware Requirements

| Tier | VRAM | Capability |
|------|------|-----------|
| **Minimum** | 8 GB | TinyLlama (1.1B), Phi-2, Phi-4-mini |
| **Recommended** | 16–24 GB (RTX 3090/4090) | Models up to 9B, wide SAEs (16k–131k features) |
| **Multi-GPU** | 2×24 GB+ | Dedicated inference + training partitions |

:::warning VRAM vs. System RAM
System RAM cannot compensate for low VRAM. Model weights and activations must reside on the GPU for acceptable speed. If a job exceeds VRAM, you'll get an "Out of Memory" (OOM) crash — the most common failure mode in local research.
:::

## Software Installation

miStudio is packaged as a Docker Compose project. The primary way to bring the stack up is `docker compose`:

1. **Prerequisites:** Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
2. **Network Setup:** Add the domain to your hosts file:
   ```bash
   sudo bash -c 'echo "127.0.0.1  mistudio.hitsai.local" >> /etc/hosts'
   ```
3. **Start the stack:**
   ```bash
   docker compose up -d
   ```

Once the containers are healthy, open the dashboard at `http://mistudio.hitsai.local` (served by the Nginx container on port 80).

:::tip start-mistudio.sh (dev convenience)
`./start-mistudio.sh` is a **development-only convenience wrapper** that starts the Docker services and then launches the backend and frontend natively for hot-reload iteration. For a normal deployment — and for anything reproducible — use `docker compose up -d` as shown above.
:::

### Services

`docker compose up -d` launches the full stack — roughly nine services:

| Service | Container | Purpose |
|---------|-----------|---------|
| **PostgreSQL** | `mistudio-postgres` | Stores all experiment metadata, labels, metrics, and settings |
| **Redis** | `mistudio-redis` | Message broker for the Celery task queue |
| **FastAPI Backend** | `mistudio-backend` | API orchestrator with WebSocket support for real-time updates |
| **React Frontend** | `mistudio-frontend` | Interactive dashboard (served behind Nginx) |
| **Celery Worker** | `mistudio-celery-worker` | Performs GPU-intensive training, extraction, and labeling tasks |
| **Celery Beat** | `mistudio-celery-beat` | Schedules periodic tasks (system monitoring, cleanup) |
| **Nginx** | `mistudio-nginx` | Reverse proxy on port 80 — routes `/api`, `/ws`, and `/` to backend and frontend |
| **Neuronpedia** | `mistudio-neuronpedia` | Feature dashboard webapp (port 3001) for browsing pushed activations |
| **Neuronpedia PostgreSQL** | `mistudio-neuronpedia-postgres` | Dedicated database backing the Neuronpedia webapp |

:::info Optional: MCP server profile
The MCP server (for agent access) is **not** started by default. Enable it with the `mcp` Compose profile:

```bash
# MCP_AUTH_TOKEN is REQUIRED — the port is LAN-reachable by default
MCP_AUTH_TOKEN=your-long-random-token docker compose --profile mcp up -d
```

It listens on port **8765** and requires `MCP_AUTH_TOKEN` to be set (the port is LAN-reachable, so firewall 8765 if agents are local-only). See the MCP server documentation for tool categories and configuration.
:::

:::info Why Docker?
A MechInterp environment requires exact versions of PyTorch, Transformers, spaCy, and CUDA kernels. Docker freezes these into a reproducible image — miStudio runs identically on a Jetson Orin and a datacenter server.
:::

## Kubernetes

Kubernetes is the recommended deployment method for shared lab environments and multi-user research clusters. The kustomize base at `k8s/base/` deploys the full miStudio stack into a dedicated `mistudio` namespace. (It is also what ArgoCD applies, so a manual `kubectl apply -k` and a GitOps sync converge on the same thing.)

### Architecture

```
┌─────────────────────────────────────────────────┐
│  Namespace: mistudio                             │
│                                                  │
│  ┌──────────┐  ┌──────────┐                     │
│  │ postgres │  │  redis   │  (persistent storage)│
│  └──────────┘  └──────────┘                     │
│                                                  │
│  ┌──────────────────────────────────────┐        │
│  │  mistudio-backend Pod (GPU node)     │        │
│  │  ├── backend      (FastAPI :8000)    │        │
│  │  ├── celery-worker (GPU tasks)       │        │
│  │  └── celery-beat  (scheduled tasks) │        │
│  └──────────────────────────────────────┘        │
│                                                  │
│  ┌────────────────────┐                          │
│  │ mistudio-frontend  │  (React/Nginx :80)       │
│  └────────────────────┘                          │
│                                                  │
│  ┌────────────────────┐                          │
│  │ ollama-proxy       │  (ExternalName service)  │
│  └────────────────────┘                          │
└─────────────────────────────────────────────────┘
         │
    NGINX Ingress
    ├── /api  → mistudio-backend:8000
    ├── /ws   → mistudio-backend:8000 (WebSocket)
    ├── /ollama → ollama-proxy:11434
    └── /    → mistudio-frontend:80
```

The backend pod runs three containers sharing a single GPU and a shared `/data` volume — FastAPI handles API requests, Celery Worker runs training/extraction/labeling jobs, and Celery Beat fires scheduled tasks like system monitoring.

### Prerequisites

**Cluster requirements:**
- Kubernetes 1.25+ (MicroK8s, k3s, or full K8s)
- NGINX Ingress Controller (`ingressClassName: public`)
- [NVIDIA Device Plugin](https://github.com/NVIDIA/k8s-device-plugin) for GPU scheduling
- At least one node with an NVIDIA GPU and the NVIDIA Container Toolkit installed

**Local tooling:**
```bash
# Verify kubectl is connected to your cluster
kubectl cluster-info

# Verify NVIDIA device plugin is running
kubectl get pods -n kube-system | grep nvidia

# Verify GPU is schedulable
kubectl describe node <gpu-node> | grep nvidia.com/gpu
```

### Step 1: Prepare Host Storage

miStudio uses `hostPath` volumes for persistent data. Create the required directories on the GPU node before deploying:

```bash
# Run on the GPU node (or via ssh)
sudo mkdir -p /data/mistudio/postgres
sudo mkdir -p /data/mistudio/redis
sudo mkdir -p /data/mistudio/data
sudo chown -R 1000:1000 /data/mistudio
```

The `/data/mistudio/data` directory holds all miStudio working data — downloaded models, datasets, SAE weights, activations, and checkpoints. Size this volume accordingly (500 GB+ recommended for active research).

### Step 2: Create the Secret and set your hostname

:::danger This section used to describe editing a manifest that no longer exists
It told you to open `k8s/mistudio-deployment.yaml` and edit `POSTGRES_PASSWORD`
and `SECRET_KEY` in place. That manifest was a **stale duplicate** of
`k8s/base` — the live deployment reads those values from a Kubernetes Secret
via `secretKeyRef`, so editing them there changed nothing that ArgoCD applies.
The file has since been deleted (MIS-E2E-144, MIS-E2E-152).
:::

Every per-install value lives in one Secret. See
[the K8s install guide](/getting-started/install-guide-k8s) for the full walk-through.

```bash
PG_PASS=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
SK=$(python3 -c "import secrets; print(secrets.token_hex(32))")

kubectl create namespace mistudio --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic mistudio-secrets -n mistudio \
  --from-literal=postgres-password="$PG_PASS" \
  --from-literal=database-url="postgresql+asyncpg://mistudio:${PG_PASS}@postgres:5432/mistudio" \
  --from-literal=database-url-sync="postgresql+psycopg2://mistudio:${PG_PASS}@postgres:5432/mistudio" \
  --from-literal=secret-key="$SK" \
  --from-literal=mcp-auth-token="$(openssl rand -hex 32)"
```

**Hostnames** live in `k8s/base/ingress.yaml`; edit them to your domain.

**GPU node pin** is a commented `nodeSelector` in `k8s/base/backend.yaml` —
uncomment it only if your cluster needs the pin.

**Optional integrations** (Ollama, a local Neuronpedia) are plain env vars in
`k8s/base/backend.yaml`; comment them out if unused.

### Step 3: Deploy

```bash
# Apply the full manifest
kubectl apply -k k8s/base

# Watch pods come up
kubectl get pods -n mistudio -w
```

Expected output once healthy:
```
NAME                                  READY   STATUS    RESTARTS   AGE
mistudio-backend-xxxxxxxxx-xxxxx      3/3     Running   0          60s
mistudio-frontend-xxxxxxxxx-xxxxx     1/1     Running   0          60s
postgres-xxxxxxxxx-xxxxx              1/1     Running   0          60s
redis-xxxxxxxxx-xxxxx                 1/1     Running   0          60s
```

:::info 3/3 on the backend pod
The backend pod runs three containers: `backend`, `celery-worker`, and `celery-beat`. All three must show Ready before the application is fully functional. Database migrations run automatically on first start via the entrypoint.
:::

### Step 4: Configure DNS

Add the ingress hostname to your DNS or local hosts file:

```bash
# On each client machine
echo "192.168.x.x  k8s-mistudio.yourdomain.com" | sudo tee -a /etc/hosts
```

Then access miStudio at `http://k8s-mistudio.yourdomain.com`.

### Verifying the Deployment

```bash
# Pod status
kubectl get pods -n mistudio

# Check backend logs (API container)
kubectl logs -n mistudio deployment/mistudio-backend -c backend --tail=50

# Check Celery worker logs
kubectl logs -n mistudio deployment/mistudio-backend -c celery-worker --tail=50

# Check Celery beat logs
kubectl logs -n mistudio deployment/mistudio-backend -c celery-beat --tail=50

# Verify GPU is allocated
kubectl exec -n mistudio deployment/mistudio-backend -c backend -- nvidia-smi

# Confirm API is responding
curl http://k8s-mistudio.yourdomain.com/api/v1/system/health
```

### Updating to New Images

miStudio publishes new images to DockerHub on every push to `main`. To update a running cluster:

```bash
# Pull latest images on the node and restart
kubectl rollout restart deployment/mistudio-backend -n mistudio
kubectl rollout restart deployment/mistudio-frontend -n mistudio

# Wait for rollout to complete
kubectl rollout status deployment/mistudio-backend -n mistudio --timeout=180s
kubectl rollout status deployment/mistudio-frontend -n mistudio --timeout=180s
```

:::info Recreate strategy
The backend uses `strategy: Recreate` — the old pod terminates completely before the new one starts. This prevents two pods from competing for the same GPU and the same data directory simultaneously.
:::

### Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVICE_TYPE` | `api` | Container role: `api`, `celery-worker`, or `celery-beat` |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Async PostgreSQL connection string |
| `DATABASE_URL_SYNC` | `postgresql+psycopg2://...` | Sync PostgreSQL connection string (Alembic) |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection string |
| `CELERY_BROKER_URL` | `redis://redis:6379/0` | Celery broker |
| `CELERY_RESULT_BACKEND` | `redis://redis:6379/0` | Celery result store |
| `SECRET_KEY` | *(change this)* | AES-256-GCM key for encrypting stored API keys |
| `DATA_DIR` | `/data` | Root for all miStudio data on the pod |
| `INTERNAL_API_URL` | `http://mistudio-backend:8000` | Internal URL for Celery→API callbacks |
| `OLLAMA_URL` | `http://ollama-proxy:11434` | Ollama endpoint for local LLM labeling |
| `NEURONPEDIA_LOCAL_URL` | *(optional)* | Local Neuronpedia instance for feature export |
| `NEURONPEDIA_LOCAL_DB_URL` | *(optional)* | Direct DB connection to local Neuronpedia |

---
sidebar_position: 5
title: "Install Guide: Kubernetes"
description: "Executable Claude Code installation guide for Kubernetes deployment"
---

# Install Guide: Kubernetes

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

> "Are you running me directly on the machine that has kubectl access to the cluster, or from a workstation that will SSH into a jump host?"

- **`local`** — `kubectl` is available directly. Use kubectl commands as-is.
- **`remote`** — Ask: *"What is the SSH user and hostname of the machine with kubectl access? (e.g. `sean@192.168.1.100`)"* Record as `SSH_TARGET`. Prefix all kubectl/ssh commands accordingly.

---

**Q2 — Missing Prerequisites**

> "If I find a required tool is missing, should I attempt to install it automatically (requires sudo), or report what's missing and stop so you can handle it?"

- **`auto`** — Attempt installation automatically.
- **`diagnose`** — Report with fix instructions and stop.

---

**Q3 — Secrets**

> "Should I generate secure random values for the database password and SECRET_KEY, or will you provide them?"

- **`generate`** — Claude Code generates values using `openssl rand`.
- **`provide`** — Ask the user for each value before proceeding.

---

**Q4 — GPU Node**

> "Should I discover the available nodes in your cluster and let you choose the GPU node, or will you provide the node name directly?"

- **`discover`** — Run `kubectl get nodes` and present the list for the user to choose from.
- **`provide`** — Ask: *"What is the exact Kubernetes node name of the GPU host?"* Record as `GPU_NODE`.

---

Record answers as `RUN_MODE`, `PREREQ_MODE`, `SECRETS_MODE`, `NODE_MODE`. Confirm with the user before proceeding.

---

## Pre-Flight Checks

Run all checks before any installation steps. For each result:
- **PASS** — continue silently
- **WARN** — print the warning and ask whether to continue
- **FAIL (auto)** — attempt the documented fix, re-check; if still failing, stop
- **FAIL (diagnose)** — print the issue and fix instructions, then stop

### Tooling

**kubectl**
```bash
kubectl version --client --short 2>/dev/null || echo "NOT_FOUND"
```
- PASS: any version returned
- FAIL auto: `sudo snap install kubectl --classic`
- FAIL diagnose: "Install kubectl: https://kubernetes.io/docs/tasks/tools/"

**Cluster reachable**
```bash
kubectl cluster-info 2>/dev/null | head -2
```
- PASS: returns control plane URL
- FAIL: "Cannot reach Kubernetes cluster. Check kubeconfig: `kubectl config current-context`. Ensure VPN or network access is active."

**Ingress controller**
```bash
kubectl get ingressclass 2>/dev/null | grep -c "public" || echo "0"
```
- PASS: returns `1` or higher
- WARN returns `0`: "No ingress class named 'public' found. The manifest uses `ingressClassName: public`. Either install an NGINX ingress controller or update the manifest to match your ingress class."
  Show available classes: `kubectl get ingressclass`

**NGINX Ingress Controller pods running**
```bash
kubectl get pods -A | grep ingress | grep -v Terminating
```
- PASS: at least one pod in Running state
- WARN: "No ingress controller pods found. Install with: `kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml`"

### GPU Node

**If NODE_MODE=discover:**
```bash
kubectl get nodes -o wide
```
Present the full node list to the user and ask them to identify the GPU node. Record as `GPU_NODE`.

**If NODE_MODE=provide:** Use the value already recorded as `GPU_NODE`.

**Node exists**
```bash
kubectl get node $GPU_NODE 2>/dev/null | grep -c "Ready" || echo "0"
```
- PASS: returns `1`
- FAIL: "Node '$GPU_NODE' not found or not Ready in the cluster. Verify the node name with `kubectl get nodes`."

**NVIDIA Device Plugin running**
```bash
kubectl get pods -n kube-system | grep nvidia-device-plugin | grep -c Running || echo "0"
```
- PASS: returns `1` or higher
- WARN returns `0`:
  ```
  "NVIDIA Device Plugin not found. GPU scheduling will not work.
  Install with:
  kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.0/nvidia-device-plugin.yml"
  ```

**GPU schedulable on node**
```bash
kubectl describe node $GPU_NODE | grep -A5 "Capacity:" | grep "nvidia.com/gpu"
```
- PASS: shows `nvidia.com/gpu: 1` (or more)
- FAIL: "GPU is not schedulable on $GPU_NODE. Check that the NVIDIA driver and device plugin are installed on that node."

### Storage

**Host storage paths — check or create on GPU node**

If `RUN_MODE=local` and the GPU node is this machine, or if `RUN_MODE=remote` and you can SSH to the GPU node directly:
```bash
# Run on the GPU node:
sudo mkdir -p /data/mistudio/postgres /data/mistudio/redis /data/mistudio/data
sudo chown -R 1000:1000 /data/mistudio
ls -la /data/mistudio/
```

If the GPU node is not directly accessible, instruct the user:
> "Please ensure these directories exist on node $GPU_NODE before continuing:
> - `/data/mistudio/postgres`
> - `/data/mistudio/redis`
> - `/data/mistudio/data`
> With ownership `1000:1000`."

**Disk space on GPU node data path**
If accessible:
```bash
df -BG /data | tail -1 | awk '{print $4}' | tr -d 'G'
```
- PASS ≥ 100 GB free
- WARN 50–99 GB: "Limited disk space on /data. Models and activation data will fill this quickly."
- FAIL < 50 GB: "Less than 50GB free on /data. Provision more storage before deploying."

### Images

**Backend image pullable**
```bash
docker pull hitsai/mistudio-backend:latest 2>&1 | tail -1
```
- PASS: `Status: Downloaded newer image` or `Status: Image is up to date`
- FAIL: "Cannot pull `hitsai/mistudio-backend:latest`. Check internet access from the cluster node and Docker Hub availability."

---

## Configuration

### GPU Node Name

Confirm `GPU_NODE` is recorded from the Pre-Flight section above.

### Domain Name

Ask the user:
> "What hostname should miStudio be accessible at? Press Enter to use the default: `k8s-mistudio.hitsai.local`"

Record as `DOMAIN`. Default: `k8s-mistudio.hitsai.local`.

Ask the user:
> "What is the IP address of the GPU node ($GPU_NODE)? This is used for the hostAlias and DNS/hosts configuration."

Record as `GPU_NODE_IP`.

### Secrets

**If SECRETS_MODE=generate:**
```bash
POSTGRES_PASSWORD=$(openssl rand -hex 16)
SECRET_KEY=$(openssl rand -hex 32)
```
Print both values:
> "Generated credentials — save these now:
> POSTGRES_PASSWORD: `$POSTGRES_PASSWORD`
> SECRET_KEY: `$SECRET_KEY`"

**If SECRETS_MODE=provide:**
Ask the user:
- "What should the PostgreSQL password be?" → `POSTGRES_PASSWORD`
- "What should the SECRET_KEY be?" → `SECRET_KEY`

### Optional Integrations

Ask the user:
> "Do you have a local Neuronpedia instance to connect to? (Press Enter to skip)"
- If yes: collect `NEURONPEDIA_URL` and `NEURONPEDIA_DB_URL`
- If no: these env vars will be left as-is in the manifest (they won't cause errors if the service is unreachable)

---

## Prepare the Manifest

Clone the repository (run from the machine with kubectl access):
```bash
# Nothing to copy or sed. The deployment is a kustomize base you apply
# directly; the values that differ per cluster live in a Secret and (if you
# need it) a nodeSelector.
```

:::danger These steps used to be four `sed` substitutions — none of which matched
This guide previously told you to copy `k8s/mistudio-deployment.yaml` and run four
`sed -i` commands against it. **None of the four target strings existed in the
shipped manifest** — it had been refactored to `secretKeyRef` against a
`mistudio-secrets` Secret, and `nodeSelector` was commented out.

`sed` exits 0 on zero matches, so every step appeared to succeed. You would
believe you had set a strong `SECRET_KEY` — the key protecting your stored API
keys — and a database password, having set neither.

One was worse than a no-op:
`sed -i "s/value: mistudio$/value: $POSTGRES_PASSWORD/g"` matched only
`POSTGRES_DB: mistudio` and `POSTGRES_USER: mistudio`, so it **renamed your
database and your database user to the password string** and still set no
password.

Recorded as MIS-E2E-152. That manifest has since been deleted outright
(MIS-E2E-144) — it was a stale duplicate of `k8s/base`.
:::

### 1. Create the Secret

Every value that differs per install lives here. `k8s/mistudio-secrets.yaml.example`
documents the same thing:

```bash
PG_PASS=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
SK=$(python3 -c "import secrets; print(secrets.token_hex(32))")
MCP_TOKEN=$(openssl rand -hex 32)

kubectl create namespace mistudio --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic mistudio-secrets -n mistudio \
  --from-literal=postgres-password="$PG_PASS" \
  --from-literal=database-url="postgresql+asyncpg://mistudio:${PG_PASS}@postgres:5432/mistudio" \
  --from-literal=database-url-sync="postgresql+psycopg2://mistudio:${PG_PASS}@postgres:5432/mistudio" \
  --from-literal=secret-key="$SK" \
  --from-literal=mcp-auth-token="$MCP_TOKEN"
```

**Verify it exists before deploying** — unlike `sed`, this one tells you if it
did not work:

```bash
kubectl get secret mistudio-secrets -n mistudio \
  -o jsonpath='{.data}' | tr ',' '\n' | cut -d'"' -f2
# expect: postgres-password, database-url, database-url-sync, secret-key, mcp-auth-token
```

### 2. Pin to your GPU node (optional)

`k8s/base/backend.yaml` carries a commented `nodeSelector`. Uncomment it and set
your node's hostname only if your cluster needs the pin; default scheduling is
fine otherwise.

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,GPU:.status.capacity.'nvidia\.com/gpu'
```

### 3. Set your hostname

The ingress in `k8s/base/ingress.yaml` lists the hostnames it answers on. Edit
them to your domain, or add yours alongside.

---

## Deployment

### Step 1 — Apply the manifest

```bash
kubectl apply -f k8s/mistudio-deployment.local.yaml
```

Expected output:
```
namespace/mistudio created (or unchanged)
deployment.apps/postgres created
service/postgres created
deployment.apps/redis created
service/redis created
deployment.apps/mistudio-backend created
service/mistudio-backend created
deployment.apps/mistudio-frontend created
service/mistudio-frontend created
service/ollama-proxy created
ingress.networking.k8s.io/mistudio-ingress created
ingress.networking.k8s.io/mistudio-websocket-ingress created
ingress.networking.k8s.io/mistudio-ollama-ingress created
```

### Step 2 — Wait for pods

```bash
kubectl rollout status deployment/postgres -n mistudio --timeout=120s
kubectl rollout status deployment/redis -n mistudio --timeout=120s
kubectl rollout status deployment/mistudio-backend -n mistudio --timeout=300s
kubectl rollout status deployment/mistudio-frontend -n mistudio --timeout=120s
```

The backend pod runs three containers (`backend`, `celery-worker`, `celery-beat`) and will show `3/3` when fully ready. Database migrations run automatically — allow up to 3 minutes on first boot.

### Step 3 — Configure DNS

Add the domain to the client machine's hosts file:
```bash
grep -q "$DOMAIN" /etc/hosts || echo "$GPU_NODE_IP  $DOMAIN" | sudo tee -a /etc/hosts
```

If `RUN_MODE=remote`, instruct the user to also add this entry on any machine that will access miStudio.

---

## Post-Install Verification

```bash
# Pod status — backend must show 3/3
kubectl get pods -n mistudio

# Backend container logs
kubectl logs -n mistudio deployment/mistudio-backend -c backend --tail=30

# Celery worker logs
kubectl logs -n mistudio deployment/mistudio-backend -c celery-worker --tail=20

# GPU allocated to backend pod
kubectl exec -n mistudio deployment/mistudio-backend -c backend -- nvidia-smi

# API health endpoint
curl -s http://$DOMAIN/api/v1/system/health

# Frontend reachable
curl -sf http://$DOMAIN > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"
```

Print access summary:
```
✓ miStudio is running at: http://$DOMAIN
✓ API docs:               http://$DOMAIN/api/docs
✓ Namespace:              mistudio
✓ GPU node:               $GPU_NODE
```

---

## Updating to New Images

When new images are available (after a CI build completes):
```bash
kubectl rollout restart deployment/mistudio-backend -n mistudio
kubectl rollout restart deployment/mistudio-frontend -n mistudio
kubectl rollout status deployment/mistudio-backend -n mistudio --timeout=300s
kubectl rollout status deployment/mistudio-frontend -n mistudio --timeout=300s
```

:::info Recreate strategy
The backend uses `strategy: Recreate`. The old pod terminates completely before the new one starts — this prevents two pods from competing for the GPU simultaneously.
:::

---

## Troubleshooting Quick Reference

| Symptom | Check | Fix |
|---------|-------|-----|
| Backend pod stuck at `0/3` | `kubectl describe pod -n mistudio -l app=mistudio-backend` | Check Events section — usually image pull failure or GPU not schedulable |
| `ImagePullBackOff` | `kubectl get events -n mistudio` | Node cannot reach Docker Hub — check internet access on GPU node |
| GPU not allocated | `kubectl describe node $GPU_NODE \| grep nvidia` | NVIDIA device plugin not running — reinstall it |
| `3/3` but API returns 503 | `kubectl logs -n mistudio deployment/mistudio-backend -c backend` | Migration failure or DB not ready — check postgres pod |
| WebSocket disconnects immediately | Check WebSocket ingress is applied | `kubectl get ingress -n mistudio` — both ingresses must exist |
| Celery worker OOM | `kubectl logs -n mistudio deployment/mistudio-backend -c celery-worker` | Reduce batch size in job config or free VRAM by unloading Ollama |
| Data missing after pod restart | `ls /data/mistudio/data` on GPU node | hostPath volume must exist with correct ownership |

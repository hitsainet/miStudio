---
sidebar_position: 100
title: "Troubleshooting"
description: "Diagnosing installation, GPU, training, labeling, and progress issues"
---

# Troubleshooting

Organized by symptom area. If your issue isn't here, check the [FAQ](/faq) or open a GitHub issue with backend logs attached.

## Installation & Startup

| Symptom | Cause | Fix |
|---------|-------|-----|
| Backend exits immediately at startup | PostgreSQL or Redis not reachable | Check `docker ps` shows the postgres/redis containers healthy; verify `DATABASE_URL`/`REDIS_URL` |
| Frontend loads but every panel is empty / spinners forever | Backend not reachable through the proxy | Verify the backend answers directly (`curl http://localhost:8000/health`), then check the nginx/ingress proxy config |
| `alembic` errors mentioning multiple heads | Migration history has parallel branches | Run `alembic upgrade heads` (plural). miStudio's entrypoint does this automatically; only manual invocations hit it |
| Jobs stay `queued` forever | No Celery worker consuming the queue | Check the worker is running (`pgrep -f celery` or the worker pod/container); check worker logs for connection errors to Redis |
| Progress bars never move but jobs finish | WebSocket connection blocked | See [Real-time updates](#real-time-updates--progress) below |

## GPU & CUDA

| Symptom | Cause | Fix |
|---------|-------|-----|
| `CUDA not available` after an OS update | Kernel upgraded; NVIDIA kernel module no longer matches the driver | Reinstall/rebuild the NVIDIA driver for the new kernel (e.g. `dkms autoinstall` or rerun the driver installer), then restart Docker/K8s node. This is the single most common "GPU vanished overnight" cause |
| `nvidia-smi` works on host but containers see no GPU | Container runtime not GPU-enabled | Verify nvidia-container-toolkit is installed and the deployment requests `nvidia.com/gpu` |
| GPU memory stays allocated after a job crashes | Zombie worker process holding VRAM | miStudio's GPU watchdog cleans these up automatically within minutes; to force it, restart the Celery worker |
| Monitor page shows no GPUs | Backend can't see CUDA | Same fixes as above — the Monitor reflects what PyTorch reports |

### Out-of-memory (OOM) sizing

Rules of thumb for fitting jobs into VRAM:

- **Model memory** ≈ `params × bytes-per-param` (FP16: 2 bytes, Q8: 1, Q4: 0.5) plus ~20% overhead
- **SAE training memory** ≈ `batch_size × d_model × expansion_factor × 4 bytes` for activations/gradients, plus the SAE weights twice (weights + optimizer state)
- **Steering** needs model + SAE + KV cache simultaneously — the KV cache grows with generation length

| Symptom | Fix |
|---------|-----|
| OOM during training | Reduce expansion factor or batch size; train from [cached activations](/advanced/multi-dataset) so the base model isn't loaded at all |
| OOM during extraction | Reduce batch size; use a quantized model; route to a bigger GPU via `gpu_id` |
| OOM during steering | Use a smaller SAE width or more aggressive model quantization; shorten max generation length |

## SAE Training Quality

| Symptom | Cause | Fix |
|---------|-------|-----|
| >50% dead neurons | Sparsity pressure too aggressive | Reduce `l1_alpha`/`sparsity_coeff`, enable sparsity warmup |
| Features look polysemantic (messy) | Sparsity too low | Increase the sparsity coefficient (~2×); target L0 of 10–100 |
| Training loss spikes | Learning rate too high | Reduce by 2–5×, increase warmup steps |
| Training loss plateaus early | Learning rate too low or not enough steps | Increase LR or `total_steps` |
| JumpReLU produces zero sparsity | `sparsity_coeff` set to an L1-scale value | These scales are **not interchangeable** — JumpReLU typical range is 1e-4 to 5e-3. See the [framework guide](/core-workflow/sae-training) |

### Key formulas

| Framework | Loss Function |
|-----------|--------------|
| **Standard** | `L = MSE(x, x̂) + λ · Σ\|z_i\|` (L1 on activations) |
| **JumpReLU** | `L = MSE(x, x̂) + λ · Σ_i H(z_i - θ_i)` (count of active features) |
| **TopK** | `L = MSE(x, x̂) + α · aux_loss(dead_features)` (no sparsity penalty — K is structural) |

## Labeling

| Symptom | Cause | Fix |
|---------|-------|-----|
| Labels say "uncategorized" | LLM couldn't interpret the feature | Increase max examples, try a larger LLM, inspect the activation examples manually |
| 503 errors from labeling jobs | The labeling endpoint has no model loaded | Load a model on your LLM server, or use **Fetch Models** in [Settings](/advanced/settings-reference) to pick one that's actually being served |
| Labeling timeouts | Local model too slow for the batch size | Reduce batch size to 1; increase the API timeout |
| Reasoning model returns empty labels | Token budget consumed by hidden reasoning | Increase max tokens — reasoning models (o-series, gpt-5) spend tokens thinking before answering. miStudio uses `max_completion_tokens` for these automatically, but the budget still has to be big enough |
| Labels contain `<think>...` fragments | Reasoning model with malformed output | miStudio strips think tags (including unclosed ones) — update to the current release if you still see them |

## Steering

| Symptom | Cause | Fix |
|---------|-------|-----|
| Steering has no visible effect | Strength too low or feature is weak | Increase strength (try 20–50); verify the feature has a crisp activation pattern in the Feature Browser |
| Steered output is gibberish | Strength too high | Back off — there's a sweet spot between "no effect" and "destroyed the model" |
| Steering job hangs long past expected time | Worker died mid-generation | The zombie-detection watchdog will fail the job; retry it. Check GPU memory wasn't exhausted |

## Real-time Updates & Progress

miStudio streams all progress over WebSocket, with automatic HTTP-polling fallback.

| Symptom | Cause | Fix |
|---------|-------|-----|
| Progress frozen but job actually running | WebSocket dropped and fallback hasn't kicked in | Refresh the page — the store re-fetches state and re-subscribes |
| Monitor page metrics stale | Celery Beat (the scheduler) not running | Check the beat process/container — it's separate from the worker |
| Everything works locally but not behind a reverse proxy | Proxy not forwarding WebSocket upgrades | Ensure the proxy passes `Upgrade`/`Connection` headers for the Socket.IO path |

## Database & Migrations

| Symptom | Cause | Fix |
|---------|-------|-----|
| `Target database is not up to date` | Pending migrations | `alembic upgrade heads` — with the **plural** `heads`, which also handles branched histories |
| Duplicate-key errors on training metrics | Pre-2026-07 databases without the unique constraint | Upgrade — the migration de-duplicates and adds `UNIQUE (training_id, step, layer_idx)` |
| Want to start fresh | — | Drop and recreate the database, then restart the backend (migrations run automatically at startup) |

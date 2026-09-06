---
sidebar_position: 9
title: "System & Task Queue API"
description: "Health, GPU/host metrics, task queue, and worker introspection"
---

# System & Task Queue API

Backs the Monitor page and operational tooling.

## System — prefix `/api/v1/system`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Backend health check |
| `POST` | `/restart` | Restart the backend process |
| `GET` | `/gpu-list` | All discovered GPUs with static info |
| `GET` | `/gpu-metrics` | Metrics for one GPU (`?gpu_id=`) |
| `GET` | `/gpu-metrics/all` | Metrics for every GPU |
| `GET` | `/gpu-info` | Detailed GPU device info |
| `GET` | `/gpu-processes` | Processes currently holding GPU memory |
| `GET` | `/metrics` | Host CPU/memory snapshot |
| `GET` | `/disk-usage` | Disk usage for the data volumes |
| `GET` | `/disk-rates` / `/network-rates` | I/O rates |
| `GET` | `/all` | Everything above in one call (what the Monitor page fetches on load) |
| `GET` | `/resource-estimate` | Estimate resources for a prospective job config |

Live metrics stream over the `system/*` WebSocket channels every 2 seconds — the REST endpoints are the polling fallback and scripting interface. See the [WebSocket reference](/reference/websocket-channels#system-monitoring).

## Task queue — prefix `/api/v1/task-queue`

The persistent record of background jobs, powering the Monitor page's **Active Operations** and **Failed Operations** sections.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `` | List task records (paginated) |
| `GET` | `/active` | Currently running/queued tasks |
| `GET` | `/failed` | Failed tasks with error details |
| `GET` | `/{id}` | One task record |
| `POST` | `/{id}/retry` | Retry a failed task |
| `DELETE` | `/{id}` | Delete a task record (204) |

## Workers — prefix `/api/v1/workers`

Celery introspection:

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/queues` | Queue depths |
| `GET` | `/active` | Tasks currently executing on workers |
| `GET` | `/stats` | Worker statistics |
| `GET` | `/health` | Worker connectivity health |

## Version — `GET /api/v1/version`

Returns the running application version.

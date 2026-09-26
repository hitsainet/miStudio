---
sidebar_position: 1
title: "API Overview"
description: "Conventions: base URL, response envelope, status codes, pagination"
---

# REST API Overview

miStudio's backend exposes a full REST API — everything the UI does, you can script. The interactive **Swagger UI** at `http://<backend>:8000/docs` is generated from the same code and is always current; these pages add organization and context.

## Base URL

All endpoints are rooted at:

```
/api/v1
```

Through the standard nginx/ingress deployment this is same-origin with the frontend (e.g., `http://mistudio.example.com/api/v1/datasets`); hitting the backend directly, it's port 8000.

## Conventions

- **Async job pattern:** endpoints that start heavy work return **`202 Accepted`** immediately with the created record; progress arrives via [WebSocket](/reference/websocket-channels) or by polling the record's `GET` endpoint. The record carries `status`, `progress` (0–100), and `error_message`.
- **Errors** return structured JSON with an appropriate status code and a `detail` message.
- **Pagination:** list endpoints accept `?page=1&limit=50` (or `skip`/`limit`) and return totals in the response body.

## Common status codes

| Code | Meaning in miStudio |
|------|--------------------|
| `200` | Success |
| `201` | Resource created |
| `202` | Background job accepted — track via WebSocket or polling |
| `204` | Deleted / no content |
| `404` | Resource not found |
| `409` | Conflict — e.g., deleting a model that a training still references |
| `410` | Endpoint removed — you're calling a deprecated path (hard-refresh the frontend) |
| `422` | Validation error (FastAPI/Pydantic detail included) |
| `503` | Dependent service unavailable — e.g., labeling LLM has no model loaded |

## Endpoint groups

| Group | Prefix | Page |
|-------|--------|------|
| Datasets & tokenization | `/datasets` | [Datasets](/reference/api/datasets) |
| Models & activation extraction | `/models` | [Models](/reference/api/models) |
| SAE training | `/trainings` | [Trainings](/reference/api/trainings) |
| SAE management | `/saes` | [SAEs](/reference/api/saes) |
| Features & labeling | *(no prefix)* `/features`, `/extractions`, `/labeling` | [Features & Labeling](/reference/api/features-labeling) |
| Steering | `/steering` | [Steering](/reference/api/steering) |
| Circuits & clusters | `/circuits`, `/circuit-capture`, `/circuit-discovery`, `/validation-manifests`, `/cluster-profiles` | [Circuits & Clusters](/reference/api/circuits) |
| Neuronpedia export & push | `/neuronpedia` | [Neuronpedia](/reference/api/neuronpedia) |
| System monitoring & task queue | `/system`, `/task-queue`, `/workers` | [System](/reference/api/system) |
| Templates & settings | `/*-templates`, `/settings` | [Templates & Settings](/reference/api/templates-settings) |

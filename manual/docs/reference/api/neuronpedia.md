---
sidebar_position: 8
title: "Neuronpedia API"
description: "Export packaging and direct-push endpoints"
---

# Neuronpedia API

Prefix: `/api/v1/neuronpedia` · UI: [Neuronpedia Export & Push](/advanced/exporting)

## Export (ZIP packages)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/export` | Start an export job — config selects features and which data to compute (logit lens, histograms, top tokens, explanations, SAELens weights) |
| `GET` | `/export/{job_id}` | Export status + stage |
| `GET` | `/export/{job_id}/download` | Download the finished ZIP |
| `POST` | `/export/{job_id}/cancel` | Cancel an in-progress export |
| `DELETE` | `/export/{job_id}` | Delete an export and its archive |
| `GET` | `/exports` | Export history |

## Direct push to local Neuronpedia

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/local-status` | Is the configured local Neuronpedia instance reachable? |
| `POST` | `/push-local` | Start a push job — creates model/SAE records and uploads features + dashboard data |
| `GET` | `/push-local/{push_job_id}` | Push progress (`features_pushed / total_features`) |
| `GET` | `/push-local` | List active push jobs |
| `POST` | `/compute-dashboard-data` | Pre-compute logit lens + histograms + statistics for an SAE without pushing |

**Progress channels:** `neuronpedia/{job_id}/export` (event `export:progress`) and `neuronpedia/push/{push_job_id}` (events `neuronpedia:push_progress|push_completed|push_failed`).

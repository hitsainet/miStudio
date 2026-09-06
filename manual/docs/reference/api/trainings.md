---
sidebar_position: 4
title: "Trainings API"
description: "SAE training job endpoints — create, control, metrics, checkpoints"
---

# Trainings API

Prefix: `/api/v1/trainings` · UI: [SAE Training](/core-workflow/sae-training)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create + start a training job (201). Config: `model_id`, dataset tokenization(s) or cached extraction(s), layers, hook types (`residual`/`mlp`/`attention`), framework + hyperparameters |
| `GET` | `` | List trainings (paginated) |
| `GET` | `/{id}` | Get training details (status, progress, live loss/L0/dead-neuron stats) |
| `PATCH` | `/{id}` | Update training metadata |
| `DELETE` | `/{id}` | Delete training and its artifacts (204) |
| `POST` | `/{id}/control` | Control a running job — body `{"action": "pause" \| "resume" \| "stop"}` |
| `GET` | `/{id}/metrics` | Time-series metrics (per step, optionally per `layer_idx` for multi-hook runs) |
| `GET` | `/{id}/checkpoints` | List saved checkpoints |
| `GET` | `/{id}/checkpoints/best` | The lowest-loss checkpoint |

**Notes**

- There is no `/{id}/retry` — "Retry" in the UI re-`POST`s a new training with the copied config.
- Multi-dataset and cached-activation training use `dataset_ids` / `extraction_ids` arrays in the create payload — see [Multi-Dataset Training](/advanced/multi-dataset).
- Metrics rows are unique per `(training_id, step, layer_idx)`; `layer_idx = null` rows are the aggregated series.

**Progress channels:** `trainings/{id}/progress` (events `training:progress|completed|failed|status_changed`), `trainings/{id}/checkpoints` (`checkpoint:created`), and `trainings/{id}/deletion` for delete progress.

## Finalization & checkpoint lifecycle

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/{id}/finalize` | Build the Community Standard export from a checkpoint |
| `GET` | `/{id}/checkpoints/prune-preview` | Read-only report of what retention would delete |
| `POST` | `/{id}/checkpoints/prune` | Apply the retention policy to this training now |
| `DELETE` | `/{id}/checkpoints/{checkpoint_id}` | Delete a single checkpoint and its file |

**`POST /{id}/finalize`** — query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `checkpoint_step` | newest complete | Step to build from |
| `allow_failed` | `false` | Permit finalizing a run whose training FAILED |
| `force` | `false` | Overwrite the export of an already-COMPLETED run |

Returns `202` with a `task_id`. Responds `409` when the training is active
(pending / initializing / running / **paused**), when it already completed and
`force` was not set, or when it failed and `allow_failed` was not set.

**`POST /{id}/control`** additionally accepts `{"action": "stop_and_finalize"}`,
which stops the run and then finalizes it from the newest checkpoint. If the run
has no checkpoints the response says so rather than reporting a finalize.

**`DELETE /{id}/checkpoints/{checkpoint_id}`** returns `204`. Deleting a
checkpoint flagged `is_best` responds `409` unless `?allow_best=true` is sent.
A `500` means the row was deliberately kept because its file could not be
removed — so a later prune can retry rather than stranding the file.

**Notes**

- Finalize is CPU-only and runs on the low-priority queue; it never waits on the GPU.
- Retention selects whole checkpoint **steps**, never individual layer rows.
- Pruning is disabled and dry-run by default (see **Settings → Storage**).

**Progress channels:** `training:completed` (carries `finalized_from_step` and
the run's real `progress`), `training:finalize_failed`.

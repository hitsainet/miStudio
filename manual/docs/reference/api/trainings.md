---
sidebar_position: 4
title: "Trainings API"
description: "SAE training job endpoints — create, control, metrics, checkpoints"
---

# Trainings API

Prefix: `/api/v1/trainings` · UI: [SAE Training](/core-workflow/sae-training)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create + start a training job (201). Config: `model_id`, dataset tokenization(s) or cached extraction(s), layers, hook types (`residual`/`mlp`/`attention`), framework + hyperparameters. **422** when a named extraction holds no activations for a requested hook type; `detail` names the missing hooks |
| `GET` | `` | List trainings (paginated) |
| `GET` | `/{id}` | Get training details (status, progress, live loss/L0/dead-neuron stats) |
| `DELETE` | `/{id}` | Delete training and its artifacts (204) |
| `POST` | `/{id}/control` | Control a running job — body `{"action": "pause" \| "resume" \| "stop"}` |
| `GET` | `/{id}/metrics` | Time-series metric rows. Query: `start_step`, `end_step`, `limit` (≤ 10,000), `aggregate_only` (only the aggregated rows, one per logged step; a raw window is shared by every SAE and held-out row of every hook type) |
| `GET` | `/{id}/checkpoints` | List saved checkpoints. After a resume, a checkpoint's `extra_metadata` carries `resume_history`: one report per resume the run's weights went through, saying whether it continued bit-identically |
| `GET` | `/{id}/checkpoints/best` | The lowest-loss checkpoint |
| `POST` | `/{id}/evaluate` | Re-run the model-cost evaluation of a completed training (202) |
| `GET` | `/{id}/evaluation` | The recorded evaluation (`{"data": …}`, `null` if never evaluated) |
| `GET` | `/{id}/features` | Features extracted from this training's SAEs: search, filter, sort, page. See [Features of a training](#features-of-a-training) |
| `GET` | `/{id}/features/by-index/{feature_idx}` | The feature id at one latent index |

**Notes**

- There is no `/{id}/retry` — "Retry" in the UI re-`POST`s a new training with the copied config.
- Multi-dataset and cached-activation training use `dataset_ids` / `extraction_ids` arrays in the create payload — see [Multi-Dataset Training](/advanced/multi-dataset).
- Per-SAE and held-out metric rows carry `hook_type` (`residual`/`mlp`/`attention`); held-out rows use `layer_idx = -1 - layer`. Rows are unique per `(training_id, step, layer_idx, COALESCE(hook_type, ''))`, so rows written before the hook was recorded keep the old per-layer key. `layer_idx = null` rows are the aggregated series and are not constrained.

- `GET /{id}` returns `current_fvu_centred` (the headline FVU) beside `current_fvu` (the legacy global-mean value), and the `evaluation` document. Metric rows carry `fvu_centred` beside `fvu`. See [Two FVU columns](/reference/data-model#two-fvu-columns-fvu-and-fvu_centred).

**Progress channels:** `trainings/{id}/progress` (events `training:progress|completed|failed|status_changed|evaluation`), `trainings/{id}/checkpoints` (`checkpoint:created`), and `trainings/{id}/deletion` for delete progress. `training:progress` carries `fvu` and `fvu_centred`.

## Features of a training

These two routes are served by the features router, under the same `/api/v1/trainings` path. The rest of
that router is in [Features & Labeling](/reference/api/features-labeling).

**`GET /{id}/features`** query parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `search` | none | Case-insensitive substring of the feature's name or description (up to 500 characters). Token contents are not searched. |
| `category` | none | Exact label category, case-insensitive |
| `is_favorite` | none | `true` or `false`; omit for both |
| `sort_by` | `activation_freq` | `activation_freq`, `max_activation`, `feature_id`, `name` or `category`. `feature_id` sorts by the feature's id string, not by its latent index. |
| `sort_order` | `desc` | `asc` or `desc` |
| `limit` | 50 | 1–500 |
| `offset` | 0 | Features to skip |

It returns `200` with `{features, total, limit, offset, statistics}`:

- `features`: for each, `id`, `extraction_job_id`, `neuron_index` (the latent index), `name`, `category`,
  `description`, `label_source`, `activation_frequency` (a fraction of tokens, 0–1), `max_activation`,
  `mean_activation`, `interpretability_score`, `is_favorite`, `star_color`, `notes`, and
  `example_context`, the feature's highest-activating example.
- `total`: the number of features that match the filters.
- `statistics`: `total_features`, `interpretable_percentage` (the share with `interpretability_score`
  above 0.5) and `avg_activation_frequency`. These cover **every** feature of the training, whatever the filters.

A training with no features returns `200` with an empty list and `total` 0, and so does an id that does
not exist; the route never answers `404`. The activation-frequency and max-activation range filters
exist only on `/extractions/{id}/features`.

**`GET /{id}/features/by-index/{feature_idx}`** returns `{"feature_id": "<id>"}`. When the training has no
feature at that index it returns `{"feature_id": null}` with `200`, not `404`. The UI uses it to open the
details of a latent added by its index. For an SAE downloaded or imported rather than trained here, use
`/api/v1/saes/{sae_id}/features/by-index/{feature_idx}`.

## Model-cost evaluation

**`POST /{id}/evaluate`** queues a GPU job. The job loads the training's base model and its exported
SAEs from `community_format/`, reads blocks each extraction never read, and records the result in
`trainings.evaluation`. See [Model-Cost Evaluation](/core-workflow/sae-training#model-cost-evaluation)
for what is measured, and [`trainings.evaluation`](/reference/data-model#trainingsevaluation) for the
document.

| Parameter | Default | Meaning |
|---|---|---|
| `token_budget` | the training's `evaluation_token_budget` (131,072) | Tokens to read from unseen blocks |
| `gpu` | the training's own request | `auto`, `all`, a GPU index or UUID. The job may split a model across GPUs. |
| `force` | `false` | Queue even though an evaluation is recorded as pending or running |

It returns `202` with `{"data": {"training_id", "task_id", "status": "pending", "gpu_request"}}` and
writes `evaluation.status = "pending"` before queueing. It responds `404` for an unknown training.
It responds `409` when the training is not completed, when it has no Community Standard export, or
when an evaluation is already pending or running and `force` was not sent. It responds `400` for a
GPU this node does not have, without writing or queueing anything.

The job records `failed` with a reason when it cannot run: the export is missing, placement is
refused, the model will not load, or the measurement raises. It records `skipped` when no block is
unseen, or when no SAE can be spliced (a transcoder, or an attention or MLP SAE). It records
`cancelled` when it is told to stop between batches: a Stop (or Stop & Finalize) of the training
during its evaluation, or a lost GPU lease; `stop_requested_by` names the control that stopped it.
None of these outcomes changes the training's status. A training is marked `completed` as soon as
its full-length export is saved, before the post-run evaluation starts, so a Stop then cancels only
the evaluation, and a Pause responds `409`. The reservation
the job asks for grows with the model's vocabulary, and a wide vocabulary reads fewer tokens per
forward pass (`config.batch_tokens` records what was used).

A running evaluation rewrites its record about once a minute. The stuck-job janitor marks one
`failed` when its record is ten minutes old and its task is no longer running. A `pending` request
is never failed automatically; the panel offers **Force re-run** (`force=true`) once it has not been
updated for 15 minutes.

`GET /{id}/metrics` rows carry `layer_idx`: `null` for the aggregate over every SAE, a layer index
for one SAE, `-1 - layer` for held-out rows, and `-1000 - layer` for spliced-CE rows written before
the `evaluation` column existed.

**`GET /{id}/evaluation`** returns `{"data": <evaluation or null>}`.

The same evaluation runs automatically after every training's Community Standard export, while the
job still holds its GPU. Set the hyperparameter `evaluate_ce_delta: false` to turn **that automatic
run** off.

It does **not** silence an explicit `POST /{id}/evaluate`: asking for an evaluation is asking for
that evaluation, so the endpoint runs it whatever the flag says. (Until 2026-09-16 the flag gated
both paths, so pressing Evaluate on such a training returned `202`, took a GPU lease, released it a
second later and recorded `"status": "skipped"` with the reason only inside the JSONB — a request
that appeared to be queued and quietly did nothing.)

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
`force` was not set, or when it failed and `allow_failed` was not set. It also
responds `409`, even with `force`, when the export already holds the run's final
weights (the loop's full-length export, whose `mistudio_checkpoint_step` is at or
past `total_steps`): finalizing would replace them with a checkpoint's older
weights. `force` still re-finalizes a run that was finalized early.

**`POST /{id}/control`** additionally accepts `{"action": "stop_and_finalize"}`,
which stops the run and then finalizes it from the newest checkpoint. If the run
has no checkpoints the response says so rather than reporting a finalize.

On a training that has already **completed**, `stop` and `stop_and_finalize`
cancel only its pending or running evaluation and finalize nothing; with no
evaluation to stop they respond `400`. A stop that arrives as the run finishes
leaves it `completed`, with the export of its final weights and no finalize.
`pause` on a completed training responds `409`.

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

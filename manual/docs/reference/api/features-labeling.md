---
sidebar_position: 6
title: "Features & Labeling API"
description: "Feature browsing, analysis, bulk labeling, and enhanced labeling endpoints"
---

# Features & Labeling API

These routers are mounted **without a prefix** — paths sit directly under `/api/v1`. UI: [Feature Extraction](/core-workflow/feature-extraction), [Auto-Labeling](/core-workflow/auto-labeling), [Enhanced Labeling](/core-workflow/enhanced-labeling).

## Extraction jobs & feature browsing

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/extractions` | List feature-extraction jobs |
| `DELETE` | `/extractions/{id}` | Delete an extraction job and its features |
| `GET` | `/extractions/{id}/features` | Features from one extraction (paginated, filterable) |
| `GET` | `/trainings/{tid}/features` | Features by source training |
| `GET` | `/features/{id}` | Feature detail |
| `GET` | `/trainings/{tid}/features/by-index/{idx}` | Look up a feature by neuron index (trained SAE) |
| `GET` | `/saes/{sae_id}/features/by-index/{idx}` | Look up a feature by neuron index (external SAE) |

## Feature detail & curation

| Method | Path | Description |
|--------|------|-------------|
| `PATCH` | `/features/{id}` | Edit name, category, description, notes. Accepts `label_source` (`user`\|`mcp_agent`) for provenance and `override_protected`; editing an aqua-starred feature's identity fields without the override returns **409 `PROTECTED_LABEL`** |
| `POST` | `/features/{id}/favorite` | Toggle favorite |
| `POST` | `/features/{id}/star` | Set star color — `?star_color=yellow\|purple\|aqua` (aqua marks completed enhanced labels and is protected from bulk overwrite) |
| `GET` | `/features/{id}/examples` | Top activating examples with per-token activations |
| `GET` | `/features/{id}/token-analysis` | Aggregated token statistics |
| `GET` | `/features/{id}/logit-lens` | Promoted/suppressed vocabulary tokens |
| `GET` | `/features/{id}/correlations` | Correlated features |
| `GET` | `/features/{id}/ablation` | Ablation analysis |

## NLP analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/extractions/{id}/analyze-nlp` | Run NLP analysis across an extraction's features |
| `POST` | `/extractions/{id}/cancel-nlp` / `.../reset-nlp` | Cancel / reset that analysis |
| `POST` | `/features/{id}/analyze-nlp` | Analyze a single feature |
| `GET` | `/features/{id}/nlp-analysis` | Retrieve stored analysis |
| `POST` | `/analysis/cleanup` | Clean up orphaned analysis artifacts |

## Cross-feature clustering (Clusters)

Powers the [Clusters view](/core-workflow/feature-groups) and the [MCP server's](/advanced/mcp-server) `groups` tools.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/extractions/{id}/feature-groups/compute` | Start the grouping precompute job (202). Idempotent per params; `?force=true` recomputes. 409 while already computing |
| `GET` | `/extractions/{id}/feature-groups/status` | Index state: `none\|pending\|computing\|completed\|failed` + counts |
| `GET` | `/extractions/{id}/feature-groups` | Paginated groups; params `token` (exact normalized), `search`, `min_group_size`, `sort_by=size\|cohesion\|token` |
| `GET` | `/extractions/{id}/feature-groups/{group_id}` | Group members with labels/stars joined live; filters `category`, `has_label`, `star_color`, `is_favorite` |
| `GET` | `/extractions/{id}/features/by-token` | Features by top token — `match=exact\|normalized\|prefix`. 409 `NO_INDEX` until computed |
| `GET` | `/features/{id}/related` | Related features via shared tokens + context overlap + cached correlations, with `link_types` per result |

## Agent approvals — prefix `/mcp/approvals`

Backs the MCP operator-approval mode; the Steering panel's approvals banner uses these.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create a pending approval request (called by the MCP server) |
| `GET` | `` | List requests (`?status=pending`) |
| `GET` | `/{id}` | Request detail incl. stored steering payload |
| `POST` | `/{id}/approve` | Approve — the backend submits the stored steering task and records its `steering_task_id` |
| `POST` | `/{id}/deny` | Deny with optional reason |

## Bulk labeling

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/labeling` | Start a bulk labeling job (LLM labels many features) |
| `POST` | `/extractions/{id}/label` | Start labeling scoped to one extraction |
| `GET` | `/labeling` | List labeling jobs |
| `GET` | `/labeling/{job_id}` | Job status + results |
| `POST` | `/labeling/{job_id}/cancel` | Cancel a running job |
| `DELETE` | `/labeling/{job_id}` | Delete a job |
| `GET` | `/labeling/models/available` | Models served by the configured local endpoint |
| `POST` | `/labeling/models/openai` | List models available to your OpenAI API key |

Returns **503** when the labeling endpoint has no model loaded — see [troubleshooting](/troubleshooting#labeling).

## Coverage & resume

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/labeling/{extraction_job_id}/coverage` | What still needs labeling, why the failures failed, and the ids of the next resume batch |
| `POST` | `/labeling/panel` | Label an explicit set of 1–2000 features (this is how a resume runs) |
| `POST` | `/labeling/{extraction_job_id}/resume-sweep` | Run many batches back to back |
| `GET` | `/labeling/resume-sweeps/{sweep_id}` | Sweep progress |
| `POST` | `/labeling/resume-sweeps/{sweep_id}/cancel` | Stop a sweep after its current batch |

### `GET /labeling/{extraction_job_id}/coverage`

| Query | Default | Meaning |
|---|---|---|
| `resume_limit` | 2000 | Ids to return for the next batch. Capped at the panel route's own limit. |
| `only` | *(both)* | `failed` or `pending` — draw the batch from ONE outcome. An unrecognised value is **422**, never widened. |
| `prompt_fingerprint` | — | With `judge_model`, also counts verdicts from a *different* judge as stale. Get it from `GET /labeling-prompt-templates`. |
| `judge_model` | — | The model half of that identity. Both or neither. |
| `max_attempts` | 3 | Features that have failed this many times stop being offered. |

```json
{
  "extraction_job_id": "extr_20260901_084144_sae_sae_eb48",
  "total": 53088,
  "by_status": { "pending": 38301, "failed": 14560, "succeeded": 227 },
  "adjudicated": 227,
  "remaining": 52861,
  "in_progress": 0,
  "unclassified": 0,
  "stale": null,
  "failures_without_a_recorded_reason": 14560,
  "caveat": "14560 of the failures carry no recorded reason…",
  "failure_reasons": [
    { "reason": "(no reason recorded)", "count": 14560,
      "sample_feature_ids": ["feat_…_00019", "feat_…_00020", "feat_…_00021"] }
  ],
  "resume_feature_ids": ["feat_…_00019", "…"]
}
```

`adjudicated` counts `succeeded` **and** `skipped`. A feature the judge found
*uninterpretable* is `succeeded` — that is a verdict, not a gap, and a resume
never offers it. `stale` is `null` unless a judge is named: reporting `0` without
one would read as "nothing is stale", which is not known.

`failure_reasons` groups on the first colon-delimited segment of the stored
error, so a variable tail (host, port, feature id) does not split one cause into
thousands of singletons. A capped tail is reported as `other (N commonest shown)`
rather than truncated — the breakdown always adds up to the failure count.

### `POST /labeling/{extraction_job_id}/resume-sweep`

```json
{ "max_batches": 27, "batch_size": 2000, "config": { "…": "a full labeling config" } }
```

`max_batches` is **required** — there is no unbounded sweep, at any layer. At
~8 s/feature a 2000-feature batch is ~4.4 hours, inside Celery's 10 h soft limit;
each batch is its own task, so a sweep never approaches that limit however long
it runs. `config` is frozen at start, so every batch uses the same judge.

Returns **409** if a sweep is already running on the extraction.

**Progress channel:** poll `GET /labeling/resume-sweeps/{sweep_id}`.
`features_labeled` and `features_failed` count what the batches actually *wrote*,
never how many batches ran.

### MCP equivalents

`get_labeling_coverage`, `resume_labeling`, `start_labeling_resume_sweep`,
`get_labeling_resume_sweep`, `cancel_labeling_resume_sweep` — category
`labeling`.

## Enhanced labeling (per-feature, two-pass)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/features/{id}/label/enhanced` | Start the two-pass analysis (parallel per-example summaries → synthesis) |
| `GET` | `/features/{id}/label/enhanced/latest` | Latest enhanced-labeling job + result for the feature |

**Progress channels:** `extraction/{id}` (feature extraction), `labeling/{job_id}/progress` + `/results` (bulk), `enhanced_labeling/{job_id}` (events `enhanced_labeling:progress|completed|failed`).

## Prompt-template trials

A trial runs one labeling prompt template over an explicit panel of features and records the labels
**without writing them to any feature row**. See
[Prompt-Template Trials](../../core-workflow/labeling-trials.md) for the workflow.

### `POST /api/v1/labeling/trials`

Start a trial. Rejects unknown fields — a typo'd key is an error rather than a silently dropped
option, because the same request shape misread would otherwise start a full-extraction labeling run.

| field | type | notes |
|---|---|---|
| `extraction_job_id` | string | required |
| `feature_ids` | string[] | required, 1–200, all must belong to that extraction |
| `prompt_template_id` | string | optional; the default template is used when omitted |
| `name` | string | optional label for the run, e.g. `baseline` |
| `labeling_method` | string | `openai` \| `openai_compatible` \| `local` |
| `openai_compatible_endpoint` | string | endpoint URL including `/v1` |
| `openai_compatible_model` | string | model name at that endpoint |

Returns `201` with `trial_run_id`, `panel_id`, and `writes_labels: false`.

| status | meaning |
|---|---|
| `404` | extraction or template not found |
| `409` | another trial is already in flight for this panel |
| `422` | a feature id is not in this extraction (the ids are named), the panel is empty or over 200, or the template is a detection/scoring template |

### `GET /api/v1/labeling/trials`

List trials. Filters: `extraction_job_id`, `panel_id`, `prompt_template_id`, plus `limit`/`offset`.
Filter by `panel_id` to find every variant run against one panel.

### `GET /api/v1/labeling/trials/{trial_run_id}`

The full record: the frozen template copy, the panel, per-feature labels, and stats.

### `GET /api/v1/labeling/trials/compare/{run_a}/{run_b}`

Per-feature comparison of two trials.

| status | meaning |
|---|---|
| `409` | the two runs used different panels — comparing them would produce a number that looks like a template difference and is not one |
| `404` | one or both trials not found |

A `200` may still carry `verdict: null` when there is nothing to compare, or `inconclusive` when
every overlapping feature errored in at least one arm. Failed labels stringify identically, so
`inconclusive` is reported rather than `identical`.

### MCP equivalents

`list_labeling_templates`, `run_labeling_trial`, `get_labeling_trial`, `list_labeling_trials`,
`compare_labeling_trials` — all in the `labeling` category.

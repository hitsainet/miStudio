---
sidebar_position: 2
title: "Datasets API"
description: "Dataset download, tokenization, and inspection endpoints"
---

# Datasets API

Prefix: `/api/v1/datasets` · UI: [Dataset Management](/core-workflow/dataset-management)

## Datasets

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `` | Create a dataset record (201) |
| `GET` | `` | List datasets (paginated) |
| `GET` | `/{id}` | Get dataset details |
| `PATCH` | `/{id}` | Update dataset metadata / filter settings |
| `DELETE` | `/{id}` | Delete dataset — cascades to its tokenizations (204) |
| `POST` | `/download` | Start a HuggingFace download (202); body includes `repo_id`, optional `split`/`config` |
| `DELETE` | `/{id}/cancel` | Cancel an in-progress download. During download: removes partial files; during processing: preserves raw files for retry |
| `GET` | `/{id}/task-status` | Status of the dataset's current background task |
| `GET` | `/{id}/samples` | Paginated raw samples (bytes fields sanitized to strings) |

## Tokenizations

A dataset has many tokenizations — unique per `(dataset, model, max_length)`.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/{id}/tokenize` | Start a tokenization job (202); body: `model_id`, `max_length`, `shuffle` (default `true`), `shuffle_seed` (optional, ≥ 0), filter options |
| `POST` | `/tokenize-preview` | Preview tokenization of sample text without running a job |
| `GET` | `/{id}/tokenizations` | List the dataset's tokenizations |
| `GET` | `/{id}/tokenizations/{model_id}` | Get the tokenization for a specific model |
| `POST` | `/{id}/tokenizations/{tok_id}/cancel` | Cancel a running tokenization |
| `DELETE` | `/{id}/tokenizations/{tok_id}` | Delete one tokenization (204) |
| `DELETE` | `/{id}/tokenization` | Clear tokenization state (legacy single-tokenization path) |

**Progress channels:** `datasets/{id}/progress` (events `dataset:progress|completed|error`) and `datasets/{id}/tokenization/{tok_id}` (events `tokenization:progress|status`) — see the [WebSocket reference](/reference/websocket-channels).

## Probe-dataset views

`/api/v1/probe-monitors/datasets` — a label-mapped view over a downloaded dataset, the input to a
probe run (see [Probe Monitors](/core-workflow/probe-monitors)).

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/probe-monitors/datasets` | Create a view: `name`, `dataset_id`, optional `config`/`split`, `input_column`, `label_column`, `label_mapping`, optional `keyword_filter`, `pair_column`, `role`, `distribution`, and optional **`dataset_version_manifest`** |
| `GET` | `/probe-monitors/datasets` | List views; each carries `counts`, `dataset_version` and `archived_at`/`archived_by`/`archive_reason`. **Archived views are hidden unless `include_archived=true`** |
| `GET` | `/probe-monitors/datasets/{id}` | One view (archived or not) |
| `GET` | `/probe-monitors/datasets/{id}/rows` | A page of the view's rows, read-only: `offset`, `limit` (1–200, default 50), `columns` (comma-separated other columns; the input, label and pair columns always come back), `label` (`positive`/`negative`/`excluded`/`filtered_out`/`unparseable`), `search` (input contains, case-insensitive). Each row has its `position` in the split, `raw_label`, `mapped_label` and `outcome`; the page carries the stored `counts`, and with a `label` filter also `recount` and `counts_agree` |
| `POST` | `/probe-monitors/datasets/{id}/archive` | Hide the view (optional body `{reason}`); idempotent. New runs, evaluations and judge baselines naming an archived view are refused **409** `probe_dataset_archived` |
| `POST` | `/probe-monitors/datasets/{id}/unarchive` | Restore it |
| `POST` | `/probe-monitors/datasets/purge/plan` | `{ids}` → per view: `purgeable`, `not_archived`, every `blocker` (`code`, `message`), what it `removes` and what is `kept`. Changes nothing |
| `POST` | `/probe-monitors/datasets/purge` | `{ids, reason?}` → permanently delete those ARCHIVED views, each on its own: `{results: [{outcome: purged\|blocked, …}], purged, blocked}` |
| `DELETE` | `/probe-monitors/datasets/{id}?reason=` | The same purge for one view: **409** `not_archived` or `purge_blocked` (every blocker named), else the purge record |
| `GET` | `/probe-monitors/dataset-purges` | The purge log: one append-only record per deleted view (who, when, why, the view as it was, what was kept) |

**Purge is operator-only.** Every purge route (plan, purge, delete, log) answers **403**
`purge_operator_only` to a request carrying `X-MiStudio-Agent` — the header the miStudio MCP server
sends on every call — and no MCP tool calls them. A purge removes the view's row only; the downloaded
dataset is kept. Blocker codes: `run_in_progress`, `run_uses_it`, `probe_calibrated_on_it`,
`evaluation_on_it`, `judge_run_in_progress`, `judge_run_uses_it`, `approval_names_it`, and
`unchecked_reference` (a new foreign key the purge has not been taught to check — it refuses rather
than guess).

**`dataset_version_manifest`** (Feature 034) is a miDataworks `midataworks.dataset-version`
manifest. When present, `split` is required (422 otherwise), the manifest must validate against the
vendored schema (422 naming the failing JSON path), must describe the split (409 listing its
splits), and must state the downloaded split's pre-filter row count (409 naming both counts). What
was verified is returned as `dataset_version`; it is `null` for a view registered without a
manifest. The field name is read by miDataworks from `/api/openapi.json` to detect support — it is
part of the cross-repo contract.

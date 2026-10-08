---
sidebar_position: 11
title: "Circuits & Clusters API"
description: "Capture, discover, validate, calibrate, promote, and export circuits and cluster profiles"
---

# Circuits & Clusters API

Prefixes: `/api/v1/circuits` · `/api/v1/circuit-capture` · `/api/v1/circuit-discovery` · `/api/v1/validation-manifests` · `/api/v1/cluster-profiles` · UI: [Circuits](/core-workflow/circuits)

This surface builds a circuit the way the evidence ladder demands — each stage **earns** a rung, and every circuit response carries both the numeric `rung` and the server-rendered `rung_language` string (the one authoritative source of causal phrasing; clients never invent it). The lifecycle runs **capture → discover → attribute → validate → build → faithfulness → promote → calibrate → export/import → record**.

GPU stages share a single-GPU guard: only one capture / discovery / attribution / validation / faithfulness / calibration / recorder pass can hold the GPU at a time. A second launch returns **`409`**. Heavy work returns **`202 Accepted`** with a `task_id`; poll the record's `GET` endpoint or subscribe to the [WebSocket channel](/reference/websocket-channels).

:::note Two planes
miStudio **discovers and calibrates** (it runs the model to learn). miLLM **serves** (it runs the model behind an OpenAI-compatible API). The boundary between them is a portable JSON document — a `mistudio.circuit-definition` or `mistudio.cluster-definition` — not a code dependency.
:::

## Capture — record activations for mining

`/api/v1/circuit-capture` collects the co-activation store discovery mines. `confirm=false` runs a GPU probe and stops at a cost estimate (status `estimated`); `POST /{id}/confirm` launches the full capture.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuit-capture` | Create a capture run. `confirm=false` → probe + estimate only; `confirm=true` → full capture. Returns a `task_id` |
| `POST` | `/circuit-capture/{run_id}/confirm` | Launch the full capture for an `estimated` (or retryable `failed`) run |
| `GET` | `/circuit-capture` | List capture runs (`?limit=&offset=`) |
| `GET` | `/circuit-capture/{run_id}` | Get one run — manifest summary, split counts, estimate |
| `POST` | `/circuit-capture/{run_id}/cancel` | Cancel a `pending`/`running`/`estimating` run |
| `DELETE` | `/circuit-capture/{run_id}` | Delete a run and its store |

**Key body fields (`POST /circuit-capture`):** `dataset_id` (required), `model_id`, `layers` (1–8 `{layer, sae_id}` entries), `epsilon` ∈ [0, 1), `theta_floor` ≥ 0, `sample_cap` ∈ [32, 100000], `split_seed`, optional `attention_capture` (`{layers, heads, top_k}`), `confirm`.

## Discover — mine candidate edges (rung 0)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuit-discovery` | Mine candidate edges from a capture store. Returns a `task_id` |
| `GET` | `/circuit-discovery` | List discovery runs (`?capture_run_id=&limit=&offset=`) |
| `GET` | `/circuit-discovery/{run_id}` | Run + first-class **report** (+ candidates unless `?include_candidates=false`) |
| `POST` | `/circuit-discovery/{run_id}/cancel` | Cancel a `pending`/`running` run |
| `DELETE` | `/circuit-discovery/{run_id}` | Delete a run (cancel it first if `running`) |

**Key body fields (`POST /circuit-discovery`):** `capture_run_id` (required), `granularity` (`feature`|`cluster`), `mode` (`open`|`seeded`), `seed_refs` (each exactly one of `feature_idx` / `cluster_profile_id`), `s_min`, `null_shuffles` ∈ [10, 1000], `null_percentile`, `fdr_q`, `cohesion_floor`, `force` (mine a stale store anyway).

The discovery **report** is the trust surface: null method, FDR discipline, replication rate, and caps are all first-class fields.

## Attribute — rank candidates by gradient (rung 1)

Attribution runs on the discovery run's **own** `attribution_*` lifecycle — a failed pass never corrupts the `completed` discovery status.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuit-discovery/{run_id}/attribution` | Tier-2 gradient attribution over the run's candidates (GPU). Body: `prompt_limit` ∈ [1, 256]. Returns a `task_id` |
| `POST` | `/circuit-discovery/{run_id}/attribution/cancel` | Cancel an in-flight attribution pass |

`409` if the discovery run is not `completed`, has no candidates, or an attribution pass is already in flight.

## Validate — intervene on edges (rung 2)

Validation performs a real intervention: it measures each edge's uplift against a null. It also runs on the discovery run's own `validation_*` lifecycle.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuit-discovery/{run_id}/validate` | Run edge-intervention validation (GPU). Returns a `task_id` |
| `POST` | `/circuit-discovery/{run_id}/validate/cancel` | Cancel an in-flight validation pass |

**Key body fields:** `ordering` (`coact`|`attr`), `k` ∈ [1, 200], `prompts_per_edge`, `null_samples`, `percentile`, `sign_frac`, `baseline` (`zero`|`corpus_mean`), `seed`.

`ordering="attr"` requires a **completed attribution pass** first (else `409`) — otherwise the coactivation order would be validated under the `attr` label and the uplift story is garbage.

## Build — turn candidates into a circuit

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuit-discovery/{run_id}/build-circuit` | Build a circuit from selected candidates of a **completed** discovery run. Body: `name` (required), `narrative`, `candidate_keys` (`[[up_layer, up_idx, down_layer, down_idx], …]`; empty ⇒ all candidates) |

The built circuit is created **unpromoted** and carries `discovery_run_id`, so a later validation pass's rung-2 effect size propagates onto it. Promote and validate separately.

### Circuit CRUD

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/circuits` | List circuits — slim rows. Filters: `?promoted=&min_rung=&granularity=&edge_type=&limit=&offset=` |
| `POST` | `/circuits` | Create a circuit directly (`201`) |
| `GET` | `/circuits/{id}` | Full circuit — SAEs, members, edges, budget, faithfulness, discovery |
| `PATCH` | `/circuits/{id}` | Update. Optimistic concurrency: pass `expected_version`; a stale value `409`s |
| `DELETE` | `/circuits/{id}` | Delete a circuit |

`edge_type` filters to circuits containing at least one edge of type `computed`, `persistence`, or `attention_mediated`. Structural fields (`name`, `saes`, `members`, `edges`) cannot be set to `null` on `PATCH` (`422`).

## Faithfulness — necessity/sufficiency (rung 3)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuits/{id}/faithfulness` | Launch a faithfulness pass: suppress the circuit's members and measure the necessity/sufficiency of the behavior they drive vs an ablate-all proxy (GPU). Returns a `task_id` |

**Key body fields:** `mode` (`necessity`|`both`), `k_nonmembers` ∈ [1, 4096], `ablate_all_n` ∈ [1, 16384], `n_prompts` ∈ [1, 256], `seed`.

`409` if the circuit has no members, has no `discovery_run_id` (v1 faithfulness needs the discovery capture store for prompts), or a pass is already in flight. Poll the task, then re-fetch the circuit for the scores (`circuit.faithfulness`).

## Promote — the badge, not the gate

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuits/{id}/promote` | Pin (or unpin) a promotion badge. Body: `promoted` — `true` or `false` (default `true`) |

Promotion is a **badge, not a gate** — and it is reversible.

## Calibrate — find the usable dial band

Calibration runs a **two-detector usable-band search** and clamps the served dial to the band it finds. It is a badge, not a gate.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuits/{id}/calibration` | Launch a strength-calibration pass (GPU). Returns a `task_id` |
| `POST` | `/circuits/calibration-manifests/{manifest_id}/reproduce` | Re-run a calibration from its manifest and record a reproduction manifest with the band-delta verdict (GPU) |

**Key body fields (`POST /circuits/{id}/calibration`):** `step_budget` ∈ [2, 40], `probe_count` ∈ [1, 10], `margin` ∈ [0, 1], `seed`, `judge_endpoint`, `judge_model` (the OpenAI-compatible judge / probe-generation LLM, carried per request like an enhanced-labeling job — required for a real run; the correctness cliff cannot be found without a judge).

The search finds two boundaries: the **onset** (minimum influence above baseline output-drift noise — no judge needed) and the **correctness cliff** (the maximum dial before the model's facts break, judged against generated neutral-topic falsifiable probes). Adaptive bisection narrows both; the served dial is clamped to `[onset, cliff]`. A weak judge reports `judge_unreliable` — never a false `no_band`.

`409` if the circuit has no members or a calibration pass is already in flight. The reproduce route `409`s if the manifest is not a `calibration` manifest, or a pass is already in flight on its circuit.

## Manifests — reproducible validation & calibration records

Manifests are self-contained records of a GPU pass. Reproducing one re-executes it from its own payload — the test that it is truly self-contained.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/validation-manifests` | List manifests (`?discovery_run_id=&circuit_id=`) |
| `GET` | `/validation-manifests/{manifest_id}` | Get one manifest — kind, payload, parent linkage |
| `POST` | `/validation-manifests/{manifest_id}/reproduce` | Re-execute an `edge_batch` manifest and store a `reproduction` manifest with per-edge deltas + a tolerance verdict |

Only `edge_batch` manifests are reproducible here (`409` otherwise); calibration manifests reproduce via the circuit route above.

## Record — Steered Transcript Recorder

Instrument, not judge. The recorder captures `(dial, prompt, unsteered, steered)` transcripts for a circuit, cluster, **or** an ad-hoc feature set — the raw material a strong model analyzes **after** the run. It is judge-free and GPU-guarded.

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/circuits/steering-samples` | Record steered/unsteered transcripts across a set of dials and prompts (GPU). Returns a `record_run_id` + `task_id` |

**Body:** `artifact` (`{kind, …}` where `kind` is `circuit`\|`cluster`\|`features` — supply `circuit_id`, `cluster_profile_id`, or `features` respectively, plus optional `model_id`), `dials` (list of floats), `prompts` (list of strings), `max_tokens`, `seed`. `422` on bad config/caps; `409` if the GPU is busy. Poll the task, then read the `steering_samples` manifest.

## Export / import — portable circuit definitions

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/circuits/{id}/export` | Download the circuit as a `mistudio.circuit-definition` JSON file |
| `POST` | `/circuits/{id}/export-slices` | Per-layer `cluster-definition/v1` slices (partial renderings), each rung-marked from the parent definition |
| `POST` | `/circuits/import` | Import a `mistudio.circuit-definition` file (`201`) |

Import is kind-keyed: the payload `kind` must be exactly `mistudio.circuit-definition` (the `/v1` suffix names the schema **version**, never the kind). Unknown kinds `422`; payloads over the 1 MB house cap `413`. The round-trip is lossless — model ref, granularity, and the authored `created_at` all survive.

## Cluster profiles

`/api/v1/cluster-profiles` is the durable, portable cluster surface (feature-granularity clusters, distinct from circuit `members`). Exports always serialize the **stored** profile (save-then-export).

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/cluster-profiles` | List profiles (`?sae_id=&search=`) |
| `POST` | `/cluster-profiles` | Create a profile (`201`) |
| `GET` | `/cluster-profiles/{id}` | Get one profile |
| `PATCH` | `/cluster-profiles/{id}` | Update a profile |
| `DELETE` | `/cluster-profiles/{id}` | Delete a profile |
| `GET` | `/cluster-profiles/{id}/export` | Export ONE profile as a `mistudio.cluster-definition/v1` file |
| `POST` | `/cluster-profiles/export-bundle` | Export several profiles as one `mistudio.cluster-bundle/v1` file. Body: `{ids: […]}` |
| `POST` | `/cluster-profiles/import` | Import a definition or bundle |

Import runs a per-item compatibility matrix: each definition **binds**, warn-binds, imports **unbound**, is **blocked** (`n_features` mismatch or member indices out of bounds), or **errors** — one bad item never poisons the rest of a bundle. Import caps: 1 MB body (`413`), ≤50 definitions per bundle, ≤20 members each.

## WebSocket channels

Each GPU stage streams progress on `circuit-{kind}/{id}` where `kind` ∈ {`capture`, `discovery`, `attribution`, `faithfulness`, `calibration`, `validation`, `steering`} (`steering` is the recorder). Events follow the house namespaced convention: `circuit_{kind}:progress`, `circuit_{kind}:completed`, `circuit_{kind}:failed`. See [WebSocket channels](/reference/websocket-channels).

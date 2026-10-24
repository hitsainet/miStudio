---
sidebar_position: 7
title: "MCP Server"
description: "Agentic access — let Claude Code and other MCP clients drive miStudio"
---

# MCP Server: Agentic Access

miStudio ships an optional **MCP (Model Context Protocol) server** that exposes the post-extraction workflow — feature analysis, [cross-feature clustering](/core-workflow/feature-groups), circuit discovery, steering, calibration, and label write-back — as tools for agentic AI clients like Claude Code. An agent can run the full *analyze → group → steer → relabel* loop autonomously, or the deeper *capture → discover → validate → calibrate → promote → export* circuit pipeline, with every action flowing through the same REST API and appearing in the UI like any other work.

The server also carries an optional **cross-product plane**: when pointed at a running [miLLM](https://github.com/hitsainet/miLLM) runtime, a family of `millm_*` tools lets the same agent move a tuned circuit or cluster into production and drive live serving. miStudio **discovers and calibrates** (it runs the model to learn); miLLM **serves** (it runs the model behind an OpenAI-compatible API). The boundary between the two is a portable document, not a code dependency.

## Enabling the Server

The MCP server is **off by default** and runs as its own container.

**Docker Compose:**

```bash
# 1. Set a token in .env (required — the port is LAN-reachable)
echo "MCP_AUTH_TOKEN=$(openssl rand -hex 32)" >> .env

# 2. Start with the mcp profile
docker compose --profile mcp up -d
```

**Kubernetes:** the server reads its token from the `mcp-auth-token` key of the `mistudio-secrets` Secret, and `k8s/base/mcp.yaml` holds the `mistudio-mcp` Deployment, Service and Ingress. Set the token, then apply:

```bash
# Create or rotate the token in the Secret (keeps the Secret's other keys).
kubectl -n mistudio patch secret mistudio-secrets --type merge \
  -p "{\"stringData\":{\"mcp-auth-token\":\"$(openssl rand -hex 32)\"}}"
kubectl -n mistudio rollout restart deployment/mistudio-mcp

# Read it back when you configure a client.
kubectl -n mistudio get secret mistudio-secrets \
  -o jsonpath='{.data.mcp-auth-token}' | base64 -d
```

The endpoint is the path `/mcp` on whatever host your Ingress serves, for example `https://mcp.mistudio.example.local/mcp`. The shipped Ingress uses streaming-friendly nginx annotations (`proxy-buffering off`, a 3600 s read timeout). Its `host:` entries in `k8s/base/mcp.yaml` are the example defaults of the reference deployment; **change them to your own host names** (and add TLS for a name reachable beyond your LAN) before you apply. The host must resolve to your ingress controller, through your DNS or a hosts-file entry on each client machine, for example `<ingress-ip>  mcp.mistudio.example.local`. No Ingress at all? `kubectl port-forward svc/mistudio-mcp 8765:8765` and use `http://localhost:8765/mcp`.

The bearer token is required on every route; only `/health` answers without it. Serve the endpoint over HTTPS anywhere the traffic leaves a network you trust, because the token travels in a request header.

**Verify:** `curl http://<host>:8765/health` returns the enabled tool categories.

:::warning Network exposure
The server binds `0.0.0.0:8765` so agents on other LAN machines can connect. A bearer token is **always** required on this transport — and that is now enforced, not just stated: `MCP_ALLOW_ANONYMOUS` is honoured on **stdio only**, and the server refuses to start if it is set over HTTP.

Until this was fixed, the flag alone satisfied the startup guard on HTTP, so following the troubleshooting remedy below produced a LAN-reachable **unauthenticated** server — on the page that told you a token was always required.

If your agents run on the same host only, firewall port 8765.
:::

## Connecting a Client

**Claude Code:**

```bash
# Docker Compose (direct port):
claude mcp add --transport http mistudio http://<host>:8765/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"

# Kubernetes (via your Ingress host):
claude mcp add --transport http mistudio https://mcp.mistudio.example.local/mcp \
  --header "Authorization: Bearer $MCP_AUTH_TOKEN"
```

**Any MCP client:** streamable-HTTP transport, URL `http://<host>:8765/mcp`, header `Authorization: Bearer <token>`. A stdio mode exists for local development: `python -m src.mcp_server --stdio` (inside the backend container/venv).

:::tip Give your agent the operating manual
After connecting, paste the [**MCP Agent Instructions**](/advanced/mcp-agent-instructions) into the agent's context (or tell it to fetch that page). It covers the tool catalog semantics, the analyze→group→steer→relabel recipes, guardrail reactions, and the evidence-notes convention — agents perform dramatically better with it.
:::

## Configuration

| Env variable | Default | Purpose |
|--------------|---------|---------|
| `MCP_AUTH_TOKEN` | *(required)* | Bearer token; startup refused if empty |
| `MCP_TOOL_CATEGORIES` | `read,groups,steering,labeling,experiments,profiles,circuits,jlens,jobs,models,probe_monitors` | Which tool categories are exposed; add `admin` to enable destructive deletes |
| `MILLM_API_URL` | *(unset)* | Base URL of a miLLM runtime. Set it to enable the seven cross-product `millm_*` categories (`millm_circuits`, `millm_clusters`, `millm_inference`, `millm_models`, `millm_probes`, `millm_runtime`, `millm_sensing`); they never appear unless this is set |
| `MCP_STEERING_MAX_CONCURRENT` | `2` | Max in-flight agent steering tasks |
| `MCP_STEERING_MAX_NEW_TOKENS` | `512` | Ceiling on generation length for agent steering |
| `MIDATAWORKS_API_URL` | *(unset)* | Base URL of a miDataworks backend. Set it to enable the `dataworks` category (the curated `dataworks_*` proxy); never enabled by default. Not `DATAWORKS_API_URL`, which miDataworks' own MCP server reads |
| `MCP_STEERING_APPROVAL` | `false` | Route agent steering through operator approval (see below) |
| `MCP_MILLM_LOAD_APPROVAL` | `true` | Route an agent's `millm_load_model` through operator approval (see below). On by default and in the shipped manifest: a load evicts the model miLLM serves |
| `MCP_PROBE_GPU_APPROVAL` | `true` | Route an agent's `start_probe_training_run` and `evaluate_probe_monitor` through operator approval (see below). On by default and in the shipped manifest: each holds a GPU for minutes to hours |

Disabled categories simply don't appear in the agent's tool list. Disabling the whole server (omit the compose profile / scale the deployment to 0) never affects the frontend or REST API.

The `circuits` category is **on by default** as of the circuits arc. The five `millm_*` categories are the **cross-product plane** — miStudio discovers and calibrates, miLLM serves — and are gated separately: they require `MILLM_API_URL` and are never enabled by default, even if you list them in `MCP_TOOL_CATEGORIES`.

## Tool Catalog

The authoritative inventory is [`docs/mcp-contract.md`](https://github.com/hitsainet/miStudio/blob/main/docs/mcp-contract.md) in the repo — an **auto-generated, diff-tested** file derived from the live tool registry. The table below mirrors it; the contract wins if they ever drift.

:::tip Agents: call `mistudio_howto` first
`mistudio_howto` is the **"call me first"** tool. A flat table can't carry ordering constraints, GPU-lock contention, id namespaces, or the failure modes that mislead — that tool holds the workflows (feature analysis, the circuit-discovery evidence ladder, the cross-plane document hand-off) and the guardrail reactions. Point your agent at it before it starts circuit or steering work.
:::

| Category | Tools |
|----------|-------|
| **read** (20) | `list_extractions`, `get_extraction_summary`, `list_trainings`, `search_features`, `get_feature`, `get_feature_examples`, `get_feature_token_analysis`, `get_feature_logit_lens`, `get_feature_correlations`, `get_feature_ablation`, `get_feature_nlp_analysis`, `mistudio_howto`. Added 2026-10-19: `get_feature_by_index` (a feature INDEX from a steering result, circuit or allocation → its row id, by SAE or training), and the ids other tools take — `list_datasets`, `get_dataset`, `list_saes`, `get_sae`, `get_training`. Added 2026-10-20 (Feature 036): `get_environment_settings` — the runtime settings and where each value comes from, read-only (no MCP write path; the operator changes them in Settings → Environment). Added 2026-10-21: `get_approval_status` — read a miStudio approval request WITHOUT spending it (pending, approved, denied, expired or consumed; who decided — the operator or the expiry, since the suite has no login — and when; and what to do next). Use it before re-calling a gated GPU tool, because that re-call consumes the approval and starts the job. It used to exist only with `MCP_STEERING_APPROVAL=true` |
| **groups** (7) | `compute_feature_groups`, `get_grouping_status`, `get_feature_groups`, `get_feature_group_members`, `find_features_by_token`, `find_related_features`. `cancel_feature_grouping` stops a running precompute (2026-10-19) |
| **steering** (10) | `steering_status`, `get_steering_mode`, `enter_steering_mode`, `exit_steering_mode`, `steer_compare`, `steer_sweep`, `steer_combined`, `compute_cluster_allocation`, `get_steering_result`, `cancel_steering_task`. Every steering task runs in a fresh worker that loads the model, generates and exits, releasing the GPU; `enter_steering_mode` only starts an idle worker (it loads nothing) and `exit_steering_mode` stops an idle or stranded one |
| **circuits** (32) *(default-on)* | `start_circuit_capture`, `confirm_circuit_capture` (launch THE estimated run, 2026-10-21 — `start_circuit_capture(confirm=true)` is a one-step capture that creates a new run), `list_circuit_captures`, `run_circuit_discovery`, `get_discovery_results`, `run_attribution_pass`, `validate_circuit_edges`, `create_circuit`, `build_circuit_from_discovery`, `update_circuit`, `get_circuit`, `list_circuits`, `delete_circuit` ⚠️, `run_circuit_faithfulness`, `calibrate_circuit_strength`, `reproduce_calibration`, `promote_circuit`, `export_circuit_definition`, `export_circuit_slices`, `import_circuit_definition`, `record_steering_samples`, `get_steering_samples`, `list_validation_manifests`, `get_validation_manifest`, `reproduce_validation`. Cancels for every GPU stage (2026-10-19): `cancel_circuit_capture`, `cancel_circuit_discovery`, `cancel_attribution_pass`, `cancel_edge_validation`, `cancel_circuit_faithfulness`, `cancel_circuit_calibration`, `cancel_steering_samples` — cooperative, so poll the stage's own reader |
| **experiments** (3) | `save_experiment`, `list_experiments`, `get_experiment` |
| **profiles** (4) | `list_cluster_profiles`, `get_cluster_profile`, `save_cluster_profile`, `export_cluster_definition` — durable cluster profiles + portable `mistudio.cluster-definition/v1` export |
| **labeling** (13) | `update_feature_label`, `run_enhanced_labeling`, `get_enhanced_label`, `list_labeling_templates`, `run_labeling_trial`, `get_labeling_trial`, `list_labeling_trials`, `compare_labeling_trials`, `get_labeling_coverage`, `resume_labeling`, `start_labeling_resume_sweep`, `get_labeling_resume_sweep`, `cancel_labeling_resume_sweep` — `run_labeling_trial` and the three that read trials score a template against a fixed feature panel and write NO labels. The last five are resume: `get_labeling_coverage` reports what still needs labeling and WHY the failures failed, `resume_labeling` does one batch of it, and the sweep trio runs many batches back to back. `start_labeling_resume_sweep` requires an explicit `max_batches` — there is no unbounded option, because a sweep is hours of GPU time per batch |
| **jlens** (33) *(default-on)* | `list_jlens_artifacts`, `preview_jlens_repo`, `acquire_jlens_artifact`, `fit_jlens_artifact`, `validate_jlens_artifact`, `publish_jlens_artifact`, `restore_jlens_artifact`, `cancel_jlens_task`, `jlens_readout`, `get_jlens_readout`, `jlens_cost_estimate`, `compute_jlens_band_report`, `get_jlens_band_report`, `record_jlens_gate`, `get_jlens_gate`, `get_jlens_replication_report`, `run_jlens_intervention`, `get_jlens_interventions`, `annotate_jlens_feature`, `jlens_attention_readout`, `start_jlens_annotation_sweep`, `list_jlens_disagreements`, `jlens_decompose`, `get_jlens_decomposition`, `build_jlens_template_lens`, `get_jlens_template_lens`, `create_jlens_watchlist` — the Jacobian-lens substrate. `jlens_decompose` is readout mode (c): a sparse non-negative decomposition by gradient pursuit, reported as explained variance in EXCESS of a size-matched random control. The annotation sweep is CPU-only and bounded (≤ 4,096 features per call); its rows feed the label-disagreement queue. The template lens builds directions for multi-token phrases from forward passes (BR-023) and carries its three documented limitations in every result. Added 2026-10-19: `jlens_probe` / `get_jlens_probe` (probe mode: score NAMED directions without ranking the vocabulary), `jlens_token_check` (is a string one token?), `revalidate_jlens_artifact` (re-validate a STAGED fit without refitting), `get_jlens_artifact_document`, `validate_jlens_document` |
| **probe_monitors** (78) *(default-on)* | `list_probe_monitors`, `get_probe_monitor_report`, `recalibrate_probe_monitor`, `build_probe_definition`, `export_probe_definition`, `publish_probe_definition` — trained probe monitors and their portable `mistudio.probe-definition/v1` export. **Read the rung before the AUROC:** a probe's rung is what its evidence supports (0 trained only, 1 held-out, 2 unseen tasks, 3 judge-compared), and `build_probe_definition` REFUSES below rung 2 unless it is given a reason — which is then written into the exported file, so whoever serves the probe can see the claim was made on a judgement. Two refusals are not waivable: an SAE probe whose dictionary has no HuggingFace home cannot be exported at all (publish the SAE first), and a run still in flight is refused because its rung can still change. `publish_probe_definition` takes NO token — it is resolved server-side from Settings → API Keys, so no credential passes through an agent transcript. `recalibrate_probe_monitor` moves the decision threshold to another false-positive budget with **no GPU and no queue**: a threshold is the (1 − target_fpr) quantile of the negatives the run already saved to disk, so re-cutting it is arithmetic over an array rather than the ~2.6-hour run it used to cost. It **previews by default**, and the preview carries what the candidate bar would do to every evaluation set the probe was measured on. ⚠ It does NOT make a threshold transfer between distributions — a re-cut bar is still a quantile of the same calibration corpus, and on the first shipped probe the five sets' own 1% thresholds spanned 24 points. Read `transfer_proposed.caution` before committing. **The lifecycle before export (2026-10-19):** `register_probe_dataset` (a label-mapped view; pass a miDataworks `dataset_version_manifest` and the split's row count is verified, or the view is refused), `list_probe_datasets` (archived views hidden unless `include_archived`), `get_probe_dataset`, `get_probe_dataset_rows` (a view's rows, paged and projected, each with its mapped label), `archive_probe_dataset` / `unarchive_probe_dataset` (hide or restore a view; there is deliberately no delete tool — permanently deleting an archived view is the operator's, from the Probes screen), `start_probe_training_run` ⚙️, `list_probe_training_runs` (filter `train_dataset_id`), `get_probe_training_run`, `cancel_probe_training_run`, `delete_probe_training_run` ⚠️, `evaluate_probe_monitor` ⚙️ (how rung 2 is reached), `get_probe_evaluation_scores` (per-row scores, paged), `score_probe_monitor`, `score_probe_monitor_batch` (one queued pass per input, at most the deployment's `probe_score_batch_cap`), `get_probe_score_result`, `start_probe_judge_run` ⚙️ (only when the judge endpoint is miLLM; takes a `prompt_id` from the library), `list_probe_judge_runs`, `cancel_probe_judge_run`. **The judge-prompt library (2026-10-21):** `list_judge_prompts`, `get_judge_prompt`, `create_judge_prompt`, `update_judge_prompt` (in place, only while no run has used the version), `new_judge_prompt_version`, `clone_judge_prompt`, `archive_judge_prompt`, `unarchive_judge_prompt`, `delete_judge_prompt` (unused versions only), `preview_judge_prompt` (renders, calls no model), `try_judge_prompt` (at most 10 rows, never loads a model), `export_judge_prompt`, `import_judge_prompt`, `list_midataworks_rubrics_for_judging`, `import_judge_prompt_from_rubric`, and `set_probe_behaviour`. Rung 3 counts a judge run only when its prompt's behaviour key equals the probe's. **The behaviour catalogue (Feature 037):** `list_behaviours` (catalogued behaviours and every uncatalogued key in use), `get_behaviour`, `get_behaviour_checklist` (the sixteen steps from definition to armed in miLLM, a success report and the probe family — exported and armed report every family member, done, partly done, not done or unknown — while the behaviour's status still reads its primary probe), `list_behaviour_versions`, `get_behaviour_history`, `create_behaviour` (adopts every prompt, run and probe already carrying the key — nothing on them changes), `new_behaviour_version`, `set_behaviour_probe_role` (primary, candidate or retired; primary needs a matching key and a confirmed training view), `link_behaviour_view`, `unlink_behaviour_view`, `archive_behaviour`, `unarchive_behaviour`. Deleting a behaviour and confirming a view's positive labels are the operator's. **Behaviour detectors (Feature 040):** `update_probe_dataset_columns` (declare more columns a breakdown may group by; additive only), `get_probe_breakdown` (one evaluation's recall and false positives per value of a declared column, at the decision in force or at a target false-positive rate, with Wilson intervals; groups under the minimum are shown and marked, never dropped), `get_detector_success_report` (judges every success criterion of the behaviour's current version; the same inputs return the same stored report), `list_detector_success_reports`, `read_detector_success_report` (says whether the report is stale, and which input moved), `compare_detectors_across_models` (one report per model, refused when the models were judged on different data). A refusal names the call that unblocks it in `next`. **Probe families (Feature 041):** `get_probe_family` (a behaviour's primary and candidate probes on one model — `model` is required when its probes span several — with each member's facts, flags where a member differs from the family's majority, and coverage of every declared sub-question; a missing one carries the `start_probe_training_run` or `register_probe_dataset` call that fills it). **Blend evaluation (Feature 041):** `evaluate_probe_blends` (fits blends of the family on its blend-fit view with grouped cross-validation, calibrates each candidate on the shared calibration sample and measures it on every test view; CPU only, returns a report id at once and refuses at the call — `member_scores_missing` names the exact `evaluate_probe_monitor` calls, with `sample_rows` for a calibration view), `get_blend_report` (status, `stale` computed on read, and the ranked candidates with Wilson intervals, per-subgroup figures, criterion verdicts and the members' own figures; a blend has no rung), `list_blend_reports`, `get_blend_report_scores` (one candidate's per-row values on one view), `rerun_blend_report`, `cancel_blend_evaluation`, `set_suggested_blend` and `get_suggested_blend` (advice for an application — it binds nothing and miLLM never reads it; the history keeps who marked it and why). Pruning a report's per-row values is the operator's. **Family scoring and export (Feature 041):** `score_probe_family` and `score_probe_family_batch` (one queued task per input, every member scored with one model load, a blend report's candidates applied with its recorded parameters; members × inputs is capped by the deployment's `probe_score_batch_cap`), `get_probe_family_score_result`, `export_probe_family` (all or nothing: each member's single-export route, sha256 and size, or every blocker with the call that clears it) and `publish_probe_family` (one Hugging Face commit for the family; the token is resolved server-side). `score_probe_monitor_batch` now sends one request and the cap is miStudio's. **Session spot-checks (Feature 038):** `start_probe_spot_check` ⚙️ (only when the judge is miLLM: live chats wait behind every judge call), `get_probe_spot_check`, `list_probe_spot_checks`, `cancel_probe_spot_check`, `get_spot_check_availability` (whether miLLM serves the recorded-session routes). The judged turn text and the disagreement export stay with the operator; agents read the report. ⚙️ = holds a GPU for minutes to hours and **waits for operator approval** (`MCP_PROBE_GPU_APPROVAL`, default on): the first call files the request, the re-call with its `approval_request_id` consumes it and submits exactly that payload |
| **models** (6) *(default-on)* | `list_models`, `get_model`, `get_model_architecture`, `download_model`, `cancel_model_download`, `delete_model` ⚠️ — the weights miStudio RUNS ON. No load/unload: miStudio loads per task and has no resident model |
| **jobs** (5) | `get_task_status`, `list_gpus` — the GPU UUIDs every `gpu` parameter accepts. Added 2026-10-21: `get_gpu_status` (every card miStudio can see, or one by UUID: memory used / free / total, utilisation, compute processes, miStudio's GPU lease on the card and the jobs that recorded it), `list_active_jobs` (every queued or running miStudio job of every kind in one call; `errors` names a source that could not be read, which is never "nothing running"), and `gpu_preflight` (one card by UUID) — "is this card free?" across the suite: miStudio's card and jobs, miLLM's resident model, lease and per-card memory (when `MILLM_API_URL` is set) and miForge's pending and running runs (when `MIFORGE_API_URL` is set). It answers `clear`, `busy` or `uncertain` with the reasons; an unconfigured sibling is reported `configured: false` and never makes it fail. Cards are named by UUID; an index is refused |
| **admin** (2) *(off by default)* | `delete_experiment` ⚠️, `delete_extraction` ⚠️ — destructive |

**Cross-product plane** — the seven `millm_*` categories drive a live miLLM runtime and only appear when `MILLM_API_URL` is set:

| Category | Tools |
|----------|-------|
| **millm_circuits** (16) | `millm_import_circuit`, `millm_list_circuits`, `millm_export_circuit`, `millm_delete_circuit` ⚠️, `millm_activate_circuit`, `millm_deactivate_circuit`, `millm_circuit_status`, `millm_set_circuit_intensity`, `millm_circuit_claims`, `millm_release_circuit_claims`, `millm_circuit_sensing_enable`, `millm_circuit_sensing_disable`, `millm_circuit_sensing_status`, `millm_circuit_sensing_events`, `millm_circuit_sensing_event`, `millm_circuit_sensing_clear` ⚠️ |
| **millm_clusters** (7) | `millm_import_cluster`, `millm_list_clusters`, `millm_export_cluster`, `millm_activate_cluster`, `millm_deactivate_cluster`, `millm_hub_search`, `millm_cluster_hub_definitions` (the filenames a Hub repo publishes) |
| **millm_models** (12) | `millm_list_models`, `millm_get_model`, `millm_preview_model_repo`, `millm_download_model`, `millm_cancel_download`, `millm_load_model`, `millm_unload_model`, `millm_delete_model` ⚠️ — the weights miLLM SERVES. It holds ONE model resident, so `millm_load_model` EVICTS the current one and waits for operator approval (see Guardrails). Leases: `millm_acquire_lease`, `millm_renew_lease`, `millm_release_lease`, `millm_lease_status` pin the resident model for one holder (one lease per server, ends at expiry or any restart; the `lease_id` is returned once and travels only in the `X-miLLM-Lease` header; no approval — a lease loads nothing) |
| **dataworks** (18) | `dataworks_list_datasets`, `dataworks_get_version`, `dataworks_get_version_manifest`, `dataworks_list_detector_sets`, `dataworks_get_detector_set`, `dataworks_start_label_run`, `dataworks_build_version_files`, `dataworks_publish_version`, `dataworks_send_detector_set`, `dataworks_delete_version` ⚠️, `dataworks_job_status`, `dataworks_cancel_job`, `dataworks_get_approval_status` — a curated proxy to miDataworks (requires `MIDATAWORKS_API_URL`). **miDataworks holds every approval**: publishing, a detector-set send (one approval for the whole send), deleting a version and labelling past its agent window answer `{"approval_id", "status": "pending", …}`; the operator approves in miDataworks, which runs the stored request once. Poll `dataworks_get_approval_status`; nothing here approves or re-submits. Every request carries `X-Dataworks-Agent: agent:mistudio-mcp` as the "who"; no tool takes a credential. When miDataworks is down: `{"unavailable": "dataworks", …}`. The send's read-back (2026-10-19): `dataworks_list_detector_sends`, `dataworks_get_detector_send` (status and steps), `dataworks_resume_detector_send` (a new job for the same send; recorded steps never repeat), `dataworks_refresh_probe_results`, `dataworks_get_probe_results` |
| **millm_inference** | `millm_score_chat`, `millm_score_completion`, `millm_generate` — score the next token, or generate (with inline `steering`, `seed`, `response_format` or `return_sae_activations` passed verbatim; `steering_state` reads `unsteered` only when miLLM's `X-miLLM-Steering` header is exactly `none`), against the model miLLM SERVES, never swapping it: every request sends `X-miLLM-Load-Policy: refuse` (a non-resident model answers `{"refused": "model_not_resident", "next": "millm_load_model"}`) and `X-miLLM-Strict: true` (an ignored field is a 400). Results carry miLLM's provenance headers verbatim (`millm_headers`). Synchronous calls are capped (32 conversations / prompts, 1,024 generated tokens); larger work goes to the batch tools: `millm_upload_batch_file` (rows → a JSON Lines file; ≤ 10,000 rows, ≤ 16 MiB, unique `custom_id`s), `millm_submit_batch`, `millm_batch_status`, `millm_cancel_batch`, `millm_batch_results` (100 rows per page). A batch never loads or swaps a model. Added 2026-10-19: `millm_embeddings` (`/v1/embeddings`, at most 128 inputs, never steered), `millm_batch_lease` (hand your lease to a batch waiting on `lease_unavailable`), and `millm_score_completion(scoring_precision=...)` with `"native"` or `"float32"` — float32 is REFUSED, never downgraded: `{"refused": "scoring_precision_unavailable", "reason"}`. Wrong-name answers are structured too: `model_not_generative`, `ambiguous_model_name` (send the tagged `repo:QUANT` name) and, on download, `model_name_taken` (give a `custom_name`). Added 2026-10-23 (miLLM 038): `millm_generate` and `millm_score_chat` lift the response body's `millm_probe_verdicts` to the top of the result, unchanged (and leave it in `result`) — one `mi.probe.result/v1` object per armed probe × window, including the ones that said nothing (`outcome` other than `completed`, `verdict: null`); `[]` when none is armed or miLLM predates 038, and never beside a refusal or an `unavailable` |
| **millm_probes** (15) | `millm_import_probe`, `millm_list_probes`, `millm_arm_probe`, `millm_recalibrate_probe`, `millm_disarm_probe`, `millm_probe_status`, `millm_probe_events`, `millm_score_probes` — `millm_score_probes` scores stored inputs with imported probes and **records nothing** (no events, nothing armed changes; `verdict: null` means the probe said nothing, never false). The rest run a probe monitor trained here against miLLM's live traffic. Arming passes four gates (armed-probe limit, model identity, evidence rung, parity); a probe below rung 2 needs `acknowledge_below_rung2`. A probe **records and reports** — nothing here stops or alters a generation `millm_recalibrate_probe` moves an **already-imported, possibly armed** probe's decision bar in place: it re-cuts the threshold here from the negatives the run already saved — no GPU — and pushes the new `decision` block to miLLM, which updates the row, the stored definition and the live runtime shape together. ⚠ It is the only way to change an imported probe's bar without losing its history: the alternative is disarm → delete → import, which assigns a new probe id and cascade-deletes every recorded verdict. It changes the BAR, never the detector — `on_conflict=replace` is still refused, and miLLM refuses any payload that could carry a detector or a cut it cannot match to the stored `provenance.probe_id`.. Added 2026-10-19: `millm_get_probe` (with its stored definition), `millm_delete_probe` ⚠️ (takes its verdict history; refused while armed), `millm_probe_parity` (re-run parity without arming — after a reload), `millm_probe_hub_search`, `millm_probe_hub_definitions`, `millm_clear_probe_events` ⚠️ (scope REQUIRED). `millm_get_probe_event` ⚠ returns ONE verdict WITH its decoded context — **user prompt content** (operator decision 2026-10-19); the list tool carries none. Added 2026-10-23 (miLLM 038): `millm_probe_events` filters by `result_id` (one event) and `trace_id` (a request named by a telemetry backend); its rows, the event detail and `millm_score_probes` results pass the `mi.probe.result/v1` fields (`result_id`, `outcome`, `reason`, `decision`, detector identity, `persistence`) through unchanged; `millm_score_probes(export_results=true)` asks miLLM to export the per-input results, which it does only when its deployment permits (the `export` block says) |
| **millm_runtime** (7) | `millm_status`, `millm_list_profiles`, `millm_activate_profile`, `millm_deactivate_profile`, `millm_set_intensity`. Added 2026-10-21: `millm_get_environment` and `millm_get_environment_history` — miLLM's runtime settings and their change history, read-only (secrets only as set / not set); no MCP tool changes a miLLM setting, the operator does in miLLM's Settings → Environment. `millm_status` also carries miLLM's `telemetry` block (038) when that miLLM reports it: export state and counters per signal |
| **millm_sensing** (7) | `millm_sensing_enable`, `millm_sensing_disable`, `millm_sensing_status`, `millm_sensing_events`, `millm_sensing_config`, `millm_sensing_clear` ⚠️ (scope REQUIRED), `millm_get_sensing_event` ⚠ (one event with its decoded context — **user prompt content**) |

⚠️ = destructive or irreversible.

Long-running operations follow the platform's async pattern: the start tool returns a task id, and the agent polls `get_task_status` / `get_steering_result`. Task ids are durable database records, so they survive agent reconnects.

### Reading results (2026-10-21)

- **Lists arrive as one array too.** A tool that returns a list still sends one content block per item, so existing clients keep working. The whole list is also in `structuredContent.result`, so a client can read one JSON array instead of joining blocks. An empty list is a single `[]` block, never zero blocks, so "nothing matched" can no longer be mistaken for a dropped response.
- **Peak GPU memory per stage.** `get_probe_training_run` returns `environment.gpu_memory`: for each stage, `peak_allocated_mib` and `peak_reserved_mib`, measured with the peak reset at the start of the stage. It also carries the run's overall peaks and the card's `gpu_uuid`, so the next run can be sized. `measured: false` with null values means the model ran on no CUDA device. Judge runs call an LLM endpoint and use no miStudio GPU.
- **Live progress (2026-10-21).** `get_probe_training_run` and `list_probe_training_runs` return `progress_detail` beside `progress`: the `stage`, `done`/`total` in its `unit` (batches, probes, sets), `current` (a probe's epochs, an evaluation set's batches), `rate_per_second` and `eta_seconds` for the stage. `list_probe_judge_runs` returns it for a judge run as `rows_done` of `rows_total` across every set, the current `set_name` with `set_rows_done`/`set_rows_total`, `rate_per_second`, `eta_seconds` and `parse_failures` so far, updated every few seconds. Null until a run reports; rate and ETA stay null until measured. In a judge run's `metrics.per_set.<id>.at_probe_fpr`, `reachable: false` means the judge cannot run at the probe's false-positive rate (a yes/no judge has one operating point) — `recall` is then null, not 0, and `reason` states the judge's own rate and recall.
- **Precision of a model.** `list_models` and `get_model` show `quantization` (the storage format: FP16 means unquantized 16-bit weights) beside `loads_at` (the dtype it runs in, usually the checkpoint's own bfloat16) and a one-line `precision_note`. A run that records `model_dtype: bfloat16` for an FP16 row does not contradict it.

### Probe monitors: running a detector on live traffic

A probe monitor is a linear detector trained here — a vector, a threshold, and a record of how well
it actually worked. `millm_probes` imports one into miLLM, arms it, and reads what it reported.

Two things about arming are worth knowing before an agent tries it:

- **Identity cannot be waived.** A probe's weights are a direction in one specific model's residual
  space. Read in another model's space they produce numbers that are plausible, stable and about
  nothing, so `millm_arm_probe` refuses with `PROBE_MODEL_MISMATCH` and names *every* field that
  differs — which tells you whether the wrong model is loaded or the wrong probe was imported.
- **Below rung 2 needs explicit consent.** The refusal is `UNVALIDATED_PROBE`, and it is resolved by
  passing `acknowledge_below_rung2=true`, not by retrying. That acknowledgement is recorded
  separately from the one inside the definition: whoever exported a weak probe and whoever arms it
  against live traffic are not necessarily the same, and only the second is choosing to monitor
  with it.

Read `rung_language` verbatim rather than composing a phrase from `rung`. Rung 3 reads
"detects on unseen tasks, **compared with** a judge" — compared with, not better than. On the
reference run here the judge won.

A quiet armed probe always says why, in `paused_reason`. Silence would read as "nothing detected",
which is a claim no probe made.

#### Probe identity: digests, `next`, and the push check (Feature 039)

- **Every probe states its identity.** `list_probe_monitors` and `get_probe_monitor_report` carry the
  built definition's `detector_digest`, `decision_digest` and `decision_revision`, with
  `identity_current` (is that decision the one in force?), `identity_stale_reason` and
  `identity_source` (`built`, `recomputed`, `not_recorded`). Null is "not recorded", never a guess.
- **Join back from a digest.** `list_probe_monitors(decision_digest="<decision digest>")` (or
  `detector_digest=`) returns the probes that state it, each with
  `matched: {by, source: current|history|last_built, revisions: [...]}` — so an event judged at an old
  revision is still traced to its bar. A malformed digest is refused before any query.
- **Re-cut with the decision you read.** `recalibrate_probe_monitor`'s preview states both ends'
  digests and revisions, `detector_unchanged: true` and `unchanged`. Commit with
  `expect_decision_digest=<the preview's current.decision_digest>`; if the decision moved meanwhile the
  commit is refused `decision_changed` (409) with `expected`, `current` and a `next` preview. An
  identical re-cut returns `committed: false, reason: "unchanged"`.
- **`next` lists.** A commit, a `built_stale` build and a stale checklist step return `next`: tool
  calls, each `{tool, arguments, why}`, runnable as they stand. An argument only you can supply (a
  below-rung-2 acknowledgement) is listed under `needs` and never filled in for you.
- **`millm_recalibrate_probe` checks before it pushes**, and posts nothing on any refusal:
  1. it reads miStudio's decision in force (`{"unavailable": "mistudio"}` if it cannot;
     `decision_not_formable` if miStudio cannot write the bar as a decision);
  2. your bar must **equal** that decision field by field — else `decision_not_current`, naming each
     field with yours and the one in force;
  3. a `decision_digest`, `decision_revision` or `calibration_id` you pass must agree — else
     `decision_identity_mismatch`; omitted ones are filled from miStudio;
  4. miLLM's copy must be the detector miStudio built — `detector_mismatch` when miLLM states a
     different digest under `mistudio.probe-identity/1` (re-import it); a miLLM that states none, or
     another algorithm, is pushed to with `detector_unverified: true` and the reason;
  5. miLLM's OpenAPI says whether its request takes the identity fields (cached
     `MCP_MILLM_OPENAPI_TTL_SECONDS`, default 60);
  6. one POST carrying miStudio's **whole** decision, with `calibration_id` = the decision digest.

### Circuits: discovery, calibration, and recording

The `circuits` category exposes the full mechanistic-interpretability pipeline. Two tools from the current arc are worth calling out:

- **Calibration** (`calibrate_circuit_strength` → `reproduce_calibration`): a **two-detector usable-band search**. It finds the ONSET where steering starts to bite (output-drift, no judge needed) and the CORRECTNESS CLIFF where the model starts producing fluent-but-false output (an LLM judge scores generated neutral-topic falsifiable probes), bisects between them, and clamps the served dial to `[onset, cliff]`. It is a **badge, not a gate** — it annotates the circuit, it does not block serving. A judge too weak to grade its own probes reports `judge_unreliable` rather than a false `no_band`.
- **Steered Transcript Recorder** (`record_steering_samples` → `get_steering_samples`): an **instrument, not a judge**. It records `(dial, prompt, unsteered, steered)` transcripts for a circuit, a cluster, or an ad-hoc feature set, so a stronger model can analyze the run *afterwards*. It never scores anything itself.

Both write **manifests** — self-contained, reproducible records — and both have a `reproduce_*` tool that re-runs from the manifest and reports a delta verdict, so a claim is reproducible rather than a one-off.

## Guardrails & Provenance

- **Label provenance:** agent label edits carry `label_source: mcp_agent`, so you can always tell agent work from human work.
- **Protected labels:** aqua-starred features (completed enhanced labels) return a 409 when an agent tries to edit their name/category/description; the agent must pass an explicit `override_protected` flag. Notes remain appendable — the documented convention is `[MCP <date>] evidence: experiment <id> — <summary>`.
- **Steering limits:** concurrency cap and generation-length ceiling are enforced before the GPU is touched.
- **Operator-approval mode:** with `MCP_STEERING_APPROVAL=true`, agent steering calls become pending requests instead of running. They appear in the **agent-approvals banner at the top of every panel** with Approve/Deny buttons; on approval the backend submits the stored request itself, and the agent's poll picks up the resulting task id. Steering approvals never expire.
- **Model loads wait for approval** (`MCP_MILLM_LOAD_APPROVAL`, on by default). Loading a model on miLLM evicts the one it serves, so an agent's `millm_load_model` comes in two shapes:
  1. **First call, no id:** a request appears in the banner (*"Load miLLM model #7 — evicts the resident model"*) and the tool returns `{"approval_request_id", "status": "pending_approval", "hint"}`. Nothing is loaded.
  2. **Re-call with `approval_request_id`** after you approve: the tool consumes the approval — once, and only for exactly the model you approved — and starts the load. A re-call before you decide answers `pending`; after a denial `denied`; a second re-call `consumed`.
  An approval that is still pending, or approved and not used, **expires after 24 hours** (backend setting `mcp_approval_ttl_hours`); the agent must then ask again. miLLM can still refuse an approved load — a model locked for steering, or a lease held by someone else — and the approval is spent either way. A lease id never enters the approval.
- **Probe training runs and evaluations wait for approval** (`MCP_PROBE_GPU_APPROVAL`, on by default). Each holds a GPU for minutes to hours and every other GPU job queues behind it. The first call files a request in the approvals banner and returns `{approval_request_id, status: "pending_approval"}`; re-call with the same arguments plus `approval_request_id` once approved — the approval is consumed once, for exactly that payload. The banner never shows a request cut short. A probe training run reads in plain words: the model by name, the graphics card by name and identifier (or "Automatic"), the training, evaluation and calibration sets by name with their positive and negative row counts, and the settings the agent chose, with every argument one click away. Any other request lists all of its arguments in full, so it is never approved blind.
- **Audit:** every tool call is logged (tool name, argument digest, status, duration).

## Worked Example: the Analyze → Group → Steer → Relabel Loop

A Claude Code session pointed at the server, instructed with natural language only:

1. *"List extractions and build feature clusters for the newest one"* → `list_extractions`, `compute_feature_groups`, `get_task_status` until complete
2. *"Show me the biggest groups"* → `get_feature_groups(sort_by="size")` — say it finds a 12-member `"love"` group with cohesion 0.81
3. *"What do the members have in common?"* → `get_feature_group_members`, then `get_feature_examples` + `get_feature_logit_lens` per member → agent hypothesizes "expressions of affection"
4. *"Validate that by steering"* → `enter_steering_mode`, `steer_sweep(feature_idx=4821, strength_values=[0, 10, 30])`, `get_steering_result` → steered outputs turn affectionate; baseline doesn't
5. *"Save the evidence and update the labels"* → `save_experiment`, then `update_feature_label(name="expressions of affection", notes="[MCP 2026-07-12] evidence: experiment exp_… — sweep shows dose-dependent affection shift")`
6. Every record — the group index, the experiment, the relabeled features with `mcp_agent` provenance — is immediately visible in the miStudio UI.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Server exits at startup: "MCP_AUTH_TOKEN is required" | Set `MCP_AUTH_TOKEN` in `.env`. **`MCP_ALLOW_ANONYMOUS` will not help here** — it is honoured on the **stdio** transport only, and the server refuses to start over HTTP with it set. That is deliberate: the HTTP port is LAN-reachable and exposes `delete_circuit`, GPU steering and label write-back. |
| 401 from every call | Client isn't sending `Authorization: Bearer <token>`, or tokens don't match |
| Tools error "backend unreachable" | The `mcp-server` container can't reach `backend:8000` — check both are on the same network and the backend is healthy |
| `steer_*` returns a guardrail message | Concurrency cap hit — poll or cancel existing tasks, or raise `MCP_STEERING_MAX_CONCURRENT` |
| Steering tools return `pending_approval` | Approval mode is on — approve the request in the banner above any panel |
| `millm_load_model` returns `pending_approval` | Model-load approval is on (the default) — approve it in the banner, then the agent re-calls with the id |
| `millm_load_model` returns `expired` | The approval sat for more than 24 hours; the agent must call again without the id |

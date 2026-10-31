<!-- GENERATED FILE — DO NOT EDIT BY HAND.

Regenerate with:
    python -c "from src.mcp_server.contract import write_contract; write_contract()"

`backend/tests/unit/test_mcp_contract_generated.py` regenerates this and fails
if it differs from what is committed, so an edit here is reverted by the next
run rather than silently kept.
-->

# miStudio MCP contract

Every tool the miStudio MCP server registers, its category, and the backend
endpoint it calls. Derived from the live registry — see
`src/mcp_server/contract.py`.

**Agents: call `mistudio_howto` instead of reading this.** A table cannot carry
ordering constraints, GPU-lock contention, id namespaces, or the failure modes
that mislead. This document is the inventory; that tool is the guidance.

Categories are gated by `MCP_TOOL_CATEGORIES`. The `millm_*` categories also
require `MILLM_API_URL`, the `dataworks` category `MIDATAWORKS_API_URL`; none of
them is enabled by default.


**333 tools across 20 categories.**


## `admin` (2 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `delete_experiment` ⚠️ | `DELETE /steering/experiments/{…}` | DESTRUCTIVE: permanently delete a saved steering experiment. |
| `delete_extraction` ⚠️ | `DELETE /extractions/{…}` | DESTRUCTIVE: permanently delete an extraction job AND every feature, label, and activation example derived from it. |

## `circuits` (32 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `build_circuit_from_discovery` | `POST /circuit-discovery/{…}/build-circuit` | Build a circuit from a discovery run's candidates — PREFER THIS over create_circuit when the circuit came from discovery. |
| `calibrate_circuit_strength` | `POST /circuits/{…}/calibration` | Calibrate a circuit's usable steering STRENGTH and clamp its served dial to it (Feature 20). |
| `cancel_attribution_pass` | `POST /circuit-discovery/{…}/attribution/cancel` | Stop a running attribution pass (GPU, rung 1). |
| `cancel_circuit_calibration` | `POST /circuits/{…}/calibration/cancel` | Stop a running strength calibration (calibrate_circuit_strength). |
| `cancel_circuit_capture` | `POST /circuit-capture/{…}/cancel` | Stop a circuit activation capture (GPU). |
| `cancel_circuit_discovery` | `POST /circuit-discovery/{…}/cancel` | Stop a circuit discovery run. |
| `cancel_circuit_faithfulness` | `POST /circuits/{…}/faithfulness/cancel` | Stop a running faithfulness test (run_circuit_faithfulness). |
| `cancel_edge_validation` | `POST /circuit-discovery/{…}/validate/cancel` | Stop a running edge validation (GPU, rung 2). |
| `cancel_steering_samples` | `POST /circuits/steering-samples/{…}/cancel` | Stop a steering-sample recorder run (record_steering_samples). |
| `confirm_circuit_capture` | `POST /circuit-capture/{…}/confirm` | Launch the full capture for an ESTIMATED run — the same run, so the estimate and the capture are one row (2026-10-21: confirming by re-calling start_circuit_cap |
| `create_circuit` | `POST /circuits` | Create a circuit from hand assembly. |
| `delete_circuit` ⚠️ | `DELETE /circuits/{…}` | Delete a circuit permanently (its manifests survive — they are first-class records). |
| `export_circuit_definition` | `GET /circuits/{…}/export` | Export the portable circuit-definition JSON (lossless: rungs, edge types, attribution, manifest refs, provenance all travel). |
| `export_circuit_slices` | `POST /circuits/{…}/export-slices` | Export per-layer cluster-definition/v1 slices (BR-014) for today's single-SAE consumers (miLLM). |
| `get_circuit` | `GET /circuits/{…}` | One circuit: members by layer, typed edges with full evidence (statistics, attribution, validation manifest refs), budget, faithfulness, rung + rung_language +  |
| `get_discovery_results` | `GET /circuit-discovery/{…}` | A discovery run + its report (null-model summary, FDR discipline, held-out replication RATE, stage counts, caps, uncovered seeds, lag-0 disclosure) + ranked can |
| `get_steering_samples` | `GET /validation-manifests`<br>`GET /validation-manifests/{…}` | Fetch recorded steering-sample transcripts: per prompt, the unsteered baseline plus the steered output at each recorded dial — the raw material for an Opus mean |
| `get_validation_manifest` | `GET /validation-manifests/{…}` | A validation manifest — the SELF-CONTAINED, reproducible record of a validation run (intervention config, baseline, prompts, seeds, null summary, per-edge effec |
| `import_circuit_definition` | `POST /circuits/import` | Import a mistudio.circuit-definition/v1 document (the BR-013 round-trip). |
| `list_circuit_captures` | `GET /circuit-capture` | List capture runs (status, corpus, layers, split, size, stale flag). |
| `list_circuits` | `GET /circuits` | List circuits with rung + rung_language on every row. |
| `list_validation_manifests` | `GET /validation-manifests` | List validation manifests for a discovery run or a circuit. |
| `promote_circuit` | `POST /circuits/{…}/promote` | Promote a circuit into a loadable multi-layer steering profile — or unpromote it (promoted=false). |
| `record_steering_samples` | `POST /circuits/steering-samples` | Record (dial, prompt, unsteered_output, steered_output) transcripts on the GPU for a circuit, cluster, or ad-hoc feature set — the raw material for a SEPARATE,  |
| `reproduce_calibration` | `POST /circuits/calibration-manifests/{…}/reproduce` | Re-run a calibration from its manifest — the SAME probes and seed — and record a reproduction manifest with the band-delta verdict (does the re-measured onset/s |
| `reproduce_validation` | `POST /validation-manifests/{…}/reproduce` | Re-execute an edge_batch manifest from its payload and compare — the test that a rung-2 claim is reproducible, not a one-off. |
| `run_attribution_pass` | `POST /circuit-discovery/{…}/attribution` | Tier-2 gradient-attribution pass over a discovery run's candidates: re-ranks the shortlist before 017's causal validation and gates rung-1 (attribution_supporte |
| `run_circuit_discovery` | `POST /circuit-discovery` | Mine a completed capture store for candidate cross-layer edges. |
| `run_circuit_faithfulness` | `POST /circuits/{…}/faithfulness` | Faithfulness-test a circuit (rung 3 — the HIGHEST tier): suppress its members and measure how much of the behavior they drive is NECESSARY (ablating them collap |
| `start_circuit_capture` | `POST /circuit-capture` | Start a circuit-capture run (per-token multi-layer SAE activations with FIRST-CLASS positions + error norms). |
| `update_circuit` | `PATCH /circuits/{…}` | Edit a circuit (rename, fix a narrative, drop a bad edge, adjust members, switch granularity) — the agent review loop is not create-only. |
| `validate_circuit_edges` | `POST /circuit-discovery/{…}/validate` | Causally validate the top-K edges of a discovery run (rung 2): suppress the upstream feature, run the model, measure the downstream effect size vs a shuffled-no |

## `dataworks` (18 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `dataworks_build_version_files` | `POST /api/v1/versions/{…}/publish-builds` | Build the version's Parquet files (one per split) — the step before publishing. |
| `dataworks_cancel_job` | `POST /api/v1/jobs/{…}/cancel` | Cancel a job you started. |
| `dataworks_delete_version` | `DELETE /api/v1/versions/{…}` | Delete (tombstone) a version. |
| `dataworks_get_approval_status` | `GET /api/v1/approvals/{…}` | Where a held action stands, verbatim from miDataworks: `pending`, `executing`, `executed` (with `result_kind`, `result_id`), `failed` (with `error`), `rejected` |
| `dataworks_get_detector_send` | `GET /api/v1/detector-sends/{…}` | One send's STATUS: its snapshot, checks, recorded steps (publish, download, register per role) and the run-request skeleton for training in miStudio. |
| `dataworks_get_detector_set` | `GET /api/v1/detector-sets/{…}` | One detector set as miDataworks reports it: its roles and the version behind each. |
| `dataworks_get_probe_results` | `GET /api/v1/detector-sets/{…}/results` | A stored probe-results snapshot for the set: the probes miStudio trained from the send, their evaluations and rungs, as miDataworks recorded them. |
| `dataworks_get_version` | `GET /api/v1/versions/{…}` | One dataset version: state, row counts per split, recipe, labelers and caveats. |
| `dataworks_get_version_manifest` | `GET /api/v1/versions/{…}/handoff-manifest` | The version's handoff manifest (`midataworks.dataset-version/v1`) — the document miStudio verifies when a probe dataset is registered with `dataset_version_mani |
| `dataworks_job_status` | `GET /api/v1/jobs/{…}` | A miDataworks job's state, progress and result. |
| `dataworks_list_datasets` | `GET /api/v1/datasets` | miDataworks' datasets, paged — step 1 of the loop: find the data. |
| `dataworks_list_detector_sends` | `GET /api/v1/detector-sets/{…}/sends` | Every send of a detector set, newest first, with its state and job. |
| `dataworks_list_detector_sets` | `GET /api/v1/detector-sets` | Detector sets — the four-role bundles (train, in-distribution test, out-of- distribution eval, calibration negatives) a probe is trained and judged on. |
| `dataworks_publish_version` | `POST /api/v1/publishes` | Publish a built version to the Hugging Face Hub. |
| `dataworks_refresh_probe_results` | `POST /api/v1/detector-sets/{…}/results/refresh` | Have miDataworks READ miStudio's runs, probes and reports for the set's latest completed send into a NEW results snapshot. |
| `dataworks_resume_detector_send` | `POST /api/v1/detector-sends/{…}/resume` | Resume a send that stopped part-way: a NEW job for the SAME send; steps already recorded are never repeated (nothing is published or registered twice). Returns  |
| `dataworks_send_detector_set` | `POST /api/v1/detector-sets/{…}/send` | Send a detector set to miStudio: publish each version to the Hub, download it into miStudio and register every role. |
| `dataworks_start_label_run` | `POST /api/v1/label-runs` | Start labelling a version. |

## `experiments` (3 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_experiment` | `GET /steering/experiments/{…}` | Get one saved experiment with its full stored data. |
| `list_experiments` | `GET /steering/experiments` | List saved steering experiments (paginated, limit ≤ 100). |
| `save_experiment` | `POST /steering/experiments` | Save a steering result as a persistent experiment record. |

## `groups` (7 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `cancel_feature_grouping` | `POST /feature-groups/runs/{…}/cancel` | Stop a running feature-grouping precompute at its next feature (2026-10-19). |
| `compute_feature_groups` | `POST /extractions/{…}/feature-groups/compute` | Start the grouping precompute job for an extraction (background job). |
| `find_features_by_token` | `GET /extractions/{…}/features/by-token` | Features whose top activating token matches. |
| `find_related_features` | `GET /features/{…}/related` | Features related to a seed feature via shared tokens, context overlap, and cached correlations. |
| `get_feature_group_members` | `GET /extractions/{…}/feature-groups/{…}` | Members of one group with current labels, stars, stats, and a context snippet each (labels are live — never stale). |
| `get_feature_groups` | `GET /extractions/{…}/feature-groups` | List feature groups (features sharing a top activating token with similar context). |
| `get_grouping_status` | `GET /extractions/{…}/feature-groups/status` | State of the grouping index: none \| pending \| computing \| completed \| failed, with progress, params, and counts. |

## `jlens` (33 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `acquire_jlens_artifact` | `POST /jlens/acquire` | Adopt a lens someone else fitted. |
| `annotate_jlens_feature` | `POST /jlens/annotate` | Describe an SAE feature in J-space: what it pushes TOWARD. |
| `build_jlens_template_lens` | `POST /jlens/template-lens` | QUEUE a template-lens build: directions for MULTI-TOKEN concepts (BR-023). |
| `cancel_jlens_task` | `POST /jlens/tasks/{…}/cancel` | Stop a running or queued J-space task. |
| `compute_jlens_band_report` | `POST /jlens/band-report` | Measure this model's band profile and derive ITS OWN boundaries. |
| `create_jlens_watchlist` | `POST /jlens/watchlists` | Author a watchlist for miLLM to evaluate per token at inference. |
| `fit_jlens_artifact` | `POST /jlens/fit` | Queue a J-lens fit. |
| `get_jlens_artifact_document` | `GET /jlens/artifacts/{…}/document` | The artifact as a portable `mistudio.jlens-artifact/v1` document — for a consumer that MOUNTS the lens directory. |
| `get_jlens_band_report` | `GET /jlens/artifacts/{…}/band-report` | This model's OWN sensory / workspace / motor boundaries, or null. |
| `get_jlens_decomposition` | `GET /jlens/decompose/{…}` | Poll a queued decomposition. |
| `get_jlens_gate` | `GET /jlens/artifacts/{…}/gate` | The recorded Phase-0 GO / NO-GO / GO-AT-LARGER-SCALE decision, or null. |
| `get_jlens_interventions` | `GET /jlens/artifacts/{…}/interventions` | What this lens has been MEASURED to do, recorded beside the weights. |
| `get_jlens_probe` | `GET /jlens/probe/{…}` | Poll a probe. |
| `get_jlens_readout` | `GET /jlens/readout/{…}` | Poll a queued readout. |
| `get_jlens_replication_report` | `GET /jlens/reports/replication?slug={…}` | The recorded replication report, or null (BR-001). |
| `get_jlens_template_lens` | `GET /jlens/template-lens/{…}` | Poll a template-lens build. |
| `jlens_attention_readout` | `POST /jlens/attention-readout` | Read a layer's attention Q/K/V/O PER HEAD through the logit lens (BR-015). |
| `jlens_cost_estimate` | `GET /jlens/cost-estimate` | Estimate an operation's cost BEFORE committing to it. |
| `jlens_decompose` | `POST /jlens/decompose` | QUEUE a sparse decomposition of the residual into lens directions. |
| `jlens_probe` | `POST /jlens/probe` | QUEUE probe mode: score NAMED directions per layer without ranking the vocabulary. |
| `jlens_readout` | `POST /jlens/readout` | QUEUE a readout of what a model is poised to say per layer and position. |
| `jlens_token_check` | `POST /jlens/token-check` | Is each string a SINGLE token in this model's vocabulary? A lens direction is one unembedding row, so a multi-token name has no direction. |
| `list_jlens_artifacts` | `GET /jlens/artifacts` | List J-lens artifacts present in the mounted registry. |
| `list_jlens_disagreements` | `GET /jlens/annotations` | The label-disagreement QUEUE, worst first (BR-013). |
| `preview_jlens_repo` | `POST /jlens/acquire/preview` | List the files in a repo that could be a J-lens, with sizes. |
| `publish_jlens_artifact` | `POST /jlens/publish` | Publish a validated lens so others can mount it. |
| `record_jlens_gate` | `POST /jlens/gate` | Record the Phase-0 GO / NO-GO decision (BR-003). |
| `restore_jlens_artifact` | `POST /jlens/artifacts/{…}/restore-superseded` | Promote the archived artifact back into service. |
| `revalidate_jlens_artifact` | `POST /jlens/artifacts/revalidate` | Re-validate a STAGED J-lens artifact — work a fit already did — and commit it if it passes, without paying for the fit again. |
| `run_jlens_intervention` | `POST /jlens/interventions` | Run an intervention AND its size-matched control in one pass. |
| `start_jlens_annotation_sweep` | `POST /jlens/annotation-sweeps` | QUEUE a bounded J-space annotation sweep over one SAE's features. |
| `validate_jlens_artifact` | `POST /jlens/artifacts/{…}/validate` | Run the BR-030 validation suite against one artifact. |
| `validate_jlens_document` | `POST /jlens/documents/validate` | Validate a J-space interchange document by its versioned `kind`. |

## `jobs` (5 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_gpu_status` | `GET /operations/gpu-status` | Live state of every GPU miStudio can see (one or many), by UUID: memory used / free / total (MiB), utilisation, compute processes, miStudio's GPU lease on the c |
| `get_task_status` | `GET /task-queue`<br>`GET /task-queue/active`<br>`GET /task-queue/{…}` | Poll background jobs. |
| `gpu_preflight` | `GET /operations/active-jobs`<br>`GET /operations/gpu-status` | "Is this card free for a GPU job?" in one call, across the suite. |
| `list_active_jobs` | `GET /operations/active-jobs` | Every miStudio job that is queued or running, of EVERY kind, in one call: SAE training, extraction, activation extraction, labeling (all three), tokenization, d |
| `list_gpus` | `GET /system/gpu-list` | The GPUs miStudio can place jobs on — the UUIDs and indices every `gpu` parameter accepts (2026-10-19: those descriptions named `/api/v1/system/gpu-list` and no |

## `labeling` (13 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `cancel_labeling_resume_sweep` | `POST /labeling/resume-sweeps/{…}/cancel` | Stop a sweep after its current batch. |
| `compare_labeling_trials` | `GET /labeling/trials/compare/{…}/{…}` | Compare two trials over the SAME panel, per feature. |
| `get_enhanced_label` | `GET /features/{…}/label/enhanced/latest` | Latest enhanced-labeling job + synthesized label for a feature. |
| `get_labeling_coverage` | `GET /labeling/{…}/coverage` | What still needs labeling in an extraction, and why the failures failed. |
| `get_labeling_resume_sweep` | `GET /labeling/resume-sweeps/{…}` | How far a resume sweep has got. |
| `get_labeling_trial` | `GET /labeling/trials/{…}` | One trial's full record: the frozen template, the panel, every label. |
| `list_labeling_templates` | `GET /labeling-prompt-templates` | List labeling prompt templates — the variable a trial tests. |
| `list_labeling_trials` | `GET /labeling/trials` | List trials. |
| `resume_labeling` | `GET /labeling/{…}/coverage`<br>`POST /labeling/panel` | Label ONE batch of the features that still need it. |
| `run_enhanced_labeling` | `POST /features/{…}/label/enhanced` | Trigger two-pass enhanced LLM labeling for one feature (background job; uses the labeling backend configured in Settings). |
| `run_labeling_trial` | `POST /labeling/trials` | Run ONE prompt template over a fixed feature panel. |
| `start_labeling_resume_sweep` | `POST /labeling/{…}/resume-sweep` | Run several resume batches back to back, one Celery task each. |
| `update_feature_label` | `PATCH /features/{…}` | Update a feature's label. |

## `millm_circuits` (16 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_activate_circuit` | `POST /api/circuits/{…}/activate` | Serve a circuit. |
| `millm_circuit_claims` | `GET /api/circuits/claims` | Which circuit holds which LAYER. |
| `millm_circuit_sensing_clear` ⚠️ | `DELETE /api/circuit-sensing/events` | Delete recorded edge observations. |
| `millm_circuit_sensing_disable` | `POST /api/circuit-sensing/{…}/disable` | Disable edge sensing for a circuit. |
| `millm_circuit_sensing_enable` | `POST /api/circuit-sensing/{…}/enable` | Enable edge sensing for a circuit (persists; arms when it serves). |
| `millm_circuit_sensing_event` | `GET /api/circuit-sensing/events/{…}` | One edge observation in full: both endpoints with their layer, feature, position and activation, the observed token lag, and the context window. |
| `millm_circuit_sensing_events` | `GET /api/circuit-sensing/events` | Observed edge firings, newest first. |
| `millm_circuit_sensing_status` | `GET /api/circuit-sensing/status` | Edge-sensing runtime state for the armed circuit. |
| `millm_circuit_status` | `GET /api/circuits/active` | Which circuits are steering RIGHT NOW, as a list. |
| `millm_deactivate_circuit` | `POST /api/circuits/{…}/deactivate` | Stop serving a circuit and release its layer claims. |
| `millm_delete_circuit` ⚠️ | `DELETE /api/circuits/{…}` | Delete an imported circuit permanently. |
| `millm_export_circuit` | `GET /api/circuits/{…}/export` | The circuit's ORIGINAL document, byte-for-byte. |
| `millm_import_circuit` | `POST /api/circuits/import` | Import a `mistudio.circuit-definition/v1` document (INLINE only). |
| `millm_list_circuits` | `GET /api/circuits` | Imported circuits with their evidence rung. |
| `millm_release_circuit_claims` | `POST /api/circuits/claims/release` | Release ONE circuit's stuck layer claims. |
| `millm_set_circuit_intensity` | `PUT /api/circuits/active/intensity` | Dial the active circuit's global lambda. |

## `millm_clusters` (7 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_activate_cluster` | `POST /api/clusters/{…}/activate` | Activate an imported cluster (applies all members at sign x strength x lambda; hard compatibility gate server-side). |
| `millm_cluster_hub_definitions` | `GET /api/clusters/hub/{…}/definitions` | The cluster definitions a Hub repo publishes (2026-10-19) — the `filename` values millm_import_cluster(repo_id=…, filename=…) needs, which before this an agent  |
| `millm_deactivate_cluster` | `POST /api/clusters/{…}/deactivate` | Deactivate a cluster (clears its live steering). |
| `millm_export_cluster` | `GET /api/clusters/{…}/export` | Re-export a cluster's lossless original v1 document (byte-honest — survives unknown additive fields). |
| `millm_hub_search` | `GET /api/clusters/hub/search` | Search public Hugging Face cluster packs (repos tagged mistudio-cluster-definition), optionally narrowed to a base model. |
| `millm_import_cluster` | `POST /api/clusters/hub/import`<br>`POST /api/clusters/import` | Import a mistudio.cluster-definition/v1 into miLLM. |
| `millm_list_clusters` | `GET /api/clusters` | List clusters imported into miLLM: bound state, warnings, current lambda intensity, active flag. |

## `millm_inference` (10 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_batch_lease` | `POST /v1/batches/{…}/lease` | Hand your live lease to a waiting batch so it runs under it (miLLM FR-26.7.7) — the remedy when millm_batch_status reports `waiting_reason: lease_unavailable` b |
| `millm_batch_results` | `GET /v1/files/{…}/content` | Page through a finished batch's output or error file. |
| `millm_batch_status` | `GET /v1/batches/{…}` | A batch's status, row counts, file ids and why it is waiting, if it is. |
| `millm_cancel_batch` | `POST /v1/batches/{…}/cancel` | Cancel a batch. |
| `millm_embeddings` | `POST /v1/embeddings` | Embed texts with the served model (`/v1/embeddings`). |
| `millm_generate` | `POST /v1/chat/completions` | Generate a reply on the served model, reporting steering exactly as miLLM does. |
| `millm_score_chat` | `POST /v1/chat/completions` | Score the next token of one or more conversations on the served model. |
| `millm_score_completion` | `POST /v1/completions` | Score the next token of raw prompts on the served model (`/v1/completions` scoring). |
| `millm_submit_batch` | `POST /v1/batches` | Run an uploaded file as a batch. |
| `millm_upload_batch_file` | `POST /v1/files` | Upload the rows of a batch to miLLM as a JSON Lines file. |

## `millm_models` (14 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_acquire_lease` | `POST /api/models/{…}/lease` | Pin the model miLLM is serving so nobody else's load or request swaps it out. |
| `millm_cancel_download` | `POST /api/models/{…}/cancel` | Stop an in-progress download and clean up its partial files. |
| `millm_delete_model` ⚠️ | `DELETE /api/models/{…}`<br>`GET /api/models/{…}` | Delete a model permanently — the database row and the files on disk. |
| `millm_download_model` | `POST /api/models` | Start downloading a model. |
| `millm_get_approval_status` | `GET /api/approvals/{…}` | One miLLM approval request: its status, who decided, when and why, and what it produced. |
| `millm_get_model` | `GET /api/models/{…}` | One model's full record — status, size, quantization, and download progress if it is still downloading. |
| `millm_lease_status` | `GET /api/models/{…}/lease` | Who holds a model's lease, why and until when — and the last lease that ended. |
| `millm_list_approvals` | `GET /api/approvals` | miLLM's approval queue, newest first, with the count still pending. |
| `millm_list_models` | `GET /api/models` | What models miLLM has, and which one is serving right now. |
| `millm_load_model` | `POST /api/models/{…}/load` | Make a model the one miLLM serves. |
| `millm_preview_model_repo` | `POST /api/models/preview` | Inspect a HuggingFace repo WITHOUT downloading it. |
| `millm_release_lease` | `DELETE /api/models/{…}/lease` | End your lease now, so others may load or swap the model again. |
| `millm_renew_lease` | `POST /api/models/{…}/lease/renew` | Extend your lease: the new expiry is now + ttl_seconds. |
| `millm_unload_model` | `POST /api/models/{…}/unload` | Evict a model from GPU memory, freeing the card. |

## `millm_probes` (32 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_arm_behaviour` | `POST /api/probes/behaviours/{…}/arm` | Arm a behaviour's probe set ALL OR NOTHING: each member runs millm_arm_probe's gates (identity, rung, scope, parity, cost), then the armed-probe backstop and th |
| `millm_arm_probe` | `POST /api/probes/{…}/arm` | Arm a probe. |
| `millm_clear_behaviour_rule_signals` | `DELETE /api/probes/behaviours/{…}/rule-signals` | Detach every rule set from a behaviour in miLLM (the sets stay imported; history keeps the change). |
| `millm_clear_probe_events` ⚠️ | `DELETE /api/probes/events` | Delete recorded probe verdicts — one probe's, or (explicitly) all of them. |
| `millm_delete_probe` | `DELETE /api/probes/{…}` | Delete an imported probe AND its recorded events. |
| `millm_delete_rule_set` | `DELETE /api/rule-sets/{…}` | Delete an imported rule set. |
| `millm_disarm_behaviour` | `POST /api/probes/behaviours/{…}/disarm` | Disarm every armed member of a behaviour's set (or the named ones). |
| `millm_disarm_probe` | `POST /api/probes/{…}/disarm` | Disarm a probe: it stops scoring and its layer hook is removed. |
| `millm_evaluate_rule_set` | `POST /api/rule-sets/{…}/evaluate` | Run one stored set on YOUR text, per expression and window: `fires`, `count`, `outcome`. |
| `millm_get_behaviour_rule_signals` | `GET /api/probes/behaviours/{…}/rule-signals` | A behaviour's attached rule sets in miLLM (and the signals each is narrowed to), with the append-only attachment history. |
| `millm_get_probe` | `GET /api/probes/{…}` | One imported probe: its summary (rung, rung_language VERBATIM, threshold, armed state, parity report) plus the full stored `definition` document. |
| `millm_get_probe_event` | `GET /api/probes/events/{…}` | ⚠ RETURNS DECODED USER PROMPT CONTENT. |
| `millm_get_probe_limit_recommendation` | `GET /api/models/{…}/probe-limits/recommendation` | The model's explained probe-limit recommendation (miLLM contract v1.25, ENH-006 D83): probes may slow each token by at most 5% of the model's own time (`PROBE_S |
| `millm_get_rule_set` | `GET /api/rule-sets/{…}` | One imported rule set: its summary, forms per expression, provenance, whether its own cases still pass on this build, the behaviours it is attached to, what is  |
| `millm_import_probe` | `POST /api/probes/hub/import`<br>`POST /api/probes/import` | Import a probe definition, from a document or from HuggingFace. |
| `millm_import_rule_set` | `POST /api/rule-sets/import` | Validate and store one rule set in miLLM (an identical document returns the existing id). |
| `millm_list_behaviours` | `GET /api/probes/behaviours` | One page of the behaviour sets miLLM holds: per key the imported and armed probes, versions, whether the set is complete (`"partial: n of m armed"` otherwise),  |
| `millm_list_probes` | `GET /api/probes` | Imported probes, with their evidence and their parity report. |
| `millm_list_rule_sets` | `GET /api/rule-sets` | miLLM's imported rule sets, paged by (behaviour key, id). |
| `millm_list_rule_validators` | `GET /api/rule-sets/validators` | Every validator miLLM's rule engine registers (id, version, parameter schema, reasons, outputs, algorithms) from its live registry, and the forms it evaluates ( |
| `millm_measure_probe_capacity` | `POST /api/models/{…}/probe-capacity/measure` | Measure the RESIDENT model's probe capacity on its card(s); the operator approves it in miLLM's Approvals page. |
| `millm_probe_capacity` | `GET /api/models/{…}/probe-capacity` | A model's probe capacity in miLLM: the measured step time and probe cost lines, the configured per-model limits, the EFFECTIVE limits admission uses, the limits |
| `millm_probe_events` | `GET /api/probes/events` | Verdicts, newest first. |
| `millm_probe_hub_definitions` | `GET /api/probes/hub/{…}/definitions` | The probe definitions a Hub repo publishes — `manifest.jsonl` preferred, loose `*.probe.json` files otherwise. |
| `millm_probe_hub_search` | `GET /api/probes/hub/search` | Search HuggingFace for repos tagged `mistudio-probe-definition`. |
| `millm_probe_parity` | `POST /api/probes/{…}/parity` | Re-run PARITY without arming, and store the report: score the definition's own test vectors on the model miLLM has loaded and compare with the scores miStudio r |
| `millm_probe_status` | `GET /api/probes/status` | What every armed probe is doing, and why any of them is NOT scoring. |
| `millm_recalibrate_probe` | `GET /api/probes/{…}`<br>`GET /probe-monitors/probes/{…}/decision`<br>`POST /api/probes/{…}/recalibrate` | Push an already re-cut decision bar onto an imported, possibly ARMED miLLM probe. |
| `millm_request_probe_limits` | `PUT /api/models/{…}/probe-limits` | Ask for a model's probe limits to change in miLLM; the operator approves it in miLLM's Approvals page. |
| `millm_score_probes` | `POST /api/probes/score` | Score stored inputs with imported probes — armed or not — and record nothing. |
| `millm_set_behaviour_rule_signals` | `PUT /api/probes/behaviours/{…}/rule-signals` | Replace a behaviour's attached rule sets. |
| `millm_validate_rule_set` | `POST /api/rule-sets/validate` | Every check an import runs — the vendored schema, the pattern dialect, compilation, the set's own cases — writing nothing. |

## `millm_runtime` (8 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_activate_profile` | `POST /api/profiles/{…}/activate` | Activate a miLLM profile by id — replaces the live steering. |
| `millm_deactivate_profile` | `POST /api/profiles/{…}/deactivate` | Deactivate a miLLM profile (clears live steering when it is the active one). |
| `millm_get_environment` | `GET /api/environment` | miLLM's runtime settings and where each value comes from (miLLM Feature 35): the value in force, the environment's value, `source` (env \| override), the group,  |
| `millm_get_environment_history` | `GET /api/environment/history` | The history of operator changes to miLLM's runtime settings, newest first: key, old and new value, who, why and when. |
| `millm_list_profiles` | `GET /api/profiles` | List miLLM steering profiles (manual rows and imported clusters — source_kind discriminates). |
| `millm_set_intensity` | `PUT /api/clusters/active/intensity` | Set the ACTIVE cluster's persistent intensity dial (lambda). |
| `millm_set_telemetry` | `PUT /api/environment` | Ask to change miLLM's telemetry settings: whether each OpenTelemetry signal is exported, where to (the receiver endpoint), sampling, queue and timing. |
| `millm_status` | `GET /api/health/detailed` | miLLM runtime status in one call: loaded model, attached SAE, inference backend, circuit breakers, and the ACTIVE steering profile (id/name/source_kind/intensit |

## `millm_sensing` (7 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `millm_get_sensing_event` | `GET /api/sensing/events/{…}` | ⚠ RETURNS DECODED USER PROMPT CONTENT. |
| `millm_sensing_clear` ⚠️ | `DELETE /api/sensing/events` | Delete recorded sensing events — one profile's, or (explicitly) all of them. |
| `millm_sensing_config` | `PUT /api/sensing/{…}/config` | SET a cluster's sensing quorum (members that must co-fire for an event) — this tool WRITES config; use millm_sensing_status to read. |
| `millm_sensing_disable` | `POST /api/sensing/{…}/disable` | Disable sensing for a cluster (disarms live if armed). |
| `millm_sensing_enable` | `POST /api/sensing/{…}/enable` | Enable sensing for a cluster (persists; arms live when that cluster is active with an SAE attached). |
| `millm_sensing_events` | `GET /api/sensing/events` | Co-activation events newest-first. |
| `millm_sensing_status` | `GET /api/sensing/status` | Sensing runtime status: armed cluster, quorum/threshold mode, per-request overhead, retention limits, and which clusters have the persistent sensing toggle enab |

## `models` (6 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `cancel_model_download` | `DELETE /models/{…}/cancel` | Stop an in-progress download and clean up its partial files. |
| `delete_model` ⚠️ | `DELETE /models/{…}`<br>`GET /models/{…}` | Delete a model permanently — the database row, and its files on disk via a background cleanup job. |
| `download_model` | `POST /models/download` | Start downloading a model into miStudio's catalogue. |
| `get_model` | `GET /models/{…}` | One model's record — status, quantization (storage format) beside `loads_at` (the dtype it runs in) and `precision_note`, disk size, architecture, and the error |
| `get_model_architecture` | `GET /models/{…}/architecture` | The model's discovered layer structure — layer count, hidden size, and the module paths extraction and steering hook. |
| `list_models` | `GET /models` | What models miStudio has, and whether each is usable. |

## `probe_monitors` (86 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `archive_behaviour` | `POST /probe-monitors/behaviours/{…}/archive` | Archive a behaviour: hidden from the list, roles frozen. |
| `archive_judge_prompt` | `POST /probe-monitors/judge-prompts/{…}/archive` | Hide a version from new runs. |
| `archive_probe_dataset` | `POST /probe-monitors/datasets/{…}/archive` | ARCHIVE a probe dataset view: hide it from listings and from new runs, evaluations and judge baselines. |
| `build_probe_definition` | `POST /probe-monitors/probes/{…}/definition` | Build a portable `mistudio.probe-definition/v1` for a probe. |
| `cancel_blend_evaluation` | `POST /probe-monitors/blend-reports/{…}/cancel` | Stop a queued or running blend evaluation (it stops between folds or candidates; its inputs are kept). |
| `cancel_detector_rules_evaluation` | `POST /probe-monitors/rules-evaluations/{…}/cancel` | Stop a pending or running rules evaluation. |
| `cancel_probe_judge_run` | `POST /probe-monitors/judge-runs/{…}/cancel` | Stop a judge baseline — every remaining row is a call to the served judge model, so stopping one nobody is waiting for saves real work. |
| `cancel_probe_spot_check` | `POST /probe-monitors/spot-checks/{…}/cancel` | Stop a running spot-check between turns; the turns judged so far are kept as a partial run. |
| `cancel_probe_training_run` | `POST /probe-monitors/runs/{…}/cancel` | Request a COOPERATIVE stop: the worker stops at its next stage boundary, so poll `get_probe_training_run` until it reads cancelled. |
| `clone_judge_prompt` | `POST /probe-monitors/judge-prompts/{…}/clone` | Copy a version into a NEW prompt (version 1 under a new name) — the quick way to start a judge for a related behaviour. |
| `compare_detectors_across_models` | `GET /probe-monitors/behaviours/{…}/model-comparison` | The behaviour's detectors on each model side by side: one success report per model (the definition's models, else every model with a probe of it), criteria verd |
| `create_behaviour` | `POST /probe-monitors/behaviours` | Catalogue a behaviour key with version 1 of its meaning and, optionally, its detector data (sub-questions, success criteria, held-out columns, models, calibrati |
| `create_judge_prompt` | `POST /probe-monitors/judge-prompts` | Create version 1 of a NEW judge prompt — how a new classification domain gets a judge. |
| `create_nonsense_text_dataset` | `POST /probe-monitors/nonsense-text` | Generate a NONSENSE-TEXT set and register its view — no GPU, no download (ENH-006 D84). |
| `delete_judge_prompt` | `DELETE /probe-monitors/judge-prompts/{…}` | Delete a version NO run has used (a draft). |
| `delete_probe_training_run` | `DELETE /probe-monitors/runs/{…}` | Delete a probe run, its probes and their artifacts. |
| `evaluate_detector_with_rules` | `POST /probe-monitors/probes/{…}/rules-evaluations` | Judge the behaviour's rule set and this probe TOGETHER on the views: rules alone, probe alone and rules or probe (a row is flagged when a rule matches, or the p |
| `evaluate_probe_blends` | `POST /probe-monitors/blend-reports` | Evaluate blends of a behaviour's probe family the way they would be deployed: fit on the blend_fit view (grouped cross-validation), calibrate each candidate on  |
| `evaluate_probe_monitor` | `POST /probe-monitors/probes/{…}/evaluate` |  |
| `evaluate_rule_signals` | `POST /probe-monitors/rule-signals/evaluate` | Try a rule set on ONE text or conversation, synchronously: `rule_match`, `rule_veto` (a veto holds only when no match fires), `unknown` with its reason (`rule_b |
| `export_judge_prompt` | `GET /probe-monitors/judge-prompts/{…}/export` | A portable `mistudio.judge-prompt/v1` document of one version; importing it elsewhere reproduces the same name/vN and content hash. |
| `export_probe_definition` | `GET /probe-monitors/probes/{…}/definition` | Fetch the built definition as JSON. |
| `export_probe_family` | `GET /probe-monitors/behaviours/{…}/family/export-index` | Export the family, ALL OR NOTHING: each member's built definition with its single-export route, sha256 and size, and the archive's route (a ZIP of the untouched |
| `get_behaviour` | `GET /probe-monitors/behaviours/{…}` | One behaviour: its current meaning and every judge prompt, run, probe (with its role: primary, candidate or retired) and dataset view that uses its key. |
| `get_behaviour_checklist` | `GET /probe-monitors/behaviours/{…}/checklist` | The behaviour's eight-step checklist (defined, data, trained, rung 2, judge prompt, rung 3, exported, armed in miLLM), each step done / not_done / unknown with  |
| `get_behaviour_history` | `GET /probe-monitors/behaviours/{…}/history` | The behaviour's append-only history: role changes, view links, confirmations, probe key changes, archive and delete — who did each (operator or agent), when, an |
| `get_blend_report` | `GET /probe-monitors/blend-reports/{…}` | One blend report: status (queued, running, done, failed, cancelled), `stale` with the reasons (computed on read: a member re-cut, retired or deleted, a new beha |
| `get_blend_report_scores` | `GET /probe-monitors/blend-reports/{…}/scores` | One candidate's per-row values on one view, in split-position order: {position, value, verdict, label}. |
| `get_detector_rules_evaluation` | `GET /probe-monitors/rules-evaluations/{…}` | One rules evaluation: its status and progress while it runs; once completed its stored result, byte-for-byte, with `stale` computed on read (which digest moved  |
| `get_detector_success_report` | `POST /probe-monitors/behaviours/{…}/success-reports` | Does this behaviour's detector meet its own bar? Judges every success criterion of the version at the decision a deployment applies: per criterion the verdict ( |
| `get_judge_prompt` | `GET /probe-monitors/judge-prompts/{…}` | One judge prompt version in full (instructions, template, answer form, hash), with every version of its name and how many runs used each. |
| `get_probe_breakdown` | `GET /probe-monitors/probes/{…}/evaluations/{…}/breakdown` | Recall and false-positive rate of one evaluation, per value of one declared column, at the decision a deployment applies — from stored arrays, no GPU. |
| `get_probe_dataset` | `GET /probe-monitors/datasets/{…}` | One probe dataset view: its mapping, filter, class counts, and — when it was registered with a miDataworks manifest — the `dataset_version` facts that were veri |
| `get_probe_dataset_rows` | `GET /probe-monitors/datasets/{…}/rows` | Read a probe dataset view's ROWS, paged and projected: each row's position in the downloaded split, its raw label, the label the view's mapping gives it (`outco |
| `get_probe_evaluation_scores` | `GET /probe-monitors/probes/{…}/evaluations/{…}/scores` | One evaluation's PER-ROW scores, paged in stored order, with the label and the source dataset's revision (035 XR-2). |
| `get_probe_family` | `GET /probe-monitors/behaviours/{…}/family` | A behaviour's probe family on one model: its primary and candidate probes (retired on request) with each member's facts (precision, render form, scope, layer, r |
| `get_probe_family_score_result` | `GET /probe-monitors/behaviours/{…}/family/score/{…}` | Collect a family score. |
| `get_probe_monitor_report` | `GET /probe-monitors/probes/{…}` | One probe's full report: per-set AUROCs with confidence intervals, the rung WITH ITS WORDING and what would raise it, the dense/SAE counterpart, any judge runs, |
| `get_probe_score_result` | `GET /probe-monitors/probes/{…}/score/{…}` | Collect an offline probe score. |
| `get_probe_spot_check` | `GET /probe-monitors/spot-checks/{…}` | One spot-check: status, progress, the sample spec and what miLLM returned (ids only), and the report — per compared probe the 2 × 2 table (probe fired vs judge  |
| `get_probe_training_run` | `GET /probe-monitors/runs/{…}` | One probe run: status, stage, progress, its config echoed verbatim and the environment it ran in (model precision included). |
| `get_spot_check_availability` | `GET /probe-monitors/spot-checks/availability` | Whether miLLM is connected and serves the recorded-session routes a spot-check needs (miLLM Feature 036: the seeded sample and the turn transcript), and the pla |
| `get_suggested_blend` | `GET /probe-monitors/behaviours/{…}/suggested-blend`<br>`GET /probe-monitors/behaviours/{…}/suggested-blend/history` | The current suggested blend of a behaviour on a model (advisory; `stale` when its report is), and with `include_history` every earlier marking with who made it  |
| `import_judge_prompt` | `POST /probe-monitors/judge-prompts/import` | Store an exported version under its own name and version. |
| `import_judge_prompt_from_rubric` | `POST /probe-monitors/judge-prompts/import-rubric` | Turn a miDataworks rubric into a library prompt (version 1 of `name`). |
| `link_behaviour_view` | `POST /probe-monitors/behaviours/{…}/views/{…}` | Link a probe dataset view to a behaviour explicitly. |
| `list_behaviour_versions` | `GET /probe-monitors/behaviours/{…}/versions` | Every version of a behaviour's definition text, newest first. |
| `list_behaviours` | `GET /probe-monitors/behaviours` | List the behaviour catalogue, one page at a time, in key order: every catalogued behaviour (key, name, meaning, version, archived, how many prompts/runs/probes/ |
| `list_blend_reports` | `GET /probe-monitors/blend-reports` | A behaviour's blend reports, newest first, each with its status and `stale`. |
| `list_detector_rules_evaluations` | `GET /probe-monitors/behaviours/{…}/rules-evaluations` | A behaviour's rules evaluations, newest first, paged (`next_cursor`, `total`). |
| `list_detector_success_reports` | `GET /probe-monitors/behaviours/{…}/success-reports` | The behaviour's stored success reports, newest first: id, version judged, model, overall verdict. |
| `list_judge_prompts` | `GET /probe-monitors/judge-prompts` | List the judge-prompt library: each version's key (name/vN), behaviour, answer form, content hash, whether it is locked (used by a run) and by how many runs. Al |
| `list_midataworks_rubrics_for_judging` | `GET /probe-monitors/judge-prompts/midataworks/rubrics` | miDataworks' rubrics, each marked importable or not (binary and pointwise only). |
| `list_probe_datasets` | `GET /probe-monitors/datasets` | List registered probe dataset views, newest first, with their class counts and role. |
| `list_probe_judge_runs` | `GET /probe-monitors/judge-runs` | List judge baseline runs with status, progress, parse failures and metrics. |
| `list_probe_monitors` | `GET /probe-monitors/probes` | List trained probe monitors with their rung, threshold, validation metrics and identity. |
| `list_probe_spot_checks` | `GET /probe-monitors/spot-checks` | List behaviour spot-checks, newest first, with status, progress, whether a run is partial, and each report (2 × 2 table, agreement, kappa per compared probe). F |
| `list_probe_training_runs` | `GET /probe-monitors/runs` | List probe training runs, newest first, with status and stage. |
| `new_behaviour_version` | `POST /probe-monitors/behaviours/{…}/versions` | A new version of a behaviour's definition. |
| `new_judge_prompt_version` | `POST /probe-monitors/judge-prompts/{…}/versions` | Make the NEXT version of a prompt (same name and behaviour) from an existing one plus changes. |
| `preview_judge_prompt` | `POST /probe-monitors/judge-prompts/preview` | Show the judge's exact messages for a few rows of a view, plus warnings when the instructions do not ask for the shape the answer form reads. |
| `publish_probe_definition` | `POST /probe-monitors/probes/{…}/publish` | Publish the definition, a merged manifest and a README to HuggingFace. |
| `publish_probe_family` | `POST /probe-monitors/behaviours/{…}/family/publish` | Publish every member's definition in ONE Hugging Face commit (manifest merged, never replaced; one README section per probe). |
| `read_detector_success_report` | `GET /probe-monitors/success-reports/{…}` | One stored success report exactly as it was built, with `stale` computed now: which inputs moved (a newer behaviour version, a removed probe or view) and the ca |
| `recalibrate_probe_monitor` | `POST /probe-monitors/probes/{…}/recalibrate` | Move a probe's decision threshold to another false-positive budget. |
| `register_probe_dataset` | `POST /probe-monitors/datasets` | Register a label-mapped VIEW over a downloaded dataset — what a probe run trains on, is evaluated on, or calibrates its threshold from. |
| `request_code_rule_approval` | `POST /code-rules/{…}/approval-requests` | Ask the operator to approve ONE operator code rule (by its hash) to run in miStudio's isolated runner. |
| `rerun_blend_report` | `POST /probe-monitors/blend-reports/{…}/rerun` | Run a report's request again against today's members and definition; the new report and the old one link to each other. |
| `score_probe_family` | `POST /probe-monitors/behaviours/{…}/family/score-batch` | Score ONE input with the whole family — a queued forward pass with ONE model load; returns a task id. |
| `score_probe_family_batch` | `POST /probe-monitors/behaviours/{…}/family/score-batch` | Score several inputs with the family: one queued task per input (an input is never half scored). |
| `score_probe_monitor` | `POST /probe-monitors/probes/{…}/score` | Score ONE input offline with a probe — a queued forward pass; returns a task id (202). |
| `score_probe_monitor_batch` | `POST /probe-monitors/probes/{…}/score-batch` |  |
| `set_behaviour_probe_role` | `PUT /probe-monitors/behaviours/{…}/probes/{…}/role` | Set a probe's role in its behaviour. |
| `set_probe_behaviour` | `PATCH /probe-monitors/probes/{…}/behaviour` | Say what a probe detects, then its rung is recomputed: only judge runs whose prompt judges this same behaviour count toward rung 3. |
| `set_suggested_blend` | `PUT /probe-monitors/behaviours/{…}/suggested-blend` | Record ADVICE for an application: which blend of armed probes to use, sliced from a report. |
| `start_probe_judge_run` | `POST /probe-monitors/judge-runs` |  |
| `start_probe_spot_check` | `POST /probe-monitors/spot-checks` | Spot-check a behaviour's live probes against a library judge (Feature 038): miLLM returns a seeded, stratified sample of RECORDED session turns (transcripts are |
| `start_probe_training_run` | `POST /probe-monitors/runs` |  |
| `try_judge_prompt` | `POST /probe-monitors/judge-prompts/try` | Send a few rows through the judge and return each reply, how it was read, and whether it agrees with the row's label. |
| `unarchive_behaviour` | `POST /probe-monitors/behaviours/{…}/unarchive` | Undo archive_behaviour. |
| `unarchive_judge_prompt` | `POST /probe-monitors/judge-prompts/{…}/unarchive` | Make an archived version available to new runs again. |
| `unarchive_probe_dataset` | `POST /probe-monitors/datasets/{…}/unarchive` | Unarchive a probe dataset view: it is listed again by `list_probe_datasets` and can be used by new training runs, evaluations and judge baselines. |
| `unlink_behaviour_view` | `DELETE /probe-monitors/behaviours/{…}/views/{…}` | Remove an explicit view link (the history keeps it). |
| `update_judge_prompt` | `PATCH /probe-monitors/judge-prompts/{…}` | Edit a version IN PLACE. |
| `update_probe_dataset_columns` | `PATCH /probe-monitors/datasets/{…}/columns` | Declare columns on a registered view: grouping columns (for example a kind or source column) so its evaluations can be broken down by them; the text column rule |
| `validate_rule_set` | `POST /probe-monitors/rule-sets/validate` | Check a rule set with miStudio's own engine: the vendored rule-set schema, every pattern compiled under the memory cap, and every case the set carries. Answers  |

## `profiles` (4 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `export_cluster_definition` | `GET /cluster-profiles/{…}/export` | Export a profile as portable `mistudio.cluster-definition/v1` JSON — the consumer-neutral artifact (never contains secrets or local paths). |
| `get_cluster_profile` | `GET /cluster-profiles/{…}` | Get one cluster profile: members with tuned strengths, budget snapshot (013 allocation), narrative, provenance. |
| `list_cluster_profiles` | `GET /cluster-profiles` | List saved cluster profiles (newest first), optionally filtered by SAE or a name/display-token substring. |
| `save_cluster_profile` | `POST /cluster-profiles` | Save a cluster profile (durable, decoupled from recomputable groups). |

## `read` (20 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `get_approval_status` | `GET /mcp/approvals/{…}` | Read a miStudio operator-approval request WITHOUT spending it: status (pending \| approved \| denied \| expired \| consumed), who decided and when, when it was cons |
| `get_dataset` | `GET /datasets/{…}` | One dataset: source, revision, splits, size and status. |
| `get_environment_settings` | `GET /environment` | miStudio's runtime settings and where each value comes from. |
| `get_extraction_summary` | `GET /extractions/{…}` | Get one extraction job's detail: SAE, status, feature counts, config. |
| `get_feature` | `GET /features/{…}` | Full feature detail: label, category, description, notes, statistics, star state. |
| `get_feature_ablation` | `GET /features/{…}/ablation` | STATISTICAL-ESTIMATE ablation impact for a feature (method= "statistical_estimate" in the response) — scored from activation frequency/magnitude/consistency, NO |
| `get_feature_by_index` | `GET /saes/{…}/features/by-index/{…}`<br>`GET /trainings/{…}/features/by-index/{…}` | Resolve a feature INDEX to its feature row id, so the index a steering result, circuit or allocation names can be fed to get_feature / get_feature_examples. Ind |
| `get_feature_correlations` | `GET /features/{…}/correlations` | Features correlated with this one (token overlap + activation-stat similarity). |
| `get_feature_examples` | `GET /features/{…}/examples` | Top activating examples with per-token activations — the primary evidence for what a feature detects. |
| `get_feature_logit_lens` | `GET /features/{…}/logit-lens` | Logit-lens: vocabulary tokens this feature promotes/suppresses when active. |
| `get_feature_nlp_analysis` | `GET /features/{…}/nlp-analysis` | Stored NLP analysis (prime-token distribution, POS tags, context patterns, semantic clusters), if it has been computed for this feature. |
| `get_feature_token_analysis` | `GET /features/{…}/token-analysis` | Aggregated token statistics for a feature (ranked token frequencies). |
| `get_sae` | `GET /saes/{…}` | One SAE: model, layer, hook, width, architecture, normalisation and where it came from (trained, downloaded, or published to HuggingFace). |
| `get_training` | `GET /trainings/{…}` | One SAE training run: model, dataset, hyperparameters, status, progress and final metrics. |
| `list_datasets` | `GET /datasets` | List miStudio's downloaded datasets. |
| `list_extractions` | `GET /extractions` | List feature-extraction jobs (the entry point for feature analysis). |
| `list_saes` | `GET /saes` | List miStudio's SAEs (sae_xxxxxxxx) — the id steering, circuit and feature-index tools take. |
| `list_trainings` | `GET /trainings` | List SAE training runs (id, model, status, hyperparameters, metrics). |
| `mistudio_howto` | — | START HERE for circuit/steering work — workflow guidance an agent cannot infer from tool signatures. |
| `search_features` | `GET /extractions/{…}/features` | Search/filter features within an extraction. |

## `steering` (10 tools)

| Tool | Endpoint | Summary |
|---|---|---|
| `cancel_steering_task` | `DELETE /steering/async/task/{…}` | Cancel a running steering task and free its guardrail slot. |
| `compute_cluster_allocation` | `POST /steering/cluster-allocation` | Compute the principled starting strength allocation for steering a CLUSTER of features (Feature 013) — or a MULTI-LAYER circuit (Feature 015). |
| `enter_steering_mode` | `POST /steering/enter-mode` | Start an idle steering worker process. |
| `exit_steering_mode` | `POST /steering/exit-mode` | Stop the steering worker(s). |
| `get_steering_mode` | `GET /steering/mode` | Is a steering worker process running (per card in multi-GPU mode)? A running worker holds no model between tasks: each generation loads its model and SAE, and t |
| `get_steering_result` | `GET /steering/async/result/{…}` | Poll an async steering task. |
| `steer_combined` | `POST /steering/async/combined` | Apply ALL selected features simultaneously in one generation pass (synergy testing). |
| `steer_compare` | `POST /steering/async/compare` | Run a steering comparison: 1-4 features, each generating steered output side-by-side with an unsteered baseline. |
| `steer_sweep` | `POST /steering/async/sweep` | Dose-response sweep: one feature generated at each strength value (e.g. |
| `steering_status` | `GET /steering/status` | Steering service health, circuit-breaker state, and reset path. |

⚠️ = destructive or irreversible.

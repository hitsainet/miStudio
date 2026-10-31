---
sidebar_position: 3
title: "Data Model"
description: "Core database tables and how they relate"
---

# Data Model

miStudio stores all metadata in PostgreSQL (heavy artifacts — weights, activations — live on the filesystem, referenced by path). This page maps the tables and their relationships. Every table in the ORM is listed here — enforced by `test_data_model_doc_covers_every_table`, which diffs this page against `Base.metadata` and fails when one is missing. (It previously claimed to be "verified against the ORM models" while omitting nine tables, including one its own ER diagram draws.)

```mermaid
erDiagram
    datasets ||--o{ dataset_tokenizations : "per (model, max_length)"
    models ||--o{ dataset_tokenizations : tokenizer
    models ||--o{ trainings : "base model"
    models ||--o{ activation_extractions : "stage 1"
    trainings ||--o{ training_metrics : "per step"
    trainings ||--o{ checkpoints : ""
    trainings |o--o{ external_saes : "import"
    external_saes ||--o{ features : "extracted from"
    trainings ||--o{ features : "extracted from"
    features ||--o{ feature_activations : "top examples"
    features ||--o{ enhanced_labeling_jobs : ""
    external_saes ||--o{ cluster_profiles : "tuned members"
    circuit_capture_runs ||--o{ circuit_discovery_runs : "soft ref"
    circuit_discovery_runs |o--o{ circuits : "soft ref"
    circuits |o--o{ validation_manifests : "circuit_id"
    circuit_discovery_runs |o--o{ validation_manifests : "discovery_run_id"
```

## Pipeline tables

### `datasets`
`id UUID` · `name`, `source`, `hf_repo_id` · `status` (`downloading|processing|ready|error`) · `progress`, `error_message` · `raw_path`, `num_samples`, `size_bytes` · token-filter settings (`tokenization_filter_enabled`, `tokenization_filter_mode`, `tokenization_junk_ratio_threshold`) · `metadata` JSONB

### `dataset_tokenizations`
`id` = `tok_{dataset}_{model}_{maxlen}` · FKs `dataset_id`, `model_id` · `max_length`, `tokenized_path`, `tokenizer_repo_id`, `vocab_size`, `num_tokens`, `avg_seq_length` · `status` (`queued|processing|ready|error`) · `celery_task_id` · punctuation-filter options · block order: `shuffled`, `shuffle_seed` · **UNIQUE `(dataset_id, model_id, max_length)`** — one tokenization per model+length

`shuffled` and `shuffle_seed` are nullable with no default, and **NULL means not recorded** rather
than false. Tokenizations written before these columns existed are in corpus order, and say so by
declining to claim otherwise. `shuffle_seed` is NULL both when nothing was recorded and when a row
was recorded as unshuffled — read `shuffled` to tell those apart. `0` is a legitimate seed, not a
sentinel.

### `models`
`id` = `m_{uuid}` · `name`, `repo_id` · `architecture` (family) + `architecture_config` JSONB (discovered dims) · `params_count` · `quantization` (`FP32|FP16|Q8|Q4|Q2`) · `status` (`downloading|loading|quantizing|ready|error`) · `file_path`, `quantized_path` · `memory_required_bytes`, `disk_size_bytes` · `celery_task_id`

### `trainings`
`id` = `train_{uuid}` · FK `model_id` · `dataset_id` **and** `dataset_ids` JSONB (multi-dataset) · `extraction_id` and `extraction_ids` JSONB (cached-activation training) · `status` (`pending|initializing|running|paused|completed|failed|cancelled`) · `progress`, `current_step`, `total_steps` · `hyperparameters` JSONB · live stats (`current_loss`, `current_l0_sparsity`, `current_fvu` (legacy), `current_fvu_centred`, `current_dead_neurons`, `current_learning_rate`) · `evaluation` JSONB (see below) · `celery_task_id`, `checkpoint_dir`, `finalized_from_step`, `error_traceback`

### `training_metrics`
`id BigInteger` · FK `training_id` · `step`, `timestamp`, `layer_idx` (NULL = aggregated series; `-1 - layer` = held-out rows) · `hook_type` (the SAE's hook on per-SAE and held-out rows; NULL on aggregated rows and on rows written before migration `d5a1f3c7e9b2`) · `loss`, `loss_reconstructed`, `loss_zero`, `l0_mean`, `l0_sparsity`, `l1_sparsity`, `dead_neurons`, `fvu` (legacy), `fvu_centred`, `learning_rate`, `grad_norm` (the total gradient norm **before** clipping, so a value above `grad_clip_norm` is one the clip brought down; written on per-SAE rows only, and only on steps that clipped — NULL when `grad_clip_norm` is unset, on aggregated and held-out rows, and on every row written before 2026-09-16), `gpu_memory_used_mb`, `samples_per_second` · **UNIQUE index `(training_id, step, layer_idx, COALESCE(hook_type, ''))`** (aggregated rows, with `layer_idx` NULL, are not constrained)

### `external_saes`
`id` = `sae_{uuid}` · `source` (`huggingface|local|trained`) · `status` (`pending|downloading|converting|ready|error|deleted`) · HF fields (`hf_repo_id`, `hf_filepath`, `hf_revision`) · `training_id` FK (nullable — external SAEs have none) · `model_name`, `model_id` FK, `layer`, `hook_type` · `n_features`, `d_model`, `architecture` · `format` (e.g. `community_standard`) · `local_path`, `file_size_bytes` · `sae_metadata` JSONB

## Feature tables

### `features`
`id` = `feat_{training}_{neuron}` or `feat_sae_{sae}_{neuron}` · FKs `training_id` / `external_sae_id` (one nullable), `extraction_job_id`, `labeling_job_id` · `neuron_index` · `name`, `category`, `description`, `notes` · `label_source` (`auto|user|llm|local_llm|openai|enhanced_llm`) · indexed stats: `activation_frequency`, `interpretability_score`, `max_activation`, `mean_activation` · `is_favorite`, `star_color` (`yellow|purple|aqua|null`) · `example_tokens_summary` JSONB, `nlp_analysis` JSONB, `nlp_processed_at`

**Adjudication state** — the OUTCOME of a labeling attempt, kept separate from
the verdict. `label_status` (`pending|in_progress|succeeded|failed|skipped`),
`label_attempts`, `label_error`, `label_error_at`, `labeled_at`,
`label_prompt_fingerprint` (sha256 of the prompt-affecting template fields) and
`label_model` (the judge that produced the verdict), and the judge's own
self-assessment of that verdict: `label_fit_count` (`"F/N"`, how many of the
examples shown its hypothesis explains) and `label_confidence`
(`high|medium|low`). Both are NULL when not recorded — labels written before
they existed, or by a template that does not ask. A reported fit below 60%
turns the label into `uninterpretable` whatever the judge named it; the
self-assessment is kept so that override can be audited. `category` carries the
semantic answer; `label_status` carries what happened. Conflating them is what
let 16,824 failed features look finished.

**Two activation frequencies, and they mean different things.**
`activation_frequency` is counted from the examples extraction emits, i.e.
**after** the junk-token filter has discarded any (document, feature) pair whose
peak landed on a filtered token — so it reads "fraction of samples where the
peak was on a kept token". It feeds the dead-feature gate and the steering
auto-baseline, whose constants were fitted against it, and is therefore left
unchanged. `activation_frequency_true` and `activation_count_true` are the same
quantity counted **before** the filter; both are NULL on extractions that
predate the measurement, which is the honest encoding — those runs did not take
it. The ratio between the two is how much the filter distorts a given feature.

### `feature_activations`
Composite PK `(id, feature_id)`, **range-partitioned by `feature_id`** for scale · `sample_index`, `max_activation` · `tokens` + `activations` JSONB (per-token values) · context split: `prefix_tokens` / `prime_token` / `suffix_tokens`, `prime_activation_index` · `dataset_id` — which corpus of the mixture this example came from

`sample_index` is a **global** offset into the concatenated mixture, so it stays unique across
corpora within one extraction. `dataset_id` is provenance, not disambiguation: it is what makes
per-corpus statistics possible. It is nullable, and NULL on rows written before the column
existed.

## Circuit & cluster tables

The circuit subsystem records discovered cross-layer structures and the tuned clusters that steer them. Heavy artifacts (capture event stores) live on the filesystem under `/data/circuit_captures/{id}/`; these tables hold the metadata and contract-shaped JSONB snapshots.

### `circuits`
`id` = `crc_{hex12}` · `name`, `narrative` (markdown), `granularity` (`feature|cluster`) · contract-shaped JSONB snapshots mirroring `mistudio.circuit-definition/v1`: `saes`, `members`, `edges`, `budget`, `faithfulness`, `calibration` (the two-detector usable-band result — onset, correctness cliff, clamped intensity range; IDL-37), `discovery` (provenance) · `rung` (denormalized min-over-edges evidence rung) · `promoted` (a promoted circuit **is** the loadable multi-layer steering profile) · `version` (optimistic-concurrency — a stale write 409s) · `discovery_run_id` (soft ref) · `model_id`, `model_hf_id` (cross-instance-stable) · faithfulness lifecycle (`faithfulness_status` `pending|running|completed|failed`, `faithfulness_task_id`) · **calibration lifecycle** (`calibration_status` `pending|running|completed|failed`, `calibration_task_id`) · `schema_version` · `created_at`, `updated_at` · index `(promoted, rung)`

### `circuit_capture_runs`
`id` = `cap_{hex12}` · `status` (`pending|estimating|running|completed|failed|cancelled`), `progress` (0–100), `error_message` · `manifest` JSONB (mirrors the on-disk `manifest.json`: corpus refs, per-layer SAE/threshold, split, counts, SAE fingerprints, optional attention capture) · `store_path`, `events_total`, `bytes_total` · `stale` (flagged, not deleted, when a referenced SAE changes) · `celery_task_id` · `created_at`, `updated_at`

### `circuit_discovery_runs`
`id` = `dsc_{hex12}` · `capture_run_id` (soft ref) · `status`, `progress`, `error_message` · `params` JSONB (granularity, seeded/open mode, seed refs, `s_min`, null shuffles, FDR `q`, cohesion floor) · `report` JSONB (null summary, FDR discipline, held-out replication rate, counts-by-stage, attribution envelope) · `candidates` JSONB (both orderings — coactivation-only and attribution-re-ranked; cap 2000) · **separate attribution lifecycle** (`attribution_status`, `attribution_progress`, `attribution_error`, `attribution_task_id`) · **separate validation lifecycle** (`validation_status`, `validation_progress`, `validation_error`, `validation_task_id`) · `celery_task_id` · `created_at`, `updated_at`

### `validation_manifests`
`id` = `vman_{hex12}` · `kind` (`edge_batch|faithfulness|reproduction|calibration|steering_samples`) · self-contained: soft parent refs `discovery_run_id`, `circuit_id`, `parent_manifest_id` (a reproduction → its source) · `payload` JSONB (everything needed to REPRODUCE the run — intervention config, baseline, prompts, seeds, cfg, null summary, per-edge/member values, `metric_id`) · `created_at` · indexes on `discovery_run_id` and `circuit_id`

:::note Manifests are the record, not a live join
A manifest carries no drift-prone live references — reproduction is the correctness test. Manifest ids travel into the portable circuit contract as `validation_manifest_ref`, so an exported circuit's causal claims point at a reproducible record.
:::

### `steering_record_runs`
In-flight marker for the Steered Transcript Recorder (records `(dial, prompt, unsteered, steered)` transcripts for a strong model to analyze after the run). A record job loads the GPU like calibration, so it must be visible to the single-GPU guard — but it may target a cluster or ad-hoc feature set with no circuit row, so the marker lives here.
`id` = `srr_{hex12}` · `status` (`pending|running|completed|failed`), `task_id` · `artifact_kind` (`circuit|cluster|features`), `artifact_ref` (circuit id / cluster-profile id; `null` for a bare feature set) · `manifest_ref` (the `steering_samples` manifest it produced) · `error` · `created_at`, `updated_at`

### `cluster_profiles`
`id` = `clp_{hex12}` · `sae_id` FK → `external_saes` (`RESTRICT` on delete; nullable for imported-unbound profiles) · `model_id`, `extraction_id` (soft context), `source_group_id` (soft ref — grouping tables are recomputable and must not destroy tuned work) · `name`, `narrative` (markdown), `display_token` · `members` JSONB (per-member `feature_idx`, `label`, `similarity`, `activation_frequency`, `max_activation`, `strength`, `sign`, `pinned`) · `budget` JSONB (`B`, `B_dir`, `G`, `f_eff`, `formula_id`, `constants`, `intensity`, `intensity_range`) · `schema_version` · `imported_from` JSONB (import provenance) · `created_at`, `updated_at` · index `(sae_id, name)` · serializes to `mistudio.cluster-definition/v1`

## Job & support tables

| Table | Purpose |
|-------|---------|
| `activation_extractions` | Stage-1 raw-activation capture jobs (per model) |
| `extraction_jobs` | Stage-2 SAE→feature extraction jobs |
| `labeling_jobs` | Bulk labeling runs |
| `enhanced_labeling_jobs` | Per-feature two-pass labeling runs |
| `labeling_trial_runs` | Prompt-template A/B results over a fixed feature panel. A trial writes labels **here only** — never onto `features` — so comparing template variants cannot destroy the labels under comparison. `panel_id` is content-addressed (`sha256(extraction_job_id \| sorted feature ids)`), so equal ids prove an identical panel and a comparison can refuse a mismatch. `labeling_job_id` is `ON DELETE SET NULL`: deleting the job that produced a measurement must not delete the measurement. |
| `labeling_jobs.skip_adjudicated` | Whether a whole-extraction job skips features that already carry a verdict. **Defaults to false** — a full relabel, which is what pressing Label has always done. Defaulting it true would silently repurpose an existing control, so the choice is visible and opt-in. |
| `labeling_resume_sweeps` | A multi-batch labeling resume, one batch per Celery task. Finishing a 53k-feature extraction is ~27 batches of 2000 at ~8 s/feature; a task that looped over them would exceed the 10 h soft limit and, on an `acks_late` queue, strand its message for the full 12 h visibility timeout — so a sweep is a ROW plus a task that re-enqueues itself. `max_batches` is NOT NULL with no default, because an open-ended sweep is a request to spend an unknown number of GPU-hours. Progress counters come from what each batch actually WROTE, never from batch count: a sweep whose every batch failed must not report itself complete. |
| `neuronpedia_export_jobs` | ZIP export jobs |
| `neuronpedia_pushes` | Direct-push jobs (`push_{sae}_{ts}`, status `queued|preparing|pushing|completed|failed`) |
| `steering_experiments` | Saved steering results |
| `prompt_templates` | Steering prompt sets — `prompts` is a JSONB *array* (multi-prompt) |
| `training_templates`, `extraction_templates`, `labeling_prompt_templates` | Saved configurations |
| `app_settings` | Key-value settings; sensitive values AES-256-GCM encrypted |
| `environment_overrides` | Operator overrides of the runtime environment settings (Settings → Environment, Feature 036). One row per overridden setting, value as JSON; no row means the environment's value is in force. Only the 13 allow-listed settings are ever written, and a row that no longer validates is ignored. |
| `environment_setting_changes` | The append-only history of those overrides: every set and reset, with the old and new value, who, when and an optional reason. A database trigger refuses UPDATE and DELETE. |
| `task_queue` | Persistent background-job records backing the Monitor page |
| `gpu_leases` | Which job holds each GPU (multi-GPU Phase 3). One row per card, keyed by its UUID, so the primary key itself refuses a second holder; a job split across cards holds one row per card, taken in one transaction or not at all. A lease expires unless its holder renews it, so a worker killed mid-job frees its card. Job rows record the card in `gpu_uuid` (the first card) and, for a model split across GPUs, every card in `gpu_uuids`. |

:::note Two extraction tables, on purpose
`activation_extractions` (model → raw activations) and `extraction_jobs` (SAE → features) are distinct pipelines that share a word — see [The Extraction Pipeline](/concepts/extraction-pipeline).
:::

### `trainings.finalized_from_step`

Nullable integer. Set when a stopped run was finalized from a checkpoint, and
`NULL` for runs that completed normally.

A finalized run has `status = 'completed'` so the SAE import path unlocks, but
`progress` and `current_step` keep their real values. This column is what
distinguishes a salvaged run from one that reached `total_steps`, and it drives
the "Finalized early @ N" badge. See
[Training Lifecycle & Checkpoints](/core-workflow/training-lifecycle).

### Two FVU columns: `fvu` and `fvu_centred`

`training_metrics.fvu_centred` and `trainings.current_fvu_centred` hold the standard FVU,
`Σ‖x − x̂‖² / Σ‖x − μ‖²` with `μ` the mean of each dimension, computed on raw activations. This is
the value the UI shows as **FVU**.

`fvu` and `current_fvu` keep the meaning they have always had: `var(x − x̂) / var(x)` with a single
mean over every element. On activations with large constant dimensions this legacy value reads
lower than the standard one (0.26 against 0.32 at LFM2.5-1.2B layer 11). Rows written before
2026-09-15 have only the legacy value, and their `fvu_centred` is `NULL`. Nothing was backfilled,
because the activations needed to compute it are not stored with the row.

### `trainings.evaluation`

Nullable JSONB, written by the post-run evaluation and by `POST /trainings/{id}/evaluate`. It is a
single document, replaced on each run:

| Key | Content |
|-----|---------|
| `status` | `pending` · `running` · `completed` · `failed` · `skipped` · `cancelled` (stopped between batches) |
| `progress` | While running: `stage` (`loading_model`, `means`, `cross_entropy`), `batches_done`, `batches` |
| `stop_requested_at` | When a Stop of the evaluation was requested; absent otherwise |
| `stop_requested_by` | The control that requested it: `stop` or `stop_and_finalize` |
| `reason` | Why it failed or was skipped |
| `trigger` | `post_run` or `rerun` |
| `task_id`, `started_at`, `completed_at`, `updated_at` | Bookkeeping |
| `config` | `token_budget`, `batch_tokens`, `seed` |
| `sources[]` | Per tokenization: `label`, `dataset_path`, `total_rows`, `rows_read_by_training`, `unseen_rows`, `weight`, `blocks_evaluated` |
| `tokens`, `predicted_tokens`, `ce_base` | Real tokens read, next-token positions scored, and CE with the model untouched |
| `layers[]` | Per residual SAE: `layer`, `hook_type`, `ce_spliced`, `ce_mean_ablated`, `ce_zero_ablated`, `ce_delta`, `loss_recovered_vs_mean`, `loss_recovered_vs_zero`, `kl`, `l0`, `fvu_centred`, `fvu_legacy` |
| `all_layers_spliced` | `layers`, `ce`, `ce_delta`, `kl` with every SAE spliced at once |
| `skipped_saes[]` | SAEs that cannot be spliced (non-residual hooks, transcoders), with the reason |

An undefined number, such as a loss recovered whose denominator is zero, is stored as `null`, never
`NaN`. The evaluation reads only rows at or above each extraction's `max_samples` — see
[Model-Cost Evaluation](/core-workflow/sae-training#model-cost-evaluation). Before this column
existed, the spliced-CE step wrote its numbers into `training_metrics` rows with
`layer_idx = -1000 - layer`, in columns meant for other quantities. It no longer writes those rows.


## J-space annotations (Feature 025, files `024_*`)

### `jlens_feature_annotations`
One SAE feature read through the lens, per **surface** (`decoder` — what the feature writes — or `encoder` — what it reads; separate readouts on purpose, because one readout of both hides the transformation the feature performs) and per `lens_type`. Unique on `(sae_id, feature_index, surface, lens_type)`: re-annotating replaces the row. `lens_kurtosis` is the geometric field and `workspace_class` the behavioural one (`unknown` without a band report for the model, never guessed). `compared_label` is the label **as it was when compared**, so a later relabel cannot make an old `disagreement_score` describe a label it never saw. The label-disagreement queue is a query over this table — `GET /jlens/annotations`, or `jlens_disagreement=true` on the extraction's feature list — and nothing here edits a label: `evidence_rung` is always 0.

## Probe monitors (Feature 032)

A probe monitor is a linear readout over one decoder layer's `resid_post`
activations — `score_t = w·ẑ_t + b` — followed by a combining rule that turns
per-token scores into one score for the whole input. Five tables carry it.

| Table | Purpose |
|---|---|
| `probe_monitor_datasets` | A label-mapped **view** over a downloaded dataset — not a copy. It records the input and label columns, the `label_mapping` (`{raw value: positive \| negative \| excluded}`), an optional keyword filter, and the `counts` that mapping produced. `dataset_version` (JSONB, nullable) records what a miDataworks dataset-version manifest verified at registration — version identity, split, manifest and downloaded row counts, repository, commit, manifest SHA-256; NULL means registered without one. One download serves several views, which is the normal case: a repo's `training` config trains a probe while its `*_balanced` configs evaluate it. `role` separates `train`, `eval`, `calibration`, `blend_fit`, `everyday_text` (ordinary prose with no positives of any behaviour; rung 2 needs few false alarms on one at the shipped bar) and `nonsense_text` (generated text with no meaning — word salad, random characters, repeated tokens, garbled code; likewise, ENH-006 D84), and they are not interchangeable — a calibration set supplies **negatives** for the false-positive-rate threshold and is not labelled for the concept at all, so it may never map a value to `positive`. `distribution` marks an evaluation set as in- or out-of-distribution, which is what rung 2 turns on. `archived_at` / `archived_by` / `archive_reason` (all NULL = not archived; 2026-10-21): archive hides a view from the list and from new runs, and only an archived view can be deleted permanently. |
| `probe_monitor_dataset_purges` | One record per permanently deleted (purged) probe dataset view: who (`purged_by`), when, why (`reason`), the view's full definition as it was (`removed`), the rows removed, and what was deliberately kept (`kept` — the downloaded dataset). No foreign key, because the row it describes is gone. **Append-only**: ORM events and a database trigger refuse UPDATE and DELETE. |
| `probe_monitor_runs` | One GPU job: render → pooled capture across the swept layers → layer selection → token capture at the chosen layers → train each rule → calibrate → evaluate → rung. `layer_selection` holds the **full** layer × pooling grid with each cell's validation AUROC, not only the winner, so a near-tie is visible; `environment` holds what FR-15 requires to reproduce the run (model revision, dtype, chat-template hash, dataset revisions, seeds, per-stage wall times). `progress_detail` (JSONB, since 2026-10-21) is the live reading behind `progress` — stage, units done and total, the item inside the stage, rate and ETA; NULL until the run reports. |
| `probe_monitors` | One trained probe: a `layer`, a `rule`, and a `variant` (`dense` or a k-sparse `sae` probe over selected SAE features). `threshold` is the operating point calibrated at `target_fpr`, and it is **not** a quality number — `realised_fpr` records what that threshold actually spends and `threshold_source` says whether it came from a held-out calibration set or from validation negatives, which are different claims. `streamable` says the rule has an exact online form. `rung` is a `ProbeRung`, never a causal claim; `rung_detail` (JSONB, nullable, 2026-10-29) is what the last rung recompute decided beside it — the next step, the everyday-text and nonsense-text false-alarm verdicts and the catch rate at the shipped bar on unseen tasks (ENH-006 D84). Feature 039 adds three columns — see *Detector and decision digests* below. |
| `probe_monitor_evaluations` | One probe scored on one probe dataset. **A refusal is a row:** below 20 examples of either class the row is written with `status` refused and the reason in `metrics`, never omitted and never given a 0.5 — a missing row reads as "not run yet", and 0.5 reads as "measured, and chance". |
| `probe_monitor_judge_runs` | An LLM monitor over the same evaluation sets, as the **baseline a probe is compared against**. Rung 3 is "judge-compared", which claims the two were measured on the same data and says nothing about the judge being right. `parse_failures` is a column because a judge whose replies stop parsing yields a silently smaller sample; over the configured share the run refuses rather than reporting the remainder. Since 2026-10-21 each run records its judge prompt as columns — `prompt_id` (the library version), `prompt_version` (`name/vN`) and `prompt_hash` — and the worker refuses to judge if the prompt no longer hashes to the recorded value. `progress_detail` (JSONB, since 2026-10-21) holds the live reading: rows judged and planned across every set, the current set, rate, ETA and parse failures so far. |
| `probe_judge_prompts` | The judge-prompt library: one row per **version** of a prompt (`name`, `version` unique). Each declares the `behaviour` it judges, the instructions, a user-message template over fixed placeholders, the longest row shown, and an `answer_form` (a rating with a "present from" value, yes/no, or a label set) that fixes how replies are read. `content_hash` covers everything sent or used to read the reply. `locked_at` is set by the first judge run that uses the version; after that a database trigger refuses any change except to `title`, `description`, `behaviour_label` and `archived_at`, and refuses deletion. Rung 3 counts a judge run only when the prompt's `behaviour` equals the probe's (`probe_monitors.behaviour`, set from `probe_monitor_runs.behaviour`). `source` is `seed` for a row loaded from a seed file (`backend/src/data/judge_prompt_seeds/`, or `JUDGE_PROMPT_SEED_DIR`, whose rows record the file name in `source_ref`); seed rows have a fixed id, `jp_<name>_v<N>` (hyphens as underscores; a hash of the key when that would exceed 36 characters). |

### Detector and decision digests (Feature 039)

Three nullable columns on `probe_monitors` (migration `8a9569207fed`), and fields inside existing JSONB:

| Where | What |
|---|---|
| `decision_snapshot` (JSONB) | The contract `decision` block of the bar in force, written in the same transaction as the bar by every bar writer (the run's calibrating stage, a re-cut), plus `digest`, `formed_at`, `threshold_source`, and — when the bar cannot be written as a contract decision (e.g. a calibration dataset with no HuggingFace id) — `decision: null` with `stale_reason: decision_not_formable` and the `detail`. NULL = never formed (a probe cut before 039; formed on first need). A build copies this block and refuses `decision_snapshot_stale` if the row's bar no longer forms it. |
| `decision_digest` (varchar 64, indexed) | The snapshot's digest — the decision in force. NULL when not formable or not yet formed. |
| `built_detector_digest` (varchar 64, indexed) | The detector digest of the last built definition. Kept when that definition is invalidated (consumers holding the file still hold it). |
| `definition_build.resolved` | Gains `detector_digest`, `decision_digest`, `decision_revision`, `identity_source` (`built`, or `recomputed` for a file built before 039, computed once on first read), `rung`. |
| `definition_build.invalidated` | Gains `cause` (`decision`, `rung`, `build_race`), `was_sha256` and `was_identity` (the identity the dropped file carried). |
| `calibration_history[*].decision_digest` | The digest of each revision's decision, written by 039 re-cuts (revision 1, seeded on the first re-cut, carries the replaced bar's). Entries written before 039 have no key — they read null and are never back-filled. |
| `published[*]` | Gains `detector_digest`, `decision_digest`, `decision_revision` of the copy published. |

The portable document gains an optional, last `identity` block
(`{algorithm: "mistudio.probe-identity/1", detector_digest, decision_digest, decision_revision}`); the
projection and canonical form are fixed by `docs/schemas/probe-identity-cases.json`, which miLLM
vendors byte-identically.

### The behaviour catalogue (Feature 037)

A **behaviour** is what a probe detects and what a judge prompt judges (`pii`, `high-stakes`). The
judge-prompt library already writes it as a plain key on `probe_judge_prompts`, `probe_monitor_runs` and
`probe_monitors`; the catalogue adds the record of each key **without changing those columns** — a record
belongs to a behaviour exactly when its `behaviour` column equals the catalogue key (a join, not a foreign
key), so a key with no catalogue row stays valid and is listed as "uncatalogued".

| Table | Purpose |
|---|---|
| `probe_behaviours` | One catalogued behaviour, keyed by the slug (`key`). `current_version` names the definition text in force; `archived_at` / `archived_by` are the only hand-set state. Its status (draft, validated, monitoring, unknown) is computed on read, never stored. |
| `probe_behaviour_versions` | The definition text, one row per version: `name`, `description`, `positive_meaning`, `negative_meaning`, `target_fpr` (the false-positive rate its probes are calibrated to). **Append-only**: a meaning edit is a new version; a database trigger refuses UPDATE, and DELETE while the behaviour exists. An exported probe carries the version current when it was exported. |
| `probe_behaviour_probe_roles` | A probe's role in its behaviour: `primary`, `candidate` (the default, with no row) or `retired`. At most one primary per behaviour, enforced by a partial unique index. A probe can be primary only when its training view's positive labels were confirmed as this behaviour. |
| `probe_behaviour_view_links` | An explicit link from a behaviour to a probe dataset view, with its `use` (train, eval, calibration). A view whose own `behaviour` column is the key is listed without one. |
| `probe_behaviour_history` | Every role change, link, confirmation, probe key change, archive and delete: who (`actor`, `operator` or `agent:<name>`), when, old and new value, reason. No foreign key, so a deleted behaviour's history stays readable. **Append-only** (ORM events and a trigger). |

### Session spot-checks (Feature 038)

| Table | Purpose |
|---|---|
| `probe_spot_checks` | One spot-check of a catalogued behaviour against a library judge prompt: the prompt as columns (`prompt_id`, `prompt_key`, `prompt_hash`), the judge `endpoint` and `model`, the seeded `sample_spec` (strata quotas, window, near-threshold band, time range, the primary and compared probes with their miLLM ids), what miLLM returned (`sample_returned`, ids and per-stratum shortfall only), `status`, `partial` (stopped after some turns), `parse_failures`, and the report in `metrics`. `pinned` and `excerpts_expire_at` govern the judged text (kept seven days unless pinned). |
| `probe_spot_check_items` | One sampled turn × one compared probe: where the turn is (`session_id`, `request_id`, `turn_index`, `event_id`), the stratum, the probe's `window`, score, threshold and `probe_fired`, what the judge saw (`judge_saw`), and the judge's answer (`judge_value`, `judge_positive`, `parse_failed`). **Holds recorded user text** — `judged_text` and `judge_reply` — which is deleted with the run and nulled when the excerpts expire; `not_recorded` marks a sampled turn with no transcript (counted, never judged). |

### Behaviour detectors (Feature 040)

A behaviour version can carry a **detector definition**: sub-questions, success criteria, held-out columns, the
models it is meant for, a calibration floor and a rule-set reference. These are six nullable columns on
`probe_behaviour_versions`, NULL on every version written before Feature 040. Reports about whether a detector
meets its bar are stored once and never edited.

| Table | Purpose |
|---|---|
| `rule_sets` | A rule set a behaviour version cites, stored by its identity digest (`rule_set_digest`) with the source it came from (inline, or a miDataworks name and version). **Append-only** (ORM events and a trigger): a changed rule set is a new row with a new digest. |
| `detector_success_reports` | One success report: the behaviour version, model and dataset identity it judged, the report body (RFC 8785 canonical JSON) and its SHA-256, the overall verdict, who built it. Keyed by `identity_sha256`, the digest of every input; building it again from the same inputs returns this row (`ON CONFLICT DO NOTHING`). **Append-only**. Whether it is stale (a later version, a removed probe, evaluation or view) is computed when it is read, never written back. |
| `probe_monitor_dataset_column_changes` | Each additive change to a view's `group_columns` when the view is not linked to a catalogued behaviour (a catalogued one records it in `probe_behaviour_history`): who, when, the columns before and after, and the snapshot digests. **Append-only**. |
| `probe_rules_evaluations` | One rules ∪ probe evaluation (`pre_…`, Feature 040 S3): the probe, the views, the behaviour version, the rule-set and engine digests, status (`pending`, `running`, `completed`, `failed`, `cancelled`, CHECK), progress and heartbeat, the cancel request, the `task_queue` row an agent polls, and the result body (RFC 8785 canonical JSON: counts, rates, digests — no row text) with its SHA-256. A **partial unique index** on `identity_sha256` over the live rows (pending, running, completed) makes a repeated request return the existing row; a failed or cancelled one never blocks a rerun. A completed body is **never rewritten or deleted** (ORM event and trigger); a moved input reads `stale` on read. |
| `rule_verdict_cache` | The index of cached per-view rule verdicts (Feature 040 S3), keyed by view, rule-set digest, engine digest and reading digest (the scope and column the rules read). The files — `rule_match`, `rule_veto` and `unknown` arrays by split position — live under `rule_verdicts/<view_id>/`; the row says they were published (atomic rename). Pruned by age and size, never while an evaluation reads it; a purged view's directory goes with it. |
| `code_rules` | Operator code-rule documents (Feature 040 S4), content-addressed: the key is `code_sha256`, the SHA-256 of the runner contract's canonical JSON, and `canonical` holds exactly those bytes, so what the isolated runner runs is what the operator approved. Stored inline (from a request) once; **append-only** (ORM event and trigger). |
| `code_rule_approvals` | The operator's per-hash decisions on operator code (Feature 040 S4): `approve` or `revoke`, a non-empty reason, who decided, and the approval request it answered. `origin` is CHECKed equal to `operator` — an agent can file a request, never write a decision. The latest row per hash is the current state, read fresh at every use and before each chunk of a rules evaluation. **Append-only.** |

`probe_monitor_datasets` gains `group_columns` (the columns a breakdown may group by), `sub_question`,
`behaviour_version` (the version `sub_question` was checked against) and `column_snapshots` (per column, the
dictionary-encoded snapshot under `probe_monitor_views/<view_id>/` and its SHA-256, so a breakdown joins by
row position on exactly the values registered). The `blend_fit` role is new: a view held back for fitting a
blend (Feature 041), never a training view. `probe_monitor_runs` and `probe_monitors` carry `sub_question` and
`behaviour_version` copied from the training view. `probe_monitors` gains `calibration_positions_path` (the split
rows of the calibration negatives, int64, aligned with the saved scores) and `calibration_n_negatives`.

Feature 040 S3 adds to `probe_monitor_datasets` `text_column` (the column rules read when it is not the input
column, read live with a drift check), `rule_columns` (producer rule verdict columns and the rule-set identity they
were computed with, checked against the manifest's rule-set listing) and `rule_set_listing` (that listing, as read
at registration).

`probe_monitor_datasets` gains `behaviour` and `positive_meaning` (sent by miDataworks with a detector set)
and `positive_labels_confirmed` (`{key, version, by, at, origin}` — who confirmed that the view's positives
are that behaviour).

### Blend reports (Feature 041)

A **blend report** evaluates combinations of one behaviour's probes on one model, measured the way an
application would deploy them. The report's authority is a canonical file (`report.json`, RFC 8785 JSON, format
`mistudio.blend-report/1`) and its per-row values (`.npy`) under `probe_monitors/blends/<report id>/`; the rows
below index them. **No foreign key** points from a report to a probe, run, view or behaviour: deleting a run or
purging a view never deletes evidence and is never blocked by it; staleness is computed on read.

| Table | Purpose |
|---|---|
| `probe_blend_reports` | One blend evaluation (`pbr_…`): behaviour key and version, model, dataset version, status (`queued`, `running`, `done`, `failed`, `cancelled`, CHECK), failure code and words, the request as sent, the inputs fixed at start (`resolved`), the report file's path, SHA-256 and size, the per-row directory and when (and by whom) it was pruned, rerun links both ways, heartbeat and cancel request, Celery task id. |
| `probe_blend_report_members` | Which probes a report cites (report FK, cascade; probe id with no FK). |
| `probe_suggested_blends` | The current suggested blend per (behaviour, model): the record bytes (sliced from a report, format `mistudio.suggested-blend/1`), its SHA-256, the report and candidate. Advisory; nothing serves it. |
| `probe_suggested_blend_history` | Every marking (`psb_…`): the record, who marked it, when and why. **Append-only** (ORM events and a trigger). |

Deleting a run deletes its probes, and deleting a probe deletes its evaluations.
Deleting an **SAE** does not delete a probe trained over it: the probe becomes
unbuildable, and that is a state to report rather than erase.

### Dataset downloads are identified by (repo, config, split)

Feature 032 changed the download identity. The raw directory is
`<datasets_dir>/<org>_<name>[__<config>][__<split>]` and the duplicate guard
compares all three, so the seven configs of one repo are seven datasets. With no
config and no split the path is byte-identical to the previous one, so rows
already on disk keep their paths. The display name carries the config, because
seven rows sharing one name are unusable in a list.

## Tables this page used to omit

Nine tables were undocumented until MIS-E2E-050. Three of them mattered
particularly: `checkpoints` is drawn in this page's own ER diagram above and had
no entry; the four `feature_group*` / `feature_token_index` tables back three
shipped features and the entire Feature Groups panel; and
`agent_approval_requests` is the gate deciding whether an MCP agent's
destructive action needs a human.

| Table | Purpose |
|---|---|
| `checkpoints` | Per-step training checkpoints — loss, L0, weights path; after a resume, `extra_metadata` carries `resume_history`. Beside each step's weights sits `training_state.pt`, which holds everything an exact resume restores: each SAE's optimizer moments, scheduler, GradScaler, dead-latent tracker (including **which** latents each of the most recent resamples revived, which nothing else records — the step counter beside it reads zero both for a revived latent and for one that merely fired) and activity statistics, plus the best loss, the random-number state, the activation source's data position, the storage `activation_plan` and `resume_history`. It is about twice the size of the weights, and three times when the step ends inside a gradient-accumulation window, which also saves the gradients. One row per layer per step on a multi-layer run. Drawn in the ER diagram above; subject to step-granular retention (see [Training Lifecycle](/core-workflow/training-lifecycle)). |
| `feature_groups` | A cluster of related features (PPRD rows 13–15) |
| `feature_group_members` | Membership rows joining a feature to its group, with per-member strength and similarity |
| `feature_grouping_runs` | One execution of the grouping algorithm, with its parameters |
| `feature_token_index` | Token → feature index backing token search and the TF-IDF context subgrouping |
| `feature_analysis_cache` | Cached per-feature analyses (logit lens, correlations, NLP) with a 7-day expiry. Unique on `(feature_id, analysis_type)` — a blind INSERT here caused a production 500 once the expiry made a row invisible to the read but still present in the table. |
| `feature_dashboard_data` | Precomputed dashboard payloads (logit-lens and histogram blobs) |
| `agent_approval_requests` | The MCP approval gate: an agent's destructive action parked for human approval. Steering requests are executed by the backend on approve; a miLLM model load (`millm_load_model`) is only marked approved and then **consumed** once by the MCP tool (`status` `consumed`, `consumed_at`). Load requests expire after 24 hours pending or approved-unused; steering requests never expire |
| `dismissed_operations` | Operations the user has dismissed from the Monitor page, so they stay dismissed |

`alembic_version` (migration bookkeeping) and `feature_activations_default` (the
default partition of `feature_activations`) are deliberately not described: one
belongs to Alembic and the other is an implementation detail of a partitioned
table already documented above.

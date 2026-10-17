import type { PrecisionLabel } from './precision';
/**
 * Probe monitor wire types (Feature 032).
 *
 * The response shapes VERBATIM as the backend sends them. No reshaping here, for the
 * reason the J-Lens client records: an adaptation layer lets the panel drift into a
 * frontend-only shape while still appearing to conform.
 *
 * ⚠ THE RUNG'S WORDING COMES FROM THE SERVER. `rung_language` and `rung_next_step` are
 * fields, not something the client composes from `rung`. A detector's language is the
 * thing most likely to drift above its evidence, and miLLM mirrors those strings
 * verbatim — so there is deliberately no client-side map from a rung number to a phrase.
 */

export type ProbeRole = 'train' | 'eval' | 'calibration';
export type ProbeDistribution = 'in_distribution' | 'out_of_distribution';
export type ProbeScope = 'all' | 'assistant' | 'user' | 'last_assistant';
export type ProbeVariant = 'dense' | 'sae';

/** The five numbers a mapping produced. `null` is not zero — see `ProbeDatasetCounts`. */
export interface ProbeDatasetCounts {
  positive: number;
  negative: number;
  excluded: number;
  filtered_out: number;
  unparseable: number;
  /** How each input row was read: plain / messages / json_messages / roles_guessed. */
  kinds?: Record<string, number>;
}

export interface ProbeDataset {
  id: string;
  name: string;
  dataset_id: string;
  config: string | null;
  split: string | null;
  input_column: string;
  label_column: string;
  label_mapping: Record<string, string>;
  keyword_filter: Record<string, unknown> | null;
  pair_column: string | null;
  role: string;
  distribution: string | null;
  counts: ProbeDatasetCounts;
  created_at: string;
}

export interface ProbeRunConfig {
  layers?: number[] | null;
  stride?: number | null;
  rules: string[];
  scope: ProbeScope;
  max_length: number;
  val_fraction: number;
  seed: number;
  top_n_layers: number;
  sae_variant: boolean;
  sae_k: number[];
  target_fpr: number;
  batch_size?: number | null;
  /** Never a choice: a run loads at its model's own precision, recorded in
   *  `environment.model_dtype`. A value that disagrees is refused at submit. */
  dtype?: 'float16' | 'bfloat16' | 'float32' | null;
}

export interface ProbeRun {
  id: string;
  model_id: string;
  train_dataset_id: string;
  eval_dataset_ids: string[];
  calibration_dataset_id: string | null;
  config: Record<string, unknown>;
  stage: string | null;
  status: string;
  progress: number | null;
  error_message: string | null;
  celery_task_id: string | null;
  gpu_request: string | null;
  gpu_uuid: string | null;
  artifact_dir: string | null;
  environment: Record<string, unknown>;
  /** The precision this run trained at — recorded, or inferred (float16, before 2026-10-03). */
  model_dtype_label?: PrecisionLabel;
  layer_selection: ProbeLayerSelection | null;
  created_at: string;
  updated_at: string;
  completed_at: string | null;
}

export interface ProbeLayerScore {
  layer: number;
  pooling: string;
  /** `null` means the cell could not be scored — never rendered as 0. */
  val_auroc: number | null;
  n_train: number;
  n_val: number;
}

export interface ProbeLayerSelection {
  grid: ProbeLayerScore[];
  chosen: number[];
  /**
   * The winner's margin over the runner-up. A layer chosen by 0.002 of AUROC is an
   * arbitrary choice, and the panel says so rather than letting a reader over-read it.
   */
  margin: number | null;
  poolings: string[];
}

export interface ProbeMonitorSummary {
  id: string;
  run_id: string;
  layer: number;
  rule: string;
  rule_params: Record<string, unknown>;
  variant: string;
  sae_id: string | null;
  sae_feature_indices: number[] | null;
  val_metrics: Record<string, unknown>;
  selected: boolean;
  /** `null` means NO threshold was placed — not "fires on everything". */
  threshold: number | null;
  target_fpr: number | null;
  /** What the threshold actually spends, which is usually not the target. */
  realised_fpr: number | null;
  threshold_source: string | null;
  streamable: boolean;
  rung: number;
  rung_reasons: string[];
  /**
   * The rung's phrase and next step, RENDERED BY THE SERVER.
   *
   * ⚠ Present on the LIST as well as the report, and that is deliberate: `RungChip` holds no
   * number→phrase map because miLLM mirrors these strings verbatim, so a frontend copy is a
   * second vocabulary that drifts. Without them here a probe tile had no honest way to state its
   * rung, so it stated nothing.
   */
  rung_language?: string;
  rung_next_step?: string;
  /** 033: set once a definition has been built; cleared when what it states changes. */
  definition_built_at?: string | null;
  definition_sha256?: string | null;
  /** What the build asked for and resolved, plus `invalidated` when it was cleared. */
  definition_build?: Record<string, unknown> | null;
  published?: ProbePublication[];
  created_at: string;
  /**
   * WHICH TOKENS. The internal scope the probe was TRAINED and calibrated under (the run's):
   * `all`, `user`, `input`, `assistant`, `last_assistant`. Optional on the wire: an older backend
   * does not send it, and the tile then says it does not know rather than guessing.
   */
  scope?: string | null;
  /**
   * `{all|prompt|response: threshold}` — each window's OWN bar. `{}` = one bar for all; `null` =
   * NOT COMPUTED by the server (2026-10-08: it used to default to `{}`, which said "one bar").
   */
  window_thresholds?: Record<string, number | null> | null;
  /** Length bands refining the bar over the probe's own scope; 0 = none; `null` = not computed. */
  length_band_count?: number | null;
  /**
   * THE RENDER FORM it was trained under (2026-10-08), e.g. `{generation_prompt: true,
   * add_special_tokens: false}` — the form miLLM serves. `null` = NOT RECORDED: rendered WITHOUT
   * the generation prompt (every probe before that date). Never read as the served form.
   */
  render_form?: Record<string, unknown> | null;
  /** Whether `render_form` is the served form — decided by the SERVER, by the gates' own rule. */
  render_served?: boolean;
  /**
   * Whether `render_form` records what the start-of-text (BOS) rule did (`bos_handling`,
   * 2026-10-08). `false` on a served form tokenized before that record existed.
   */
  render_bos_recorded?: boolean;
}

export interface ProbeRocPoint {
  /** `null` is the origin: a threshold above every score, firing on nothing. */
  threshold: number | null;
  fpr: number;
  tpr: number;
}

export interface ProbeOperatingPoint {
  target_fpr: number;
  threshold: number | null;
  realised_fpr: number;
  recall: number;
}

export interface ProbeLengthBand {
  band: number;
  n: number;
  min_length: number | null;
  max_length: number | null;
  minimum: number;
  scored: boolean;
  auroc?: number | null;
  reason?: string;
  n_positive?: number;
  n_negative?: number;
}

/**
 * ⚠ `scored: false` CARRIES A REASON, AND THE PANEL MUST RENDER IT. Never a 0, never a
 * blank: a missing number reads as "not run yet" and a 0.5 reads as "measured, and
 * chance". Neither is what a refusal means.
 */
export interface ProbeMetrics {
  name?: string;
  out_of_distribution?: boolean;
  scored: boolean;
  reason?: string;
  auroc?: number | null;
  ci?: { low: number; high: number; resamples: number; alpha: number } | null;
  n_positive?: number | null;
  n_negative?: number | null;
  roc?: ProbeRocPoint[];
  operating_points?: ProbeOperatingPoint[];
  length_bands?: ProbeLengthBand[] | null;
}

export interface ProbeEvaluation {
  id: string;
  probe_id: string;
  dataset_id: string;
  status: string;
  n_positive: number | null;
  n_negative: number | null;
  metrics: ProbeMetrics;
  created_at: string;
}

export interface ProbeJudgeRunSummary {
  id: string;
  model: string;
  status: string;
  prompt_version: string;
  parse_failures: number;
  metrics: Record<string, unknown>;
}

export interface ProbeReport {
  probe: ProbeMonitorSummary;
  /** From the server. The client never maps a rung number to a phrase. */
  rung_language: string;
  rung_next_step: string;
  evaluations: ProbeEvaluation[];
  paired_probe_id: string | null;
  /**
   * 033: the SAE's HuggingFace repo. `null` on a dense probe AND on an SAE probe whose dictionary
   * has no published home — which the export section distinguishes by `probe.variant`, because an
   * SAE probe with no repo cannot be exported at all.
   */
  sae_hf_repo?: string | null;
  judge_runs: ProbeJudgeRunSummary[];
  /**
   * ⚠ ASSEMBLED ON EVERY REPORT SINCE 032 AND RENDERED NOWHERE UNTIL RECALIBRATION SHIPPED.
   *
   * It is the one thing that says a probe's absolute score does NOT transfer between
   * distributions: on the first shipped probe the five evaluation sets' own 1% thresholds spanned
   * 24 points, so one number behaves very differently on each. The server has returned it all
   * along; this interface simply omitted it, which is why nobody saw it. A fast threshold dial
   * without this on screen is an invitation to lower the bar until the probe fires.
   */
  threshold_transfer?: ThresholdTransfer | null;
  /** `val_auroc` is a maximum over the epochs and layers the same split chose, so it is an
   *  optimistic upper bound rather than a held-out estimate. Also server-assembled, also unused. */
  validation_caveat?: Record<string, unknown> | null;
}

/** What one threshold does on one evaluation set, looked up against that set's stored ROC. */
export interface ThresholdTransferSet {
  name: string | null;
  auroc: number | null;
  own_threshold_at_1pct: number | null;
  max_score: number | null;
  recall_at_shipped?: number | null;
  fpr_at_shipped?: number | null;
  /** The bar is above every score the set produces, so recall there is zero however well it ranks. */
  unreachable?: boolean;
}

export interface ThresholdTransfer {
  shipped_threshold: number | null;
  threshold_source: string | null;
  per_set: ThresholdTransferSet[];
  own_threshold_spread: number | null;
  unreachable_sets: string[];
  /** Server-written prose. The client never composes this sentence itself. */
  caution: string | null;
}

/**
 * One end of a proposed or committed threshold move — the probe's OWN decision bar.
 *
 * ⚠ NOT `ProbeOperatingPoint`, WHICH ALREADY EXISTS AND MEANS SOMETHING ELSE: that one is a point
 * on an evaluation SET's ROC curve, the bar that set would need to spend a given FPR. This one is
 * the single number the probe actually serves. The first draft of this interface reused the name
 * and TypeScript silently MERGED the two declarations, surfacing only where a field's nullability
 * differed — two different concepts would otherwise have become one type with both their fields.
 */
export interface ProbeDecisionBar {
  threshold: number | null;
  target_fpr: number | null;
  realised_fpr: number | null;
  threshold_source?: string | null;
  revision?: number;
  n_negatives?: number;
  /** A bar above every negative. A real operating point, and the one to never set by accident. */
  fires_on_nothing?: boolean;
}

export interface ProbeRecalibration {
  probe_id: string;
  current: ProbeDecisionBar;
  proposed: ProbeDecisionBar;
  window_decisions: Record<string, Record<string, unknown>> | null;
  length_bands: Array<Record<string, unknown>> | null;
  transfer_current: ThresholdTransfer;
  transfer_proposed: ThresholdTransfer;
  /** A definition already on HuggingFace is append-only and cannot be reached. */
  published_copies_go_stale: boolean;
  committed: boolean;
  definition_invalidated: boolean;
}

export interface ProbeJudgeRun {
  id: string;
  endpoint: string;
  model: string;
  prompt_version: string;
  dataset_ids: string[];
  probe_id: string | null;
  status: string;
  progress: number | null;
  parse_failures: number;
  metrics: Record<string, unknown>;
  error_message: string | null;
}

export interface ProbeScoreToken {
  /** The VOCABULARY string — `Ġonline`, `Ċ`, `<|im_start|>`. Kept because it is what a tokenizer
   *  bug is diagnosed from, and because the raw spelling is the thing that identifies a token. */
  token: string;
  /** The same token as TEXT, decoded by the tokenizer that produced it. `Ġ`→space, `Ċ`→newline.
   *  Optional so a trace from an older build still renders. */
  text?: string;
  /** Chat-template scaffolding rather than content. A probe firing on these is a FINDING. */
  special?: boolean;
  scored: boolean;
}

export interface ProbeScoreResult {
  probe_id: string;
  aggregate: number;
  /**
   * THE BAR THIS INPUT IS ACTUALLY JUDGED AGAINST — the length band's threshold when the probe has
   * a band table, its global threshold otherwise. Until 2026-10-03 this was always the global
   * bar, so the panel and the live monitor disagreed about which bar applied.
   */
  threshold: number | null;
  /** The probe's headline bar, shown beside the applied one when a band moved it. Optional on the
   *  wire: a backend from before 2026-10-03 omits it. */
  global_threshold?: number | null;
  /** Which length band applied, or null when the probe has no band table. `threshold_source:
   *  "global"` marks a band with too few negatives to be cut, which inherits the global bar. */
  threshold_band?: {
    min_tokens: number;
    max_tokens: number | null;
    threshold_source: 'band' | 'global' | null;
  } | null;
  /** `null` when no threshold was placed: the probe has said nothing, not "no". */
  fires: boolean | null;
  tokens: ProbeScoreToken[];
  token_scores: number[];
  n_scored: number;
  role_mask_reliable: boolean;
  truncated: boolean;
}

export interface ProbeRunAccepted {
  id: string;
  task_id: string;
  status: string;
}


// ── 033: the portable definition ───────────────────────────────────────────

/**
 * `mistudio.probe-definition/v1`, as the API returns it.
 *
 * ⚠ TYPED LOOSELY ON PURPOSE, AND ONLY WHERE THE UI READS IT. The authority is the pydantic
 * contract and `docs/schemas/probe-definition-v1.json`; a full hand-written mirror here would be a
 * second declaration of the same shape, free to drift, and this estate has already paid for that
 * (miLLM's hand-written mirror silently dropped a re-vendored field). The fields below are the ones
 * the panel displays — everything else travels through untouched.
 */
export interface ProbeDefinition {
  kind: string;
  name: string;
  description?: string | null;
  concept?: string | null;
  model: {
    hf_id: string;
    revision: string;
    d_model: number;
    n_layers: number;
    architecture: string;
  };
  read: { layer: number; hook_point: string };
  scope: string;
  basis: 'residual' | 'sae_features';
  aggregation: { rule: string; streamable: boolean };
  decision: { threshold: number | null; target_fpr: number | null; realised_fpr: number | null };
  evidence: {
    rung: number;
    rung_language: string;
    acknowledgement: { by: string; at: string; reason: string } | null;
    evaluations: Array<{ auroc: number; distribution: string }>;
  };
  test_vectors: { tolerance: number; vectors: unknown[] };
}

/** One HuggingFace publication of a probe. Appended, never replaced. */
export interface ProbePublication {
  repo_id: string;
  revision: string | null;
  path: string;
  private: boolean;
  at: string;
  sha256?: string | null;
}


/**
 * The envelope `GET /probes/{id}/score/{task_id}` returns around a `ProbeScoreResult`.
 *
 * ⚠ IT IS A SEPARATE TYPE, AND THE FIRST VERSION OF IT WAS NOT. I declared a second interface
 * ALSO called `ProbeScoreResult`, and TypeScript merged the two declarations instead of
 * rejecting them — so `tsc` stayed green while every existing fixture silently acquired two
 * required fields it did not have. The merge is invisible at the definition site and only shows
 * up as a mismatch wherever the older shape is constructed.
 *
 * `status` is Celery's, passed through unchanged: SUCCESS carries `result`, FAILURE carries
 * `error` and a null `result`, and PENDING means EITHER queued OR a task id Celery has never
 * heard of — the server cannot distinguish those and says so rather than guessing, so a UI must
 * not render PENDING as "queued" with any confidence.
 */
export interface ProbeScoreTask {
  task_id: string;
  probe_id: string;
  status: string;
  /** The score itself, in the shape `score_one` returns — `aggregate`, `fires`, the trace. */
  result?: ProbeScoreResult | null;
  error?: string;
}

/** The body `POST /judge-runs` takes. */
export interface SubmitJudgeRunBody {
  endpoint: string;
  model: string;
  dataset_ids: string[];
  probe_id?: string | null;
  max_rows_per_set?: number;
  parse_failure_limit?: number;
}

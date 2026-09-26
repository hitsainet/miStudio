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
  dtype: 'float16' | 'bfloat16' | 'float32';
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
  /** 033: set once a definition has been built; cleared when what it states changes. */
  definition_built_at?: string | null;
  definition_sha256?: string | null;
  /** What the build asked for and resolved, plus `invalidated` when it was cleared. */
  definition_build?: Record<string, unknown> | null;
  published?: ProbePublication[];
  created_at: string;
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
  token: string;
  scored: boolean;
}

export interface ProbeScoreResult {
  probe_id: string;
  aggregate: number;
  threshold: number | null;
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

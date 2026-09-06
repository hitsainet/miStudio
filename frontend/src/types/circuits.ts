/**
 * Circuit types (Feature 018) — mirrors the /circuits API. Rung language is
 * SERVER-rendered (rung_language / rung_next_step): the UI must display those
 * strings verbatim and never compose its own causal phrasing (IDL-35).
 */

import { EvidenceRung } from './evidenceLadder';

export type EdgeType = 'computed' | 'persistence' | 'attention_mediated';

export interface CircuitEdge {
  up: { layer: number; kind?: string; feature_idx?: number; cluster_profile_id?: string };
  down: { layer: number; kind?: string; feature_idx?: number; cluster_profile_id?: string };
  type: EdgeType;
  type_signals?: Record<string, unknown> | null;
  rung: EvidenceRung;
  tested_and_failed: EvidenceRung[];
  coactivation?: { pmi?: number; lift?: number; support?: number; null_percentile?: number;
                   replicated_heldout?: boolean } | null;
  weight_prior?: number | null;
  attribution?: { score?: number; sign_consistency?: number; method?: string } | null;
  validation_manifest_ref?: string | null;
  effect_size?: number | null;
}

export interface CircuitMember {
  layer: number;
  member_kind: 'feature_ref' | 'cluster_ref';
  feature?: { feature_idx: number; label?: string | null; strength: number } | null;
  cluster_profile_id?: string | null;
  cluster_name?: string | null;
  expanded_members?: { feature_idx: number; label?: string | null; strength: number }[] | null;
}

export interface CircuitSummary {
  id: string;
  name: string;
  granularity: 'feature' | 'cluster';
  layers: number[];
  member_count: number;
  edge_count: number;
  rung: EvidenceRung;
  rung_language: string;
  rung_next_step: string;
  promoted: boolean;
  model_id: string | null;
  version: number;  // optimistic-concurrency token (017 Task 3.0)
  updated_at: string;
}

export interface Circuit extends CircuitSummary {
  name: string;
  narrative: string | null;
  granularity: 'feature' | 'cluster';
  saes: { mistudio_sae_id?: string; layer?: number }[];
  members: CircuitMember[];
  edges: CircuitEdge[];
  budget: Record<string, unknown> | null;
  // Faithfulness (rung 3) — necessity/sufficiency causal evidence. `sufficiency_k`
  // is the k-non-members restored for the sufficiency test; metric_id/manifest_ref
  // point at the recorded run.
  faithfulness: {
    necessity?: number;
    sufficiency?: number;
    sufficiency_k?: number | null;
    metric_id?: string | null;
    manifest_ref?: string | null;
  } | null;
  // Lifecycle of a faithfulness run (pending|running|completed|failed|null).
  faithfulness_status?: string | null;
  // Calibration (Feature 20) — the usable steering band. When present, the
  // served dial is clamped to [onset, cliff] and defaults to sweet_spot.
  // `provisional` = generated probes / discovery-plane measurement.
  calibration?: {
    onset: number;
    sweet_spot: number;
    cliff: number;
    provisional?: boolean;
    non_monotone?: boolean;
    judge_metric_id?: string | null;
    manifest_ref?: string | null;
  } | null;
  // Lifecycle: pending|running|completed|no_band|judge_unreliable|failed|null.
  calibration_status?: string | null;
  discovery: Record<string, unknown> | null;
  // Producing discovery run, if any — a faithfulness run needs it for prompts.
  discovery_run_id?: string | null;
  created_at: string;
}

// ── Feature 016: Capture ──────────────────────────────────────────────────

/** A cost estimate returned by a probe run (confirm=false). */
export interface CaptureEstimate {
  events?: number;
  bytes?: number;
  minutes?: number;
  [k: string]: unknown;
}

export interface CaptureAttentionConfig {
  layers: number[];
  heads?: number[] | null;
  top_k: number;
}

export interface CircuitCapture {
  id: string;
  status: string;
  progress: number | null;
  error_message?: string | null;
  corpus: unknown;
  model_id: string | null;
  layers: { layer: number; sae_id: string }[] | null;
  split: { heldout_count: number; [k: string]: unknown } | null;
  estimate: CaptureEstimate | null;
  attention_capture: CaptureAttentionConfig | null;
  counts: Record<string, unknown> | null;
  bytes: number | null;
  events_total: number | null;
  stale: boolean;
  created_at: string;
  updated_at: string;
}

export interface CircuitCaptureCreate {
  dataset_id: string;
  model_id?: string;
  layers: { layer: number; sae_id: string }[];
  epsilon?: number;
  theta_floor?: number;
  sample_cap?: number;
  split_seed?: number;
  attention_capture?: CaptureAttentionConfig;
  confirm: boolean;
}

// ── Feature 016: Discovery ────────────────────────────────────────────────

export type DiscoveryGranularity = 'feature' | 'cluster';
export type DiscoveryMode = 'seeded' | 'open';

/** A seed reference — either a feature index or a cluster profile, per layer. */
export interface DiscoverySeedRef {
  layer: number;
  feature_idx?: number;
  cluster_profile_id?: string;
}

export interface DiscoveryCreate {
  capture_run_id: string;
  granularity: DiscoveryGranularity;
  mode: DiscoveryMode;
  seed_refs?: DiscoverySeedRef[];
  s_min?: number;
  null_shuffles?: number;
  null_percentile?: number;
  fdr_q?: number;
  cohesion_floor?: number;
  seed?: number;
  force?: boolean;
}

/** A candidate node ref — feature index or cluster profile. */
export interface DiscoveryNodeRef {
  layer: number;
  feature_idx?: number;
  cluster_profile_id?: string;
  cluster_name?: string;
}

export interface DiscoveryCandidate {
  up: DiscoveryNodeRef;
  down: DiscoveryNodeRef;
  granularity: DiscoveryGranularity;
  stats: {
    pmi?: number;
    lift?: number;
    support?: number;
    spearman?: number;
    null_pct?: number;
    p_value?: number;
    pooled_p?: number;
  };
  replicated_heldout: boolean;
  attribution?: {
    score?: number;
    sign_consistency?: number;
    method?: string;
    rung1_gate?: boolean;
  } | null;
  orderings?: { coact_rank?: number; attr_rank?: number } | null;
  // Feature 017: rung-2 validation write-back onto the discovery candidate.
  validation?: {
    ordering: 'coact' | 'attr';
    effect_size: number;
    passed: boolean;
    manifest_id: string;
  } | null;
  validated_rung?: number | null;
  tested_and_failed_history?: { ordering: string; reason: string }[];
}

export interface DiscoveryReport {
  granularity: DiscoveryGranularity;
  mode: DiscoveryMode;
  supernode_activation?: string;
  lag0_disclosure: string;
  null_summary: { method: string; shuffles: number; percentile: number };
  fdr: {
    discipline: string;
    p_source: string;
    p_resolution?: number;
    q: number;
    tested: number;
    passed: number;
  };
  replication: { tested: number; replicated: number; rate: number | null };
  counts_by_stage: {
    pairs_considered: number;
    post_support: number;
    null_tested: number;
    post_fdr: number;
    candidates_persisted: number;
  };
  caps: {
    candidates_truncated: boolean;
    unit_cap_hit_layers: number[];
    null_cap_hit: boolean;
    [k: string]: unknown;
  };
  uncovered_seeds: { layer: number; ref: unknown; reason: string }[];
  attribution: Record<string, unknown> | null;
  uplift: null;
  wall_clock_seconds?: number;
  // Feature 017: a completed validation pass records its batch summary here.
  validation?: {
    ordering: 'coact' | 'attr';
    k: number;
    survival: number | null;
    passed: number;
    manifest_id: string;
    wall_clock_seconds?: number;
    // Both-orderings comparison (US-1): once both coact and attr orderings have
    // been validated, `uplift` = survival(attr) − survival(coact) and each
    // ordering's own survival/passed/k/manifest lands under `by_ordering`.
    by_ordering?: {
      coact?: { survival: number | null; passed: number; k: number; manifest_id: string };
      attr?: { survival: number | null; passed: number; k: number; manifest_id: string };
    } | null;
    uplift?: number | null;
  } | null;
}

export interface DiscoveryRun {
  id: string;
  capture_run_id: string;
  status: string;
  progress: number | null;
  error_message?: string | null;
  params: Record<string, unknown> | null;
  report: DiscoveryReport | null;
  candidate_count: number;
  candidates?: DiscoveryCandidate[];
  // Attribution's own lifecycle (R2 B4) — the discovery status stays
  // 'completed' regardless of the attribution pass's outcome.
  attribution_status?: string | null;
  attribution_progress?: number | null;
  attribution_error?: string | null;
  // Validation's own lifecycle (Feature 017) — discovery status stays
  // 'completed' regardless of a validation pass's outcome.
  validation_status?: string | null;
  validation_progress?: number | null;
  validation_error?: string | null;
  created_at: string;
  updated_at: string;
}

// ── Feature 017: Validation + manifests ───────────────────────────────────

/** Config body for POST /circuit-discovery/{id}/validate. */
export interface ValidateConfig {
  ordering: 'coact' | 'attr';
  k: number;
  prompts_per_edge: number;
  null_samples: number;
  percentile: number;
  sign_frac: number;
  baseline: 'zero' | 'corpus_mean';
  seed: number;
}

/** One edge verdict inside an edge_batch manifest's payload.edges. */
export interface ValidationEdge {
  up: { layer: number; feature_idx: number };
  down: { layer: number; feature_idx: number };
  effect_size: number;
  sign_consistency: number;
  sigma_d: number;
  n_prompts: number;
  null_percentile_value: number;
  verdict: { passed: boolean; reason: string };
  rung: 2 | null;
  tested_and_failed: boolean;
}

/** A self-contained, reproducible validation manifest. */
export interface ValidationManifest {
  id: string;
  kind: 'edge_batch' | 'faithfulness' | 'reproduction';
  discovery_run_id: string | null;
  circuit_id: string | null;
  parent_manifest_id: string | null;
  payload: {
    intervention?: { kind?: string; baseline?: string };
    config?: Record<string, unknown>;
    seeds?: number[];
    ordering?: string;
    k?: number;
    edges?: ValidationEdge[];
    survival?: number | null;
    null_summary?: { samples?: number; percentile?: number; kind?: string };
    // reproduction manifests carry the verdict NESTED under `verdict` (R1 #3)
    // and the re-executed edges under `reproduced_edges`.
    reproduce_of?: string;
    reproduced_edges?: ValidationEdge[];
    verdict?: {
      within_tolerance: boolean | null;
      max_delta?: number | null;
      tolerance?: number;
      reason?: string;
      deltas?: { edge: unknown; delta: number }[];
    };
    [k: string]: unknown;
  };
  created_at: string;
}

/**
 * Probe families and blend reports (Feature 041). The server's shapes, unchanged: every word, figure and
 * flag comes from the backend (`services/probe_family.py`); nothing here is computed.
 */
import type { NextCall } from './probeMonitor';

export interface FamilyFlag {
  flag: string;
  field?: string;
  value?: unknown;
  majority?: unknown;
  majority_tie?: boolean;
  reason?: string;
  sub_question?: string;
  version?: number | null;
}

export interface FamilyMember {
  probe_id: string;
  run_id: string;
  model_id: string;
  role: 'primary' | 'candidate' | 'retired';
  role_recorded: boolean;
  precision: string | null;
  precision_refusal: string | null;
  render_form: string;
  scope: string;
  layer: number;
  rule: string;
  variant: string;
  sub_question: string | null;
  behaviour_version: number | null;
  rung: number;
  rung_language: string;
  threshold: number | null;
  decision_revision: number | null;
  decision_digest: string | null;
  detector_digest: string | null;
  identity_current: boolean | null;
  training_view: string;
  training_dataset_version: string | null;
  definition_built: boolean;
  default_blend: boolean;
  flags: FamilyFlag[];
}

export interface CoverageEntry {
  sub_question: string | null;
  group?: 'main_question';
  state: 'covered' | 'missing' | 'empty';
  members: string[];
  next?: NextCall[];
}

export interface ProbeFamily {
  key: string;
  version: number;
  model: string | null;
  models: { model_id: string; members: number }[];
  declared_sub_questions: string[];
  coverage: CoverageEntry[];
  default_blend_set: string[];
  flags_basis: string;
  members: FamilyMember[];
  total: number;
  offset: number;
  limit: number;
  next_offset: number | null;
}

export interface FamilyQuery {
  model?: string | null;
  member_ids?: string[];
  include_retired?: boolean;
  offset?: number;
  limit?: number;
}

// ── S2: blend reports and the suggested blend ──────────────────────────────────────────────

export interface StaleReason {
  code: string;
  probe_id?: string;
  fields?: string[];
  [extra: string]: unknown;
}

export interface BlendFigures {
  recall: number | null;
  wilson: [number, number] | null;
  auroc: number | null;
  auroc_reason?: string | null;
  n_pos: number;
  n_neg: number;
  realised_fpr: number | null;
  breakdown?: Record<string, unknown[]>;
}

export interface BlendCandidate {
  index: number;
  name: string;
  kind: string;
  produces: 'score' | 'verdict';
  params: Record<string, unknown>;
  calibration: { threshold: number | null; n_negatives: number; few_negatives: boolean; realised_fpr: number; k?: number | null };
  cv: { in_fold: boolean; auroc?: number | null; recall?: number | null; not_assessable?: boolean; reason?: string[]; groups?: number; rows?: number };
  tests: Record<string, BlendFigures>;
  criteria: { id: string; verdict: string; reasons: { code: string }[] }[];
}

export interface BlendReport {
  format: string;
  report_id: string;
  views: { blend_fit: string; tests: { id: string; held_out: boolean }[]; calibration: { id: string; sample_rows: number } };
  members: { probe_id: string; layer: number; rule: string; sub_question: string | null }[];
  candidates: BlendCandidate[];
  members_own_figures: Record<string, Record<string, BlendFigures>>;
  ranking: { order: string[]; basis: 'held_out_min' | 'in_distribution_only' };
  flags: string[];
  statement: string;
  has_rung: false;
}

export interface BlendEnvelope {
  report_id: string;
  behaviour_key: string;
  model_id: string;
  status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  failure: { code: string; message: string } | null;
  stale: boolean;
  stale_reasons: StaleReason[];
  next: NextCall[];
  members: string[];
  report_sha256: string | null;
  rows_pruned_at: string | null;
  created_at: string | null;
  report?: BlendReport;
}

export interface BlendReportList {
  behaviour_key: string;
  reports: BlendEnvelope[];
  offset: number;
  limit: number;
  next_offset: number | null;
}

export interface SuggestedBlend {
  key: string;
  model: string;
  report_id: string;
  candidate: string;
  record: Record<string, unknown>;
  record_sha256: string;
  history_id: string;
  updated_at: string | null;
  stale: boolean;
  stale_reasons: StaleReason[];
}

export interface SuggestedBlendHistory {
  key: string;
  model: string;
  entries: { id: string; report_id: string; candidate: string; record_sha256: string; actor: string; reason: string | null; created_at: string | null }[];
  next_offset: number | null;
}

// ── S3: family export ───────────────────────────────────────────────────────────────────────

export interface FamilyExportIndex {
  key: string;
  model: string | null;
  members: { probe_id: string; filename: string; sha256: string; size: number; export_route: string }[];
  archive_route: string;
}

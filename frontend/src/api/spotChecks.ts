/**
 * Session spot-check API client (Feature 038, FR-13). Thin: the server's shapes, unchanged.
 */
import { fetchAPI, API_V1_BASE } from './client';

export interface SpotCheckStats {
  n: number; tp: number; fp: number; fn: number; tn: number;
  agreement: number | null;
  agreement_ci: { low: number; high: number; resamples: number; alpha: number } | null;
  kappa: number | null;
}

export interface SpotCheckProbeReport {
  millm_probe_id: string | null;
  newest_turn_only: boolean;
  overall: SpotCheckStats;
  strata: Record<'fired' | 'near_threshold' | 'not_fired', SpotCheckStats>;
  n_parse_failed: number;
  n_not_recorded: number;
  judge_saw: Record<string, number>;
  threshold_revisions: number[];
  disagreements: number;
}

export interface SpotCheckReport {
  statement: string;
  overall_note: string;
  partial: boolean;
  shortfall: Record<string, { requested: number; returned: number; short: number }>;
  probes: Record<string, SpotCheckProbeReport>;
  newest_turn_probes: Record<string, SpotCheckProbeReport>;
}

export interface SpotCheck {
  id: string; behaviour: string; prompt_id: string; prompt_key: string; endpoint: string; model: string;
  status: string; progress: number | null; error_message: string | null; partial: boolean;
  parse_failures: number; metrics: SpotCheckReport | null; pinned: boolean; excerpts_expire_at: string | null;
  created_at: string | null; sample_spec: Record<string, unknown>;
}

export interface SpotCheckItem {
  id: number; session_id: string; request_id: string; turn_index: number | null; stratum: string;
  probe_id: string; window: string; judge_saw: string; probe_score: number | null; threshold: number | null;
  probe_fired: boolean | null; not_recorded: boolean; judged_text: { role: string; content: string }[] | null;
  judge_reply: string | null; judge_value: string | null; judge_positive: boolean | null; parse_failed: boolean;
  disagrees: boolean;
}

export interface StartSpotCheckBody {
  behaviour: string; prompt_id: string; endpoint: string; model: string; size: number;
  since?: string;
}

const BASE = '/probe-monitors/spot-checks';

export const spotChecksApi = {
  availability: () => fetchAPI<{ available: boolean; reason: string | null; url: string | null }>(`${BASE}/availability`),
  list: (behaviour: string) => fetchAPI<{ spot_checks: SpotCheck[] }>(`${BASE}?behaviour=${encodeURIComponent(behaviour)}`),
  get: (id: string) => fetchAPI<SpotCheck>(`${BASE}/${encodeURIComponent(id)}`),
  start: (body: StartSpotCheckBody) => fetchAPI<{ id: string; status: string }>(BASE, { method: 'POST', body: JSON.stringify(body) }),
  items: (id: string) =>
    fetchAPI<{ items: SpotCheckItem[]; total: number }>(`${BASE}/${encodeURIComponent(id)}/items?disagreements_only=true&limit=100`),
  cancel: (id: string) => fetchAPI<unknown>(`${BASE}/${encodeURIComponent(id)}/cancel`, { method: 'POST' }),
  pin: (id: string, pinned: boolean) =>
    fetchAPI<SpotCheck>(`${BASE}/${encodeURIComponent(id)}/pin`, { method: 'POST', body: JSON.stringify({ pinned }) }),
  exportUrl: (id: string) => `${API_V1_BASE}${BASE}/${encodeURIComponent(id)}/disagreements.jsonl`,
};

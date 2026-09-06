/**
 * Circuits API client (Feature 018) — through the shared fetchAPI client
 * (auth header, base URL, typed APIError; review R1 fix #1).
 *
 * List rows are SLIM summaries; full members/edges arrive on detail fetch.
 */

import { fetchAPI, buildQueryString, API_V1_BASE } from './client';
import type {
  Circuit, CircuitSummary,
  CircuitCapture, CircuitCaptureCreate,
  DiscoveryRun, DiscoveryCreate,
  ValidateConfig, ValidationManifest,
} from '../types/circuits';

export const circuitsApi = {
  list: (params?: { promoted?: boolean; min_rung?: number; edge_type?: string;
                    limit?: number; offset?: number }) => {
    // buildQueryString returns WITHOUT a leading '?' (R2 B2 — a bare concat
    // produced /circuitspromoted=true and 404'd the first filtered call).
    const qs = buildQueryString(params ?? {});
    return fetchAPI<{ circuits: CircuitSummary[]; total: number }>(
      `/circuits${qs ? `?${qs}` : ''}`);
  },

  get: (id: string) => fetchAPI<Circuit>(`/circuits/${id}`),

  update: (id: string, body: {
    name?: string; narrative?: string; expected_version?: number;
  }) =>
    fetchAPI<Circuit>(`/circuits/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(body),
    }),

  setPromoted: (id: string, promoted: boolean) =>
    fetchAPI<Circuit>(`/circuits/${id}/promote`, {
      method: 'POST',
      body: JSON.stringify({ promoted }),
    }),

  remove: (id: string) =>
    fetchAPI<{ deleted: string }>(`/circuits/${id}`, { method: 'DELETE' }),

  /** Browser-navigable export URL (Content-Disposition download). */
  exportUrl: (id: string) => `${API_V1_BASE}/circuits/${id}/export`,

  exportSlices: (id: string) =>
    fetchAPI<{ parent_rung: number; parent_rung_language: string; slices: unknown[] }>(
      `/circuits/${id}/export-slices`, { method: 'POST' }),

  importDefinition: (definition: unknown) =>
    fetchAPI<Circuit>('/circuits/import', {
      method: 'POST',
      body: JSON.stringify(definition),
    }),

  // ── Feature 016: Capture ────────────────────────────────────────────────

  listCaptures: (params?: { limit?: number; offset?: number }) => {
    const qs = buildQueryString(params ?? {});
    return fetchAPI<{ captures: CircuitCapture[]; limit: number; offset: number }>(
      `/circuit-capture${qs ? `?${qs}` : ''}`);
  },

  getCapture: (id: string) => fetchAPI<CircuitCapture>(`/circuit-capture/${id}`),

  /** confirm=false → probe + estimate only; confirm=true → full capture. */
  createCapture: (body: CircuitCaptureCreate) =>
    fetchAPI<{ id: string; task_id: string; status: string; confirmed: boolean }>(
      '/circuit-capture', { method: 'POST', body: JSON.stringify(body) }),

  /** Launch the full capture for an 'estimated' run. */
  confirmCapture: (id: string) =>
    fetchAPI<{ id: string; task_id: string; status: string }>(
      `/circuit-capture/${id}/confirm`, { method: 'POST' }),

  cancelCapture: (id: string) =>
    fetchAPI<{ id: string; status: string }>(
      `/circuit-capture/${id}/cancel`, { method: 'POST' }),

  deleteCapture: (id: string) =>
    fetchAPI<{ deleted: string }>(`/circuit-capture/${id}`, { method: 'DELETE' }),

  // ── Feature 016: Discovery ──────────────────────────────────────────────

  listDiscoveries: (params?: { capture_run_id?: string; limit?: number; offset?: number }) => {
    const qs = buildQueryString(params ?? {});
    return fetchAPI<{ discoveries: DiscoveryRun[]; limit: number; offset: number }>(
      `/circuit-discovery${qs ? `?${qs}` : ''}`);
  },

  getDiscovery: (id: string, includeCandidates = true) => {
    const qs = buildQueryString({ include_candidates: includeCandidates });
    return fetchAPI<DiscoveryRun>(`/circuit-discovery/${id}${qs ? `?${qs}` : ''}`);
  },

  createDiscovery: (body: DiscoveryCreate) =>
    fetchAPI<{ id: string; task_id: string; status: string }>(
      '/circuit-discovery', { method: 'POST', body: JSON.stringify(body) }),

  startAttribution: (id: string, body?: { prompt_limit?: number }) =>
    fetchAPI<{ id: string; task_id: string; status: string }>(
      `/circuit-discovery/${id}/attribution`,
      { method: 'POST', body: JSON.stringify(body ?? {}) }),

  cancelDiscovery: (id: string) =>
    fetchAPI<{ id: string; status: string }>(
      `/circuit-discovery/${id}/cancel`, { method: 'POST' }),

  cancelAttribution: (id: string) =>
    fetchAPI<{ id: string; attribution_status: string }>(
      `/circuit-discovery/${id}/attribution/cancel`, { method: 'POST' }),

  deleteDiscovery: (id: string) =>
    fetchAPI<{ deleted: string }>(`/circuit-discovery/${id}`, { method: 'DELETE' }),

  // ── Feature 017: Validation + manifests ─────────────────────────────────

  /** Validate the top-K edges of a completed discovery run (rung-2 causal
   *  tier). 409 if not completed / no candidates / a pass is already running;
   *  422 on a bad ordering. */
  startValidation: (id: string, body: ValidateConfig) =>
    fetchAPI<{ id: string; task_id: string; status: string }>(
      `/circuit-discovery/${id}/validate`,
      { method: 'POST', body: JSON.stringify(body) }),

  cancelValidation: (id: string) =>
    fetchAPI<{ id: string; validation_status: string }>(
      `/circuit-discovery/${id}/validate/cancel`, { method: 'POST' }),

  getManifest: (id: string) =>
    fetchAPI<ValidationManifest>(`/validation-manifests/${id}`),

  listManifests: (params?: { discovery_run_id?: string; circuit_id?: string }) => {
    const qs = buildQueryString(params ?? {});
    return fetchAPI<{ manifests: ValidationManifest[] }>(
      `/validation-manifests${qs ? `?${qs}` : ''}`);
  },

  reproduceManifest: (id: string) =>
    fetchAPI<{ reproduce_of: string; task_id: string; status: string }>(
      `/validation-manifests/${id}/reproduce`, { method: 'POST' }),

  // ── Feature 017: Faithfulness (rung 3) ──────────────────────────────────

  /** Launch a faithfulness run for a promoted/discovered circuit (rung-3 causal
   *  tier: necessity ± sufficiency). 409 if the circuit has no members / no
   *  discovery_run_id / a run is already in flight. */
  startFaithfulness: (
    id: string,
    body?: {
      mode?: 'necessity' | 'both';
      k_nonmembers?: number;
      ablate_all_n?: number;
      n_prompts?: number;
      seed?: number;
    },
  ) =>
    fetchAPI<{ circuit_id: string; task_id: string; status: string }>(
      `/circuits/${id}/faithfulness`,
      { method: 'POST', body: JSON.stringify(body ?? {}) }),

  // Feature 20: calibrate the usable steering band + clamp the served dial.
  startCalibration: (
    id: string,
    body: {
      judge_endpoint: string;
      judge_model: string;
      step_budget?: number;
      probe_count?: number;
      margin?: number;
      seed?: number;
    },
  ) =>
    fetchAPI<{ circuit_id: string; task_id: string; status: string }>(
      `/circuits/${id}/calibration`,
      { method: 'POST', body: JSON.stringify(body) }),
};

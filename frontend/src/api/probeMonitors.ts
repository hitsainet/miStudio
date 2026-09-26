/**
 * Probe monitor API client (Feature 032).
 *
 * Thin. Every response is returned as the backend sends it — no reshaping, for the
 * reason `api/jlens.ts` records: an adaptation layer here would let the panel drift into
 * a frontend-only shape while still appearing to conform.
 *
 * ⚠ NO CLIENT-SIDE SCORING, RANKING OR RUNG DERIVATION. `GET /probes/{id}` already
 * assembles the report, including the rung's wording and the dense↔SAE pair. Anything
 * this module computed would be a second definition of a number the server publishes.
 */

import { fetchAPI } from './client';
import type {
  ProbeDataset,
  ProbeDefinition,
  ProbeEvaluation,
  ProbeJudgeRun,
  ProbeMonitorSummary,
  ProbeReport,
  ProbeRun,
  ProbeRunAccepted,
} from '../types/probeMonitor';

const BASE = '/probe-monitors';

export interface CreateProbeDatasetBody {
  name: string;
  dataset_id: string;
  config?: string | null;
  split?: string | null;
  input_column: string;
  label_column: string;
  label_mapping: Record<string, 'positive' | 'negative' | 'excluded'>;
  keyword_filter?: { terms: string[]; mode?: 'any' | 'all'; case_sensitive?: boolean } | null;
  pair_column?: string | null;
  role?: 'train' | 'eval' | 'calibration';
  distribution?: 'in_distribution' | 'out_of_distribution' | null;
}

export interface SubmitProbeRunBody {
  model_id: string;
  train_dataset_id: string;
  eval_dataset_ids?: string[];
  calibration_dataset_id?: string | null;
  config?: Record<string, unknown>;
  gpu?: string;
}

export const probeMonitorsApi = {
  listDatasets: (role?: string) =>
    fetchAPI<ProbeDataset[]>(
      `${BASE}/datasets${role ? `?role=${encodeURIComponent(role)}` : ''}`
    ),

  createDataset: (body: CreateProbeDatasetBody) =>
    fetchAPI<ProbeDataset>(`${BASE}/datasets`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  deleteDataset: (datasetId: string) =>
    fetchAPI<void>(`${BASE}/datasets/${encodeURIComponent(datasetId)}`, {
      method: 'DELETE',
    }),

  listRuns: (status?: string) =>
    fetchAPI<ProbeRun[]>(
      `${BASE}/runs${status ? `?status=${encodeURIComponent(status)}` : ''}`
    ),

  getRun: (runId: string) => fetchAPI<ProbeRun>(`${BASE}/runs/${encodeURIComponent(runId)}`),

  submitRun: (body: SubmitProbeRunBody) =>
    fetchAPI<ProbeRunAccepted>(`${BASE}/runs`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  /** Cooperative: the worker stops at its next stage boundary. */
  cancelRun: (runId: string) =>
    fetchAPI<{ id: string; status: string }>(
      `${BASE}/runs/${encodeURIComponent(runId)}/cancel`,
      { method: 'POST' }
    ),

  deleteRun: (runId: string) =>
    fetchAPI<void>(`${BASE}/runs/${encodeURIComponent(runId)}`, { method: 'DELETE' }),

  listProbes: (runId?: string, selectedOnly = false) => {
    const params = new URLSearchParams();
    if (runId) params.set('run_id', runId);
    if (selectedOnly) params.set('selected_only', 'true');
    const query = params.toString();
    return fetchAPI<ProbeMonitorSummary[]>(`${BASE}/probes${query ? `?${query}` : ''}`);
  },

  /** The assembled report — evaluations, the rung WITH its wording, the SAE pair. */
  getReport: (probeId: string) =>
    fetchAPI<ProbeReport>(`${BASE}/probes/${encodeURIComponent(probeId)}`),

  evaluate: (probeId: string, datasetIds: string[]) =>
    fetchAPI<{ probe_id: string; task_id: string; status: string }>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/evaluate`,
      { method: 'POST', body: JSON.stringify(datasetIds) }
    ),

  score: (probeId: string, input: { text?: string; messages?: Array<Record<string, string>> }) =>
    fetchAPI<{ probe_id: string; task_id: string; status: string }>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/score`,
      { method: 'POST', body: JSON.stringify(input) }
    ),

  listJudgeRuns: (probeId?: string) =>
    fetchAPI<ProbeJudgeRun[]>(
      `${BASE}/judge-runs${probeId ? `?probe_id=${encodeURIComponent(probeId)}` : ''}`
    ),

  submitJudgeRun: (body: {
    endpoint: string;
    model: string;
    dataset_ids: string[];
    probe_id?: string | null;
    max_rows_per_set?: number;
    parse_failure_limit?: number;
  }) =>
    fetchAPI<ProbeRunAccepted>(`${BASE}/judge-runs`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  cancelJudgeRun: (judgeRunId: string) =>
    fetchAPI<{ id: string; status: string }>(
      `${BASE}/judge-runs/${encodeURIComponent(judgeRunId)}/cancel`,
      { method: 'POST' }
    ),

  // ── 033: the portable definition ─────────────────────────────────────────

  /**
   * Build and cache the probe's `mistudio.probe-definition/v1`. A GPU job (202) — the
   * definition's test vectors are real forward passes through the same scoring path that
   * produced the probe's metrics.
   *
   * `acknowledgeReason` is REQUIRED below rung 2 and is written into the exported document,
   * so whoever serves the probe can see the claim was made on a judgement.
   */
  buildDefinition: (
    probeId: string,
    options: { acknowledgeReason?: string; vectorCount?: number; seed?: number } = {}
  ) =>
    fetchAPI<ProbeRunAccepted>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/definition`,
      {
        method: 'POST',
        body: JSON.stringify({
          ...(options.acknowledgeReason
            ? { acknowledge_below_rung2: { reason: options.acknowledgeReason } }
            : {}),
          vector_count: options.vectorCount ?? 16,
          seed: options.seed ?? 1337,
        }),
      }
    ),

  /** The built definition as JSON. 409 when none exists, or when one was invalidated. */
  getDefinition: (probeId: string) =>
    fetchAPI<ProbeDefinition>(`${BASE}/probes/${encodeURIComponent(probeId)}/definition`),

  /**
   * The download URL. Returned rather than fetched: a file download is the browser's job, and
   * routing 2 MB through `fetchAPI` only to re-serialise it would be slower and lose the
   * server's filename.
   */
  exportUrl: (probeId: string) =>
    `/api/v1/probe-monitors/probes/${encodeURIComponent(probeId)}/export`,

  /**
   * Publish to HuggingFace (202).
   *
   * ⚠ NO TOKEN PARAMETER. It is resolved server-side from Settings → API Keys, so no credential
   * is held in the browser or sent in this body. A 401 comes back before any work is queued when
   * none is stored.
   */
  publishDefinition: (probeId: string, body: { repo_id: string; private?: boolean }) =>
    fetchAPI<ProbeRunAccepted>(`${BASE}/probes/${encodeURIComponent(probeId)}/publish`, {
      method: 'POST',
      body: JSON.stringify({ private: true, ...body }),
    }),
};

export type { ProbeEvaluation };

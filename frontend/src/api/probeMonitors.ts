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
  JudgePrompt,
  JudgePromptContent,
  JudgePromptList,
  JudgePromptPreviewResult,
  JudgePromptTryResult,
  MiDataworksRubric,
  ProbeDataset,
  ProbeDatasetPurgeBatch,
  ProbeDatasetPurgePlan,
  ProbeDatasetRowsPage,
  ProbeRowOutcome,
  ProbeScoreTask,
  ProbeDefinition,
  ProbeEvaluation,
  ProbeJudgeRun,
  ProbeMonitorSummary,
  ProbeRecalibration,
  ProbeReport,
  ProbeRun,
  ProbeRunAccepted,
  SubmitJudgeRunBody,
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
  role?: 'train' | 'eval' | 'calibration' | 'everyday_text' | 'nonsense_text';
  distribution?: 'in_distribution' | 'out_of_distribution' | null;
}

export interface SubmitProbeRunBody {
  model_id: string;
  train_dataset_id: string;
  eval_dataset_ids?: string[];
  calibration_dataset_id?: string | null;
  config?: Record<string, unknown>;
  gpu?: string;
  /** What the probes detect: a short key (`<behaviour-key>`) and its words. Rung 3 matches the key. */
  behaviour?: string | null;
  behaviour_label?: string | null;
}

/** Fields of a judge prompt a create, edit, new version or clone may send. */
export interface JudgePromptFields extends Partial<JudgePromptContent> {
  name?: string;
  title?: string;
  description?: string | null;
  behaviour?: string;
  behaviour_label?: string | null;
}

/** Exactly one of a saved version or an unsaved draft, plus the rows to use. */
export type PromptOrDraft =
  | { prompt_id: string; draft?: undefined }
  | { prompt_id?: undefined; draft: JudgePromptContent & { name?: string; behaviour?: string } };

const JP = `${BASE}/judge-prompts`;
const jp = (id: string) => `${JP}/${encodeURIComponent(id)}`;

export const probeMonitorsApi = {
  /** Archived views are included: run tiles still need their names, and the Datasets tab
   * filters by its own "Show archived" toggle. New-run pickers filter them out. */
  listDatasets: (role?: string) => {
    const params = new URLSearchParams({ include_archived: 'true' });
    if (role) params.set('role', role);
    return fetchAPI<ProbeDataset[]>(`${BASE}/datasets?${params.toString()}`);
  },

  createDataset: (body: CreateProbeDatasetBody) =>
    fetchAPI<ProbeDataset>(`${BASE}/datasets`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  archiveDataset: (datasetId: string, reason?: string) =>
    fetchAPI<ProbeDataset>(`${BASE}/datasets/${encodeURIComponent(datasetId)}/archive`, {
      method: 'POST',
      body: JSON.stringify(reason ? { reason } : {}),
    }),

  unarchiveDataset: (datasetId: string) =>
    fetchAPI<ProbeDataset>(`${BASE}/datasets/${encodeURIComponent(datasetId)}/unarchive`, {
      method: 'POST',
    }),

  /** What deleting each view would remove and keep, and everything blocking it. Changes nothing. */
  planDatasetPurge: (ids: string[]) =>
    fetchAPI<{ items: ProbeDatasetPurgePlan[] }>(`${BASE}/datasets/purge/plan`, {
      method: 'POST',
      body: JSON.stringify({ ids }),
    }).then((body) => body.items),

  /** Permanently delete exactly these ARCHIVED views; each answers on its own. */
  purgeDatasets: (ids: string[], reason?: string) =>
    fetchAPI<ProbeDatasetPurgeBatch>(`${BASE}/datasets/purge`, {
      method: 'POST',
      body: JSON.stringify(reason ? { ids, reason } : { ids }),
    }),

  datasetRows: (
    datasetId: string,
    query: {
      offset?: number;
      limit?: number;
      columns?: string[] | null;
      label?: ProbeRowOutcome | null;
      search?: string | null;
    } = {}
  ) => {
    const params = new URLSearchParams();
    params.set('offset', String(query.offset ?? 0));
    params.set('limit', String(query.limit ?? 50));
    if (query.columns) params.set('columns', query.columns.join(','));
    if (query.label) params.set('label', query.label);
    if (query.search) params.set('search', query.search);
    return fetchAPI<ProbeDatasetRowsPage>(
      `${BASE}/datasets/${encodeURIComponent(datasetId)}/rows?${params.toString()}`
    );
  },

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

  /**
   * Move the decision threshold to another false-positive budget, or preview the move.
   *
   * ⚠ 200, NOT 202, AND THAT IS THE POINT. A threshold is the `(1 - target_fpr)` quantile of the
   * negative scores the run already saved, so this is arithmetic over an array — not a GPU job.
   * It used to cost a ~2.6-hour run, which is why every calibration question was settled by
   * argument instead of measurement.
   *
   * `preview` defaults TRUE server-side and is sent explicitly here, because a dial whose
   * default is "commit" is a dial that gets turned without being read. The response carries the
   * candidate bar's effect on every evaluation set beside the current bar's.
   */
  recalibrate: (
    probeId: string,
    body: {
      target_fpr: number;
      preview: boolean;
      reason?: string;
      allow_fire_on_nothing?: boolean;
    }
  ) =>
    fetchAPI<ProbeRecalibration>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/recalibrate`,
      { method: 'POST', body: JSON.stringify(body) }
    ),

  score: (probeId: string, input: { text?: string; messages?: Array<Record<string, string>> }) =>
    fetchAPI<{ probe_id: string; task_id: string; status: string }>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/score`,
      { method: 'POST', body: JSON.stringify(input) }
    ),

  /**
   * The other half of `score`'s contract.
   *
   * ⚠ THE ROUTE EXISTED AND THIS CLIENT HAD NO METHOD FOR IT, so scoring was write-only from the
   * frontend: you could queue a score and had no way to read it back. The endpoint's own docstring
   * warns about precisely this shape one layer down — it was added because the POST alone made the
   * feature "exercised and never observed". The client then reproduced it.
   *
   * `status` is Celery's: SUCCESS carries `result`, FAILURE carries `error`, and PENDING means
   * either queued or an id Celery has never heard of — which it cannot tell apart, so neither can
   * this.
   */
  getScoreResult: (probeId: string, taskId: string) =>
    fetchAPI<ProbeScoreTask>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/score/${encodeURIComponent(taskId)}`
    ),

  listJudgeRuns: (probeId?: string) =>
    fetchAPI<ProbeJudgeRun[]>(
      `${BASE}/judge-runs${probeId ? `?probe_id=${encodeURIComponent(probeId)}` : ''}`
    ),

  submitJudgeRun: (body: SubmitJudgeRunBody) =>
    fetchAPI<ProbeRunAccepted & { counts_toward_rung?: boolean | null; rung_note?: string | null }>(`${BASE}/judge-runs`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  cancelJudgeRun: (judgeRunId: string) =>
    fetchAPI<{ id: string; status: string }>(
      `${BASE}/judge-runs/${encodeURIComponent(judgeRunId)}/cancel`,
      { method: 'POST' }
    ),

  // ── the judge-prompt library (PADR IDL-57) ───────────────────────────────

  listJudgePrompts: (options: { behaviour?: string; includeArchived?: boolean } = {}) => {
    const params = new URLSearchParams();
    if (options.behaviour) params.set('behaviour', options.behaviour);
    if (options.includeArchived) params.set('include_archived', 'true');
    const query = params.toString();
    return fetchAPI<JudgePromptList>(`${JP}${query ? `?${query}` : ''}`);
  },

  /** One version with every version of its name (`versions`) and its "used by N runs". */
  getJudgePrompt: (promptId: string) => fetchAPI<JudgePrompt>(jp(promptId)),

  createJudgePrompt: (body: JudgePromptFields) =>
    fetchAPI<JudgePrompt>(JP, { method: 'POST', body: JSON.stringify(body) }),

  /** In place. Content only while no run has used the version (409 `prompt_locked` after). */
  updateJudgePrompt: (promptId: string, body: JudgePromptFields) =>
    fetchAPI<JudgePrompt>(jp(promptId), { method: 'PATCH', body: JSON.stringify(body) }),

  newJudgePromptVersion: (promptId: string, body: JudgePromptFields) =>
    fetchAPI<JudgePrompt>(`${jp(promptId)}/versions`, { method: 'POST', body: JSON.stringify(body) }),

  cloneJudgePrompt: (promptId: string, body: JudgePromptFields & { name: string }) =>
    fetchAPI<JudgePrompt>(`${jp(promptId)}/clone`, { method: 'POST', body: JSON.stringify(body) }),

  archiveJudgePrompt: (promptId: string, archived: boolean) =>
    fetchAPI<JudgePrompt>(`${jp(promptId)}/${archived ? 'archive' : 'unarchive'}`, { method: 'POST' }),

  deleteJudgePrompt: (promptId: string) => fetchAPI<void>(jp(promptId), { method: 'DELETE' }),

  /** The judge's exact messages for a few rows. No model is called. */
  previewJudgePrompt: (body: PromptOrDraft & { dataset_id: string; rows?: number }) =>
    fetchAPI<JudgePromptPreviewResult>(`${JP}/preview`, { method: 'POST', body: JSON.stringify(body) }),

  /** A few rows through the judge. Records nothing and never loads a model. */
  tryJudgePrompt: (
    body: PromptOrDraft & { dataset_id: string; endpoint: string; model: string; rows?: number }
  ) => fetchAPI<JudgePromptTryResult>(`${JP}/try`, { method: 'POST', body: JSON.stringify(body) }),

  exportJudgePrompt: (promptId: string) => fetchAPI<Record<string, unknown>>(`${jp(promptId)}/export`),

  importJudgePrompt: (document: Record<string, unknown>) =>
    fetchAPI<JudgePrompt & { created: boolean }>(`${JP}/import`, {
      method: 'POST',
      body: JSON.stringify({ document }),
    }),

  /** 503 when miDataworks is not connected — the dialog then offers a file instead. */
  listMiDataworksRubrics: () =>
    fetchAPI<{ rubrics: MiDataworksRubric[] }>(`${JP}/midataworks/rubrics`),

  importRubric: (body: {
    rubric_id?: string;
    document?: Record<string, unknown>;
    name: string;
    title: string;
    behaviour: string;
    behaviour_label?: string | null;
    field_map?: Record<string, string>;
    literals?: Record<string, string>;
    positive_verdicts?: string[];
    positive_from?: number;
  }) => fetchAPI<JudgePrompt>(`${JP}/import-rubric`, { method: 'POST', body: JSON.stringify(body) }),

  /** Say what a probe detects; the rung is recomputed and returned. */
  setProbeBehaviour: (probeId: string, behaviour: string | null, behaviourLabel: string | null) =>
    fetchAPI<{ probe_id: string; behaviour: string | null; rung: { rung: number } }>(
      `${BASE}/probes/${encodeURIComponent(probeId)}/behaviour`,
      { method: 'PATCH', body: JSON.stringify({ behaviour, behaviour_label: behaviourLabel }) }
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

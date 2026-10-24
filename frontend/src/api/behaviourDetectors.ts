/**
 * Behaviour-detector reports API client (Feature 040 S2; FTDD §5.2). Thin: every response is the server's
 * shape. Each method has a caller in `stores/behaviourDetectorsStore.ts`, and each store action a caller
 * in a component (`api/__tests__/probeClientIsReachable.test.ts`, derived pairs).
 */
import { fetchAPI } from './client';
import type {
  BreakdownQuery,
  ModelComparison,
  ProbeBreakdown,
  SuccessReportEnvelope,
  SuccessReportList,
  SuccessReportRequest,
} from '../types/probeMonitor';

const PM = '/probe-monitors';
const enc = encodeURIComponent;

export const behaviourDetectorsApi = {
  updateViewColumns: (datasetId: string, groupColumns: string[]) =>
    fetchAPI<{ view_id: string; group_columns: string[]; added: string[] }>(`${PM}/datasets/${enc(datasetId)}/columns`, {
      method: 'PATCH',
      body: JSON.stringify({ group_columns: groupColumns }),
    }),
  getBreakdown: (probeId: string, evaluationId: string, query: BreakdownQuery) => {
    const params = new URLSearchParams({ by_column: query.by_column });
    for (const [column, value] of Object.entries(query.filters ?? {})) params.append('filter', `${column}=${value}`);
    if (query.mode) params.set('mode', query.mode);
    if (query.target_fpr != null) params.set('target_fpr', String(query.target_fpr));
    if (query.min_rows != null) params.set('min_rows', String(query.min_rows));
    if (query.limit != null) params.set('limit', String(query.limit));
    if (query.cursor) params.set('cursor', query.cursor);
    return fetchAPI<ProbeBreakdown>(`${PM}/probes/${enc(probeId)}/evaluations/${enc(evaluationId)}/breakdown?${params.toString()}`);
  },
  createSuccessReport: (key: string, body: SuccessReportRequest) =>
    fetchAPI<SuccessReportEnvelope>(`${PM}/behaviours/${enc(key)}/success-reports`, {
      method: 'POST',
      body: JSON.stringify(body),
    }),
  listSuccessReports: (key: string, cursor?: string | null) =>
    fetchAPI<SuccessReportList>(`${PM}/behaviours/${enc(key)}/success-reports${cursor ? `?cursor=${enc(cursor)}` : ''}`),
  readSuccessReport: (reportId: string) => fetchAPI<SuccessReportEnvelope>(`${PM}/success-reports/${enc(reportId)}`),
  compareModels: (key: string, query: { version?: number; dataset_version?: string } = {}) => {
    const params = new URLSearchParams();
    if (query.version != null) params.set('version', String(query.version));
    if (query.dataset_version) params.set('dataset_version', query.dataset_version);
    const qs = params.toString();
    return fetchAPI<ModelComparison>(`${PM}/behaviours/${enc(key)}/model-comparison${qs ? `?${qs}` : ''}`);
  },
};

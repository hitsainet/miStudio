/**
 * Probe-family and blend-report API client (Feature 041; FTID §6). Thin: every response is the server's
 * shape. Each method has a caller in `stores/probeFamiliesStore.ts`, and each store action a caller in a
 * component (`api/__tests__/probeClientIsReachable.test.ts`, derived pairs).
 */
import { fetchAPI } from './client';
import type {
  BlendEnvelope,
  BlendReportList,
  FamilyExportIndex,
  FamilyQuery,
  ProbeFamily,
  SuggestedBlend,
  SuggestedBlendHistory,
} from '../types/probeFamily';

const PM = '/probe-monitors';
const enc = encodeURIComponent;

export const probeFamiliesApi = {
  getFamily: async (key: string, query: FamilyQuery = {}) => {
    const params = new URLSearchParams();
    if (query.model) params.set('model', query.model);
    for (const id of query.member_ids ?? []) params.append('member_ids', id);
    if (query.include_retired) params.set('include_retired', 'true');
    if (query.offset) params.set('offset', String(query.offset));
    if (query.limit != null) params.set('limit', String(query.limit));
    const qs = params.toString();
    return fetchAPI<ProbeFamily>(`${PM}/behaviours/${enc(key)}/family${qs ? `?${qs}` : ''}`);
  },
  listReports: async (key: string, model?: string | null) => {
    const params = new URLSearchParams({ behaviour_key: key });
    if (model) params.set('model', model);
    return fetchAPI<BlendReportList>(`${PM}/blend-reports?${params.toString()}`);
  },
  getReport: async (reportId: string) => fetchAPI<BlendEnvelope>(`${PM}/blend-reports/${enc(reportId)}`),
  getSuggestedBlend: async (key: string, model: string) =>
    fetchAPI<SuggestedBlend>(`${PM}/behaviours/${enc(key)}/suggested-blend?model=${enc(model)}`),
  getSuggestedBlendHistory: async (key: string, model: string) =>
    fetchAPI<SuggestedBlendHistory>(`${PM}/behaviours/${enc(key)}/suggested-blend/history?model=${enc(model)}`),
  setSuggestedBlend: async (key: string, body: { model: string; report_id: string; candidate: string; reason: string | null }) =>
    fetchAPI<SuggestedBlend>(`${PM}/behaviours/${enc(key)}/suggested-blend`, {
      method: 'PUT',
      body: JSON.stringify(body.reason ? body : { model: body.model, report_id: body.report_id, candidate: body.candidate }),
    }),
  pruneReport: async (reportId: string) =>
    fetchAPI<{ report_id: string; rows_pruned_at: string | null }>(`${PM}/blend-reports/${enc(reportId)}/prune-rows`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  exportIndex: async (key: string, model?: string | null) =>
    fetchAPI<FamilyExportIndex>(`${PM}/behaviours/${enc(key)}/family/export-index${model ? `?model=${enc(model)}` : ''}`),
  /** The ZIP's URL (a download link, not a fetch): the browser saves the untouched bytes. */
  exportUrl: (key: string, model?: string | null) =>
    `/api/v1${PM}/behaviours/${enc(key)}/family/export${model ? `?model=${enc(model)}` : ''}`,
  publishFamily: async (key: string, body: { model: string | null; repo_id: string; private: boolean }) =>
    fetchAPI<{ task_id: string; members: string[] }>(`${PM}/behaviours/${enc(key)}/family/publish`, {
      method: 'POST',
      body: JSON.stringify(body.model ? body : { repo_id: body.repo_id, private: body.private }),
    }),
};

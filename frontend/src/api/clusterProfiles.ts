/**
 * API client for cluster profiles (Feature 014).
 */

import {
  ClusterProfile,
  ClusterProfileCreate,
  ClusterProfileListResponse,
  ClusterProfileUpdate,
  ImportResponse,
} from '../types/clusterProfile';
import { fetchAPI, buildQueryString } from './client';

export const clusterProfilesApi = {
  list: (params?: { sae_id?: string; search?: string }): Promise<ClusterProfileListResponse> => {
    const qs = buildQueryString(params ?? {});
    return fetchAPI<ClusterProfileListResponse>(`/cluster-profiles${qs ? `?${qs}` : ''}`);
  },

  get: (id: string): Promise<ClusterProfile> => fetchAPI<ClusterProfile>(`/cluster-profiles/${id}`),

  create: (data: ClusterProfileCreate): Promise<ClusterProfile> =>
    fetchAPI<ClusterProfile>('/cluster-profiles', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  update: (id: string, data: ClusterProfileUpdate): Promise<ClusterProfile> =>
    fetchAPI<ClusterProfile>(`/cluster-profiles/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  delete: (id: string): Promise<{ message: string }> =>
    fetchAPI<{ message: string }>(`/cluster-profiles/${id}`, { method: 'DELETE' }),

  /** Export ONE profile as a portable definition (already JSON-shaped). */
  exportDefinition: (id: string): Promise<Record<string, unknown>> =>
    fetchAPI<Record<string, unknown>>(`/cluster-profiles/${id}/export`),

  exportBundle: (ids: string[]): Promise<Record<string, unknown>> =>
    fetchAPI<Record<string, unknown>>('/cluster-profiles/export-bundle', {
      method: 'POST',
      body: JSON.stringify({ ids }),
    }),

  import: (payload: Record<string, unknown>, bindSaeId?: string | null): Promise<ImportResponse> =>
    fetchAPI<ImportResponse>('/cluster-profiles/import', {
      method: 'POST',
      body: JSON.stringify({ payload, bind_sae_id: bindSaeId ?? null }),
    }),
};

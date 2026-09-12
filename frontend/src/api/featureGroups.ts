/**
 * API client for cross-feature grouping + agent approvals (Feature 010).
 */

import { fetchAPI } from './client';
import type {
  ApprovalRequest,
  FeatureGroupDetail,
  FeatureGroupListResponse,
  GroupingParams,
  GroupingStatusResponse,
  RelatedFeaturesResponse,
} from '../types/featureGroups';

export async function computeFeatureGroups(
  extractionId: string,
  params: GroupingParams = {},
  force = false
): Promise<{ task_id: string | null; run_id: string; status: string; message: string }> {
  return fetchAPI(`/extractions/${extractionId}/feature-groups/compute`, {
    method: 'POST',
    body: JSON.stringify({ params, force }),
  });
}

export async function getGroupingStatus(extractionId: string): Promise<GroupingStatusResponse> {
  return fetchAPI(`/extractions/${extractionId}/feature-groups/status`);
}

export async function getFeatureGroups(
  extractionId: string,
  options: {
    token?: string;
    search?: string;
    min_group_size?: number;
    sort_by?: 'size' | 'cohesion' | 'token';
    limit?: number;
    offset?: number;
  } = {}
): Promise<FeatureGroupListResponse> {
  const params = new URLSearchParams();
  Object.entries(options).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value));
  });
  const query = params.toString();
  return fetchAPI(`/extractions/${extractionId}/feature-groups${query ? `?${query}` : ''}`);
}

export async function getGroupMembers(
  extractionId: string,
  groupId: string,
  filters: { category?: string; has_label?: boolean; star_color?: string } = {}
): Promise<FeatureGroupDetail> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== '') params.set(key, String(value));
  });
  const query = params.toString();
  return fetchAPI(
    `/extractions/${extractionId}/feature-groups/${groupId}${query ? `?${query}` : ''}`
  );
}

export async function getRelatedFeatures(
  featureId: string,
  minSimilarity = 0.2,
  limit = 25
): Promise<RelatedFeaturesResponse> {
  return fetchAPI(
    `/features/${featureId}/related?min_similarity=${minSimilarity}&limit=${limit}`
  );
}

// ── Agent approvals (operator-approval mode) ────────────────────────────────

export async function listApprovals(status?: string): Promise<{ approvals: ApprovalRequest[]; total: number }> {
  return fetchAPI(`/mcp/approvals${status ? `?status=${status}` : ''}`);
}

export async function approveRequest(requestId: string): Promise<ApprovalRequest> {
  return fetchAPI(`/mcp/approvals/${requestId}/approve`, { method: 'POST' });
}

export async function denyRequest(requestId: string, reason?: string): Promise<ApprovalRequest> {
  return fetchAPI(`/mcp/approvals/${requestId}/deny`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason ?? null }),
  });
}

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

/**
 * The waiting list. ENH-006 D81: asks for up to 500 (the route's maximum), not the route's default 50, so the
 * banner count and "Approve all code rules (N)" count every waiting request, not the first 50.
 */
export async function listApprovals(status?: string): Promise<{ approvals: ApprovalRequest[]; total: number }> {
  return fetchAPI(`/mcp/approvals?limit=${LIST_LIMIT}${status ? `&status=${status}` : ''}`);
}

/** The approvals list route's maximum `limit`. */
export const LIST_LIMIT = 500;

/** ENH-008: the Approvals page's History — decided requests, the most recent decision first. */
export const HISTORY_PAGE = 200;

export async function listDecidedApprovals(
  offset = 0,
  limit = HISTORY_PAGE,
): Promise<{ approvals: ApprovalRequest[]; total: number }> {
  return fetchAPI(`/mcp/approvals?decided=true&limit=${limit}&offset=${offset}`);
}

/** ENH-008 (D4): a code hash's CURRENT approval state, read when its card is shown. */
export async function getCodeRuleState(
  codeSha256: string,
): Promise<{ code_sha256: string; state: 'approved' | 'revoked' | 'pending' | 'unknown'; revoked_row_id?: number | null }> {
  return fetchAPI(`/code-rules/${encodeURIComponent(codeSha256)}`);
}

export async function approveRequest(requestId: string): Promise<ApprovalRequest> {
  return fetchAPI(`/mcp/approvals/${requestId}/approve`, { method: 'POST' });
}

/** ENH-006 D81: what one batch approve did, per request (the server's order is the order sent). */
export interface ApproveBatchResult {
  batch_id: string;
  kind: ApproveBatchKind;
  results: { id: string; status: 'approved' | 'failed'; error?: string | null; approval?: ApprovalRequest | null }[];
}

/** The batch kinds. miStudio's queue holds code rules; it has no settings-change request (D79). */
export type ApproveBatchKind = 'code_rules';

/**
 * ENH-006 D81: approve every listed request of ONE kind in one operator call. The server refuses the whole batch
 * (409, naming them) when any id is not a pending request of that kind, before approving anything.
 */
export async function approveBatch(kind: ApproveBatchKind, approvalIds: string[]): Promise<ApproveBatchResult> {
  return fetchAPI('/mcp/approvals/approve-batch', {
    method: 'POST',
    body: JSON.stringify({ kind, approval_ids: approvalIds }),
  });
}

export async function denyRequest(requestId: string, reason?: string): Promise<ApprovalRequest> {
  return fetchAPI(`/mcp/approvals/${requestId}/deny`, {
    method: 'POST',
    body: JSON.stringify({ reason: reason ?? null }),
  });
}

/**
 * Types for cross-feature grouping (Feature 010).
 * Mirrors backend/src/schemas/feature_group.py.
 */

export type GroupingIndexStatus = 'none' | 'pending' | 'computing' | 'completed' | 'failed';

export interface GroupingParams {
  context_window?: number;
  similarity_threshold?: number;
  top_examples?: number;
  min_group_size?: number;
}

export interface GroupingStatusResponse {
  status: GroupingIndexStatus;
  run_id: string | null;
  progress: number | null;
  params: Record<string, unknown> | null;
  feature_count: number | null;
  group_count: number | null;
  error_message: string | null;
  computed_at: string | null;
}

export interface FeatureGroupSummary {
  group_id: string;
  normalized_token: string;
  display_token: string;
  member_count: number;
  cohesion: number;
  sample_labels: string[];
}

export interface FeatureGroupListResponse {
  groups: FeatureGroupSummary[];
  total: number;
  limit: number;
  offset: number;
  index_status: string;
}

export interface FeatureGroupMember {
  feature_id: string;
  neuron_index: number;
  name: string;
  category: string | null;
  label_source: string | null;
  star_color: string | null;
  is_favorite: boolean;
  max_activation: number | null;
  activation_frequency: number | null;
  similarity: number;
  context_snippet: string | null;
}

export interface FeatureGroupDetail {
  group_id: string;
  normalized_token: string;
  display_token: string;
  cohesion: number;
  member_count: number;
  members: FeatureGroupMember[];
}

export interface RelatedFeature {
  feature_id: string;
  neuron_index: number | null;
  name: string | null;
  category: string | null;
  score: number;
  link_types: Array<'shared_token' | 'context' | 'correlation'>;
}

export interface RelatedFeaturesResponse {
  seed_feature_id: string;
  related: RelatedFeature[];
}

export interface FeatureGroupsProgressEvent {
  extraction_id: string;
  progress: number;
  stage: string;
}

export interface ApprovalRequest {
  id: string;
  tool_name: string;
  payload: Record<string, unknown>;
  status: 'pending' | 'approved' | 'denied' | 'expired';
  reason: string | null;
  steering_task_id: string | null;
  created_at: string | null;
  resolved_at: string | null;
}

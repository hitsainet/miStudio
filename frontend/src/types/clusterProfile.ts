/**
 * Cluster profile types (Feature 014, IDL-30) — mirrors
 * backend/src/schemas/cluster_profile.py.
 *
 * Profiles are durable, user-authored snapshots (name + narrative + tuned
 * strengths) decoupled from the recomputable grouping index. The interchange
 * kinds (`mistudio.cluster-definition/v1`) are the portable artifacts.
 */

export interface ProfileMember {
  feature_idx: number;
  label?: string | null;
  similarity?: number | null;
  activation_frequency?: number | null;
  max_activation?: number | null;
  strength: number;
  sign?: 1 | -1;
  pinned?: boolean;
}

export interface ProfileBudget {
  B?: number | null;
  B_dir?: number | null;
  G?: number | null;
  f_eff?: number | null;
  formula_id?: string | null;
  constants?: Record<string, number> | null;
  intensity?: number;
  intensity_range?: number[];
}

export interface ClusterProfile {
  id: string;
  sae_id: string | null;
  model_id: string | null;
  extraction_id: string | null;
  source_group_id: string | null;
  name: string;
  narrative: string | null;
  display_token: string | null;
  members: ProfileMember[];
  budget: ProfileBudget | null;
  schema_version: string;
  imported_from: Record<string, unknown> | null;
  created_at: string;
  updated_at: string;
}

export interface ClusterProfileCreate {
  sae_id?: string | null;
  model_id?: string | null;
  extraction_id?: string | null;
  source_group_id?: string | null;
  name: string;
  narrative?: string | null;
  display_token?: string | null;
  members: ProfileMember[];
  budget?: ProfileBudget | null;
}

export interface ClusterProfileUpdate {
  name?: string;
  narrative?: string | null;
  members?: ProfileMember[];
  budget?: ProfileBudget | null;
}

export interface ClusterProfileListResponse {
  data: ClusterProfile[];
  total: number;
}

export interface ImportItemResult {
  name: string;
  status: 'imported' | 'imported_unbound' | 'blocked' | 'error';
  profile_id?: string | null;
  warnings: string[];
  error?: string | null;
}

export interface ImportResponse {
  results: ImportItemResult[];
  imported: number;
  blocked: number;
  errors: number;
}

export const PROFILE_NAME_MAX = 120;
export const PROFILE_NARRATIVE_MAX = 10_000;
export const PROFILE_MAX_MEMBERS = 20;

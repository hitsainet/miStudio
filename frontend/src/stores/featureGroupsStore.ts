/**
 * Zustand store for the Feature Groups view (Feature 010).
 *
 * WebSocket-first with polling fallback (standard store pattern): while the
 * grouping job runs, progress arrives on extractions/{id}/feature-groups; if
 * the socket is down, a 3s status poll takes over until it reconnects.
 */

import { create } from 'zustand';
import {
  computeFeatureGroups,
  getFeatureGroups,
  getGroupingStatus,
  getGroupMembers,
  getRelatedFeatures,
} from '../api/featureGroups';
import type {
  FeatureGroupDetail,
  FeatureGroupSummary,
  GroupingParams,
  GroupingStatusResponse,
  RelatedFeaturesResponse,
} from '../types/featureGroups';

interface GroupFilters {
  search: string;
  minGroupSize: number;
  sortBy: 'size' | 'cohesion' | 'token';
}

// A selected group member, carrying the stats the steering hand-off needs.
export interface SelectedMember {
  neuron_index: number;
  max_activation: number | null;
  activation_frequency: number | null;
  /**
   * Provenance stamp (Feature 012): the cluster this member was selected FROM,
   * recorded at toggle time. Makes the steering hand-off's cluster identity
   * independent of which cluster happens to be expanded when "Steer selected"
   * is clicked.
   */
  group_id: string;
  display_token: string;
  /** Member context similarity — feeds the 013 strength allocation. */
  similarity: number | null;
  /** Source cluster cohesion (group-level) — feeds the 013 cohesion gate. */
  cohesion: number | null;
}

/**
 * Derive single-cluster provenance from selection stamps (Features 012/013).
 * Returns the cluster identity only when EVERY member was selected from the
 * same cluster and the display token is non-empty — mixed or unstamped
 * selections yield null (no false cluster claims).
 */
export function deriveSourceCluster(
  members: SelectedMember[],
): { group_id: string; display_token: string } | null {
  const firstGroup = members[0]?.group_id;
  const token = members[0]?.display_token?.trim();
  if (!firstGroup || !token) return null;
  return members.every((m) => m.group_id === firstGroup)
    ? { group_id: firstGroup, display_token: token }
    : null;
}

interface FeatureGroupsState {
  extractionId: string | null;
  status: GroupingStatusResponse | null;
  computeProgress: { progress: number; stage: string } | null;

  groups: FeatureGroupSummary[];
  groupsTotal: number;
  groupsOffset: number;
  filters: GroupFilters;

  expandedGroupId: string | null;
  groupDetail: FeatureGroupDetail | null;

  relatedFor: string | null;
  related: RelatedFeaturesResponse | null;

  // feature_id → selection info, for steering hand-off. Carries the stats the
  // steering auto-baseline needs (Feature 011); they'd otherwise be lost here.
  selection: Map<string, SelectedMember>;

  isLoading: boolean;
  error: string | null;
  isWebSocketConnected: boolean;
  pollTimer: ReturnType<typeof setInterval> | null;

  setExtraction: (extractionId: string | null) => void;
  fetchStatus: () => Promise<void>;
  computeIndex: (params?: GroupingParams, force?: boolean) => Promise<void>;
  fetchGroups: (offset?: number) => Promise<void>;
  setFilters: (filters: Partial<GroupFilters>) => void;
  expandGroup: (groupId: string | null) => Promise<void>;
  fetchRelated: (featureId: string) => Promise<void>;
  clearRelated: () => void;
  toggleSelect: (featureId: string, member: SelectedMember) => void;
  setSelected: (
    members: Array<{ feature_id: string } & SelectedMember>,
    selected: boolean,
  ) => void;
  clearSelection: () => void;

  handleProgressEvent: (progress: number, stage: string) => void;
  handleCompletedEvent: () => void;
  handleFailedEvent: (error: string) => void;
  setWebSocketConnected: (connected: boolean) => void;
}

const DEFAULT_FILTERS: GroupFilters = { search: '', minGroupSize: 2, sortBy: 'size' };
const PAGE_SIZE = 50;

export const useFeatureGroupsStore = create<FeatureGroupsState>((set, get) => ({
  extractionId: null,
  status: null,
  computeProgress: null,
  groups: [],
  groupsTotal: 0,
  groupsOffset: 0,
  filters: { ...DEFAULT_FILTERS },
  expandedGroupId: null,
  groupDetail: null,
  relatedFor: null,
  related: null,
  selection: new Map<string, SelectedMember>(),
  isLoading: false,
  error: null,
  isWebSocketConnected: false,
  pollTimer: null,

  setExtraction: (extractionId) => {
    const { pollTimer } = get();
    if (pollTimer) clearInterval(pollTimer);
    set({
      extractionId,
      status: null,
      computeProgress: null,
      groups: [],
      groupsTotal: 0,
      groupsOffset: 0,
      expandedGroupId: null,
      groupDetail: null,
      related: null,
      relatedFor: null,
      selection: new Map(),
      error: null,
      pollTimer: null,
    });
    if (extractionId) {
      void get().fetchStatus();
    }
  },

  fetchStatus: async () => {
    const { extractionId } = get();
    if (!extractionId) return;
    try {
      const status = await getGroupingStatus(extractionId);
      set({ status, error: null });
      if (status.status === 'completed' && get().groups.length === 0) {
        void get().fetchGroups();
      }
      // Poll while computing and the socket is down
      const { pollTimer, isWebSocketConnected } = get();
      const active = status.status === 'computing' || status.status === 'pending';
      if (active && !isWebSocketConnected && !pollTimer) {
        const timer = setInterval(() => void get().fetchStatus(), 3000);
        set({ pollTimer: timer });
      } else if (!active && pollTimer) {
        clearInterval(pollTimer);
        set({ pollTimer: null, computeProgress: null });
      }
    } catch (error: any) {
      set({ error: error.detail || error.message || 'Failed to load grouping status' });
    }
  },

  computeIndex: async (params = {}, force = false) => {
    const { extractionId } = get();
    if (!extractionId) return;
    set({ error: null });
    try {
      const response = await computeFeatureGroups(extractionId, params, force);
      if (response.status === 'completed') {
        // Idempotent short-circuit — index already there
        await get().fetchStatus();
        return;
      }
      set({ computeProgress: { progress: 0, stage: 'queued' } });
      await get().fetchStatus();
    } catch (error: any) {
      set({ error: error.detail?.message || error.detail || error.message || 'Failed to start job' });
    }
  },

  fetchGroups: async (offset = 0) => {
    const { extractionId, filters } = get();
    if (!extractionId) return;
    set({ isLoading: true });
    try {
      const response = await getFeatureGroups(extractionId, {
        search: filters.search || undefined,
        min_group_size: filters.minGroupSize,
        sort_by: filters.sortBy,
        limit: PAGE_SIZE,
        offset,
      });
      set({
        groups: response.groups,
        groupsTotal: response.total,
        groupsOffset: offset,
        isLoading: false,
        error: null,
      });
    } catch (error: any) {
      set({ isLoading: false, error: error.detail || error.message || 'Failed to load groups' });
    }
  },

  setFilters: (filters) => {
    set((state) => ({ filters: { ...state.filters, ...filters } }));
    void get().fetchGroups(0);
  },

  expandGroup: async (groupId) => {
    const { extractionId, expandedGroupId } = get();
    if (!extractionId || !groupId || groupId === expandedGroupId) {
      set({ expandedGroupId: null, groupDetail: null });
      return;
    }
    set({ expandedGroupId: groupId, groupDetail: null });
    try {
      const detail = await getGroupMembers(extractionId, groupId);
      // Only apply if still the expanded group
      if (get().expandedGroupId === groupId) set({ groupDetail: detail });
    } catch (error: any) {
      set({ error: error.detail || error.message || 'Failed to load group members' });
    }
  },

  fetchRelated: async (featureId) => {
    set({ relatedFor: featureId, related: null });
    try {
      const related = await getRelatedFeatures(featureId);
      if (get().relatedFor === featureId) set({ related });
    } catch (error: any) {
      set({ error: error.detail || error.message || 'Failed to load related features' });
    }
  },

  clearRelated: () => set({ relatedFor: null, related: null }),

  toggleSelect: (featureId, member) => {
    set((state) => {
      const selection = new Map(state.selection);
      if (selection.has(featureId)) selection.delete(featureId);
      else selection.set(featureId, member);
      return { selection };
    });
  },

  setSelected: (members, selected) => {
    set((state) => {
      const selection = new Map(state.selection);
      for (const m of members) {
        if (selected) {
          selection.set(m.feature_id, {
            neuron_index: m.neuron_index,
            max_activation: m.max_activation,
            activation_frequency: m.activation_frequency,
            group_id: m.group_id,
            display_token: m.display_token,
            similarity: m.similarity,
            cohesion: m.cohesion,
          });
        } else {
          selection.delete(m.feature_id);
        }
      }
      return { selection };
    });
  },

  clearSelection: () => set({ selection: new Map() }),

  handleProgressEvent: (progress, stage) => set({ computeProgress: { progress, stage } }),

  handleCompletedEvent: () => {
    set({ computeProgress: null });
    void get().fetchStatus();
    void get().fetchGroups(0);
  },

  handleFailedEvent: (error) => {
    set({ computeProgress: null, error });
    void get().fetchStatus();
  },

  setWebSocketConnected: (connected) => {
    set({ isWebSocketConnected: connected });
    const { pollTimer, status } = get();
    if (connected && pollTimer) {
      clearInterval(pollTimer);
      set({ pollTimer: null });
    } else if (!connected && (status?.status === 'computing' || status?.status === 'pending')) {
      void get().fetchStatus(); // restarts polling
    }
  },
}));

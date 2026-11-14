/**
 * Probe-family store (Feature 041; FTDD §6.8, T33). Holds the selected model, the family page, the blend
 * reports, the open report and the suggested blend the behaviour screen shows. No optimistic updates: every
 * value is the server's, re-read after each write. A refusal is kept as data (code, message, `next`) and, for
 * `model_required`, the models the server listed — so the screen offers the model picker the refusal asks for.
 */
import { create } from 'zustand';

import { APIError } from '../api/client';
import { probeFamiliesApi } from '../api/probeFamilies';
import type {
  BlendEnvelope,
  FamilyExportIndex,
  BlendReportList,
  ProbeFamily,
  SuggestedBlend,
  SuggestedBlendHistory,
} from '../types/probeFamily';
import { refusalOf, type Refusal } from './behaviourDetectorsStore';

interface ProbeFamiliesState {
  key: string | null;
  model: string | null;
  family: ProbeFamily | null;
  /** The models a `model_required` refusal listed, with their member counts. */
  models: { model_id: string; members: number }[];
  includeRetired: boolean;
  loading: boolean;
  busy: boolean;
  refusal: Refusal | null;
  reports: BlendReportList | null;
  openReport: BlendEnvelope | null;
  suggested: SuggestedBlend | null;
  history: SuggestedBlendHistory | null;
  exportIndex: FamilyExportIndex | null;
  published: { task_id: string; members: string[] } | null;

  loadFamily: (key: string, model?: string | null, offset?: number, includeRetired?: boolean) => Promise<void>;
  loadReports: (key: string, model: string | null) => Promise<void>;
  openReportById: (reportId: string) => Promise<void>;
  loadSuggested: (key: string, model: string) => Promise<void>;
  markSuggested: (key: string, model: string, reportId: string, candidate: string, reason: string | null) => Promise<void>;
  pruneReport: (reportId: string) => Promise<void>;
  loadExportIndex: (key: string, model: string | null) => Promise<void>;
  archiveUrl: (key: string, model: string | null) => string;
  publishFamily: (key: string, model: string | null, repoId: string, isPrivate: boolean) => Promise<boolean>;
}

export const useProbeFamiliesStore = create<ProbeFamiliesState>()((set, get) => ({
  key: null,
  model: null,
  family: null,
  models: [],
  includeRetired: false,
  loading: false,
  busy: false,
  refusal: null,
  reports: null,
  openReport: null,
  suggested: null,
  history: null,
  exportIndex: null,
  published: null,

  loadFamily: async (key, model = null, offset = 0, includeRetired) => {
    const retired = includeRetired ?? (get().key === key ? get().includeRetired : false);
    set({ loading: true, refusal: null, key, model, includeRetired: retired });
    try {
      const family = await probeFamiliesApi.getFamily(key, { model, offset, include_retired: retired });
      set({ family, loading: false, model: family.model, models: family.models });
    } catch (error) {
      const models =
        error instanceof APIError && Array.isArray(error.data?.models)
          ? (error.data?.models as { model_id: string; members: number }[])
          : [];
      set({ loading: false, family: null, refusal: refusalOf(error), models });
    }
  },

  loadReports: async (key, model) => {
    try {
      set({ reports: await probeFamiliesApi.listReports(key, model) });
    } catch (error) {
      set({ refusal: refusalOf(error) });
    }
  },

  openReportById: async (reportId) => {
    try {
      set({ openReport: await probeFamiliesApi.getReport(reportId) });
    } catch (error) {
      set({ refusal: refusalOf(error) });
    }
  },

  loadSuggested: async (key, model) => {
    try {
      const history = await probeFamiliesApi.getSuggestedBlendHistory(key, model);
      let suggested: SuggestedBlend | null = null;
      if (history.entries.length) suggested = await probeFamiliesApi.getSuggestedBlend(key, model);
      set({ suggested, history });
    } catch (error) {
      set({ refusal: refusalOf(error) });
    }
  },

  markSuggested: async (key, model, reportId, candidate, reason) => {
    set({ busy: true, refusal: null });
    try {
      await probeFamiliesApi.setSuggestedBlend(key, { model, report_id: reportId, candidate, reason });
      set({ busy: false });
      await get().loadSuggested(key, model);
    } catch (error) {
      set({ busy: false, refusal: refusalOf(error) });
    }
  },

  pruneReport: async (reportId) => {
    set({ busy: true, refusal: null });
    try {
      await probeFamiliesApi.pruneReport(reportId);
      set({ busy: false });
      await get().openReportById(reportId);
    } catch (error) {
      set({ busy: false, refusal: refusalOf(error) });
    }
  },

  loadExportIndex: async (key, model) => {
    set({ refusal: null, exportIndex: null });
    try {
      set({ exportIndex: await probeFamiliesApi.exportIndex(key, model) });
    } catch (error) {
      set({ refusal: refusalOf(error) });
    }
  },

  archiveUrl: (key, model) => probeFamiliesApi.exportUrl(key, model),

  publishFamily: async (key, model, repoId, isPrivate) => {
    set({ busy: true, refusal: null, published: null });
    try {
      const published = await probeFamiliesApi.publishFamily(key, { model, repo_id: repoId, private: isPrivate });
      set({ busy: false, published });
      return true;
    } catch (error) {
      set({ busy: false, refusal: refusalOf(error) });
      return false;
    }
  },
}));

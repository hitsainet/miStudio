/**
 * Behaviour-detector reports store (Feature 040 S2; FTDD §7, 040-T39).
 *
 * Holds what the report surfaces show: the last breakdown per (probe, evaluation, column), success reports
 * per behaviour, and the model comparison. Every value is the server's; nothing is computed or invented
 * here. A refusal is kept as data (`refusal`: code, message, next) so a screen renders its `next` action
 * rather than a bare error string (FR-40).
 *
 * Each action calls exactly one client method and is called by a component
 * (`api/__tests__/probeClientIsReachable.test.ts` derives the pairs and checks both links).
 */
import { create } from 'zustand';

import { behaviourDetectorsApi } from '../api/behaviourDetectors';
import { APIError } from '../api/client';
import type {
  BreakdownQuery,
  ModelComparison,
  NextCall,
  ProbeBreakdown,
  SuccessReportEnvelope,
  SuccessReportList,
  SuccessReportRequest,
} from '../types/probeMonitor';

export interface Refusal {
  code: string;
  message: string;
  next: NextCall[];
}

export function refusalOf(error: unknown): Refusal {
  if (error instanceof APIError && error.data) {
    return {
      code: String(error.data.code ?? 'refused'),
      message: String(error.data.message ?? error.detail),
      next: (error.data.next as NextCall[] | undefined) ?? [],
    };
  }
  return { code: 'error', message: error instanceof Error ? error.message : String(error), next: [] };
}

interface BehaviourDetectorsState {
  breakdown: ProbeBreakdown | null;
  report: SuccessReportEnvelope | null;
  reports: SuccessReportList | null;
  comparison: ModelComparison | null;
  busy: boolean;
  refusal: Refusal | null;

  fetchBreakdown: (probeId: string, evaluationId: string, query: BreakdownQuery) => Promise<void>;
  fetchSuccessReport: (key: string, body: SuccessReportRequest) => Promise<void>;
  readSuccessReport: (reportId: string) => Promise<void>;
  listSuccessReports: (key: string) => Promise<void>;
  fetchModelComparison: (key: string) => Promise<void>;
  updateViewColumns: (datasetId: string, groupColumns: string[]) => Promise<boolean>;
}

export const useBehaviourDetectorsStore = create<BehaviourDetectorsState>()((set) => {
  async function run<T>(work: () => Promise<T>, apply: (value: T) => Partial<BehaviourDetectorsState>): Promise<boolean> {
    set({ busy: true, refusal: null });
    try {
      const value = await work();
      set({ ...apply(value), busy: false });
      return true;
    } catch (error) {
      set({ busy: false, refusal: refusalOf(error) });
      return false;
    }
  }

  return {
    breakdown: null,
    report: null,
    reports: null,
    comparison: null,
    busy: false,
    refusal: null,

    fetchBreakdown: async (probeId, evaluationId, query) => {
      await run(() => behaviourDetectorsApi.getBreakdown(probeId, evaluationId, query), (breakdown) => ({ breakdown }));
    },
    fetchSuccessReport: async (key, body) => {
      await run(() => behaviourDetectorsApi.createSuccessReport(key, body), (report) => ({ report }));
    },
    readSuccessReport: async (reportId) => {
      await run(() => behaviourDetectorsApi.readSuccessReport(reportId), (report) => ({ report }));
    },
    listSuccessReports: async (key) => {
      await run(() => behaviourDetectorsApi.listSuccessReports(key), (reports) => ({ reports }));
    },
    fetchModelComparison: async (key) => {
      await run(() => behaviourDetectorsApi.compareModels(key), (comparison) => ({ comparison }));
    },
    updateViewColumns: (datasetId, groupColumns) =>
      run(() => behaviourDetectorsApi.updateViewColumns(datasetId, groupColumns), () => ({})),
  };
});

/**
 * Probe Monitors panel store (Feature 032).
 *
 * Three properties are load-bearing and each has a test pinning it:
 *
 *  1. NO SYNTHETIC DATA, ANYWHERE. There is no fixture builder, no seeded report and no
 *     placeholder AUROC. A synthetic number is indistinguishable from a measured one once
 *     rendered, and this panel's whole job is to report measurements honestly.
 *  2. A REFETCH NEVER BLANKS WHAT IS ON SCREEN. `isLoading` is set without clearing the
 *     lists or the open report, so a background refresh cannot unmount a report the user
 *     is reading — the house regression already fixed in ExtractionsPanel.
 *  3. A REFUSAL IS CARRIED THROUGH, NOT NORMALISED. `metrics.scored === false` keeps its
 *     `reason`, and nothing here substitutes 0 or 0.5. The components render the reason.
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import { probeMonitorsApi, type CreateProbeDatasetBody, type SubmitProbeRunBody } from '../api/probeMonitors';
import type {
  ProbeDataset,
  ProbeJudgeRun,
  ProbeMonitorSummary,
  ProbeReport,
  ProbeRun,
} from '../types/probeMonitor';

export type ProbeTab = 'datasets' | 'runs' | 'probes' | 'judge';

interface ProbeMonitorsState {
  activeTab: ProbeTab;
  datasets: ProbeDataset[];
  runs: ProbeRun[];
  probes: ProbeMonitorSummary[];
  judgeRuns: ProbeJudgeRun[];
  report: ProbeReport | null;
  openProbeId: string | null;

  isLoading: boolean;
  error: string | null;

  setActiveTab: (tab: ProbeTab) => void;
  loadDatasets: (role?: string) => Promise<void>;
  createDataset: (body: CreateProbeDatasetBody) => Promise<ProbeDataset | null>;
  deleteDataset: (id: string) => Promise<void>;
  loadRuns: () => Promise<void>;
  submitRun: (body: SubmitProbeRunBody) => Promise<string | null>;
  cancelRun: (id: string) => Promise<void>;
  deleteRun: (id: string) => Promise<void>;
  loadProbes: (runId?: string) => Promise<void>;
  openReport: (probeId: string) => Promise<void>;
  closeReport: () => void;
  loadJudgeRuns: (probeId?: string) => Promise<void>;
  /** Merge a WebSocket progress event into the run list without a refetch. */
  applyRunProgress: (runId: string, patch: Partial<ProbeRun>) => void;
  clearError: () => void;
}

function message(error: unknown): string {
  if (error instanceof Error) return error.message;
  return String(error);
}

export const useProbeMonitorsStore = create<ProbeMonitorsState>()(
  devtools(
    (set, get) => ({
      activeTab: 'runs',
      datasets: [],
      runs: [],
      probes: [],
      judgeRuns: [],
      report: null,
      openProbeId: null,
      isLoading: false,
      error: null,

      setActiveTab: (tab) => set({ activeTab: tab }),

      loadDatasets: async (role) => {
        // isLoading WITHOUT clearing `datasets`: see property 2 in the module docstring.
        set({ isLoading: true, error: null });
        try {
          set({ datasets: await probeMonitorsApi.listDatasets(role), isLoading: false });
        } catch (error) {
          set({ error: message(error), isLoading: false });
        }
      },

      createDataset: async (body) => {
        set({ isLoading: true, error: null });
        try {
          const created = await probeMonitorsApi.createDataset(body);
          set((state) => ({
            datasets: [created, ...state.datasets],
            isLoading: false,
          }));
          return created;
        } catch (error) {
          // A 422 here is the server REFUSING an unscoreable view, and its detail names
          // both class counts. Surfaced verbatim: rewording it would lose the numbers a
          // user needs to fix the mapping.
          set({ error: message(error), isLoading: false });
          return null;
        }
      },

      deleteDataset: async (id) => {
        try {
          await probeMonitorsApi.deleteDataset(id);
          set((state) => ({ datasets: state.datasets.filter((d) => d.id !== id) }));
        } catch (error) {
          set({ error: message(error) });
        }
      },

      loadRuns: async () => {
        set({ isLoading: true, error: null });
        try {
          set({ runs: await probeMonitorsApi.listRuns(), isLoading: false });
        } catch (error) {
          set({ error: message(error), isLoading: false });
        }
      },

      submitRun: async (body) => {
        set({ isLoading: true, error: null });
        try {
          const accepted = await probeMonitorsApi.submitRun(body);
          await get().loadRuns();
          set({ isLoading: false });
          return accepted.id;
        } catch (error) {
          set({ error: message(error), isLoading: false });
          return null;
        }
      },

      cancelRun: async (id) => {
        try {
          await probeMonitorsApi.cancelRun(id);
          // "cancelling", not "cancelled": the worker stops at its next stage boundary,
          // and claiming it has already stopped would be a lie the UI tells for seconds.
          set((state) => ({
            runs: state.runs.map((run) =>
              run.id === id ? { ...run, status: 'cancelling' } : run
            ),
          }));
        } catch (error) {
          set({ error: message(error) });
        }
      },

      deleteRun: async (id) => {
        try {
          await probeMonitorsApi.deleteRun(id);
          set((state) => ({ runs: state.runs.filter((run) => run.id !== id) }));
        } catch (error) {
          set({ error: message(error) });
        }
      },

      loadProbes: async (runId) => {
        set({ isLoading: true, error: null });
        try {
          set({ probes: await probeMonitorsApi.listProbes(runId), isLoading: false });
        } catch (error) {
          set({ error: message(error), isLoading: false });
        }
      },

      openReport: async (probeId) => {
        set({ isLoading: true, error: null, openProbeId: probeId });
        try {
          set({ report: await probeMonitorsApi.getReport(probeId), isLoading: false });
        } catch (error) {
          set({ error: message(error), isLoading: false });
        }
      },

      closeReport: () => set({ report: null, openProbeId: null }),

      loadJudgeRuns: async (probeId) => {
        try {
          set({ judgeRuns: await probeMonitorsApi.listJudgeRuns(probeId) });
        } catch (error) {
          set({ error: message(error) });
        }
      },

      applyRunProgress: (runId, patch) =>
        set((state) => ({
          runs: state.runs.map((run) => (run.id === runId ? { ...run, ...patch } : run)),
        })),

      clearError: () => set({ error: null }),
    }),
    { name: 'probe-monitors' }
  )
);

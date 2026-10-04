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
import { devtools, persist } from 'zustand/middleware';
import { probeMonitorsApi, type CreateProbeDatasetBody, type SubmitProbeRunBody } from '../api/probeMonitors';
import type {
  ProbeRecalibration,
  ProbeScoreTask,
  SubmitJudgeRunBody,
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
  /**
   * ⚠ THE FOUR BELOW EXISTED IN `api/probeMonitors.ts` AND NOTHING CALLED THEM.
   *
   * An audit of the client's nineteen methods against their callers found four with none:
   * `submitJudgeRun`, `cancelJudgeRun`, `evaluate` and `score`. The consequence was not
   * cosmetic — the evidence ladder was **unclimbable from the UI**. Rung 1 and 2 need
   * `evaluate` (more sets on an existing probe) and rung 3 needs a judge run, so a probe could
   * only ever be advanced by hand with `curl`. `score` is the other one: without it there is no
   * way to point a probe at your own text and see what it says.
   *
   * A client method with no caller is the unregistered-MCP-tool defect wearing a third hat: the
   * code is written, tested and unreachable.
   */
  submitJudgeRun: (body: SubmitJudgeRunBody) => Promise<string | null>;
  cancelJudgeRun: (id: string) => Promise<void>;
  evaluateProbe: (probeId: string, datasetIds: string[]) => Promise<boolean>;
  /**
   * Preview or commit a threshold move. Returns the whole proposal, including what the candidate
   * bar does to every evaluation set, so the caller can show the consequence before committing.
   *
   * ⚠ ON COMMIT THE REPORT IS RELOADED, NOT PATCHED. A re-cut changes the threshold, the realised
   * FPR, the per-window bars, the per-length table AND whether the cached definition is still
   * valid; patching the fields a component happens to know about is how a panel ends up showing
   * a new threshold beside an old operating-point source.
   */
  recalibrateProbe: (
    probeId: string,
    body: {
      target_fpr: number;
      preview: boolean;
      reason?: string;
      allow_fire_on_nothing?: boolean;
    }
  ) => Promise<ProbeRecalibration | null>;
  scoreProbe: (
    probeId: string,
    input: { text?: string; messages?: Array<Record<string, string>> }
  ) => Promise<ProbeScoreTask | null>;
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
    persist(
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

      submitJudgeRun: async (body) => {
        set({ isLoading: true, error: null });
        try {
          const accepted = await probeMonitorsApi.submitJudgeRun(body);
          await get().loadJudgeRuns();
          set({ isLoading: false });
          return accepted.id ?? null;
        } catch (error) {
          set({ error: message(error), isLoading: false });
          return null;
        }
      },

      cancelJudgeRun: async (id) => {
        try {
          await probeMonitorsApi.cancelJudgeRun(id);
          await get().loadJudgeRuns();
        } catch (error) {
          set({ error: message(error) });
        }
      },

      evaluateProbe: async (probeId, datasetIds) => {
        set({ isLoading: true, error: null });
        try {
          await probeMonitorsApi.evaluate(probeId, datasetIds);
          set({ isLoading: false });
          return true;
        } catch (error) {
          set({ error: message(error), isLoading: false });
          return false;
        }
      },

      recalibrateProbe: async (probeId, body) => {
        set({ error: null });
        try {
          const outcome = await probeMonitorsApi.recalibrate(probeId, body);
          if (!body.preview) {
            // The bar actually moved, so everything derived from it is stale here too.
            await get().openReport(probeId);
            await get().loadProbes();
          }
          return outcome;
        } catch (error) {
          set({ error: message(error) });
          return null;
        }
      },

      scoreProbe: async (probeId, input) => {
        set({ error: null });
        try {
          return await probeMonitorsApi.score(probeId, input);
        } catch (error) {
          set({ error: message(error) });
          return null;
        }
      },

      applyRunProgress: (runId, patch) =>
        set((state) => ({
          runs: state.runs.map((run) => (run.id === runId ? { ...run, ...patch } : run)),
        })),

      clearError: () => set({ error: null }),
    }),
      {
        name: 'probe-monitors',
        /*
         * ⚠ ONLY THE SUB-TAB. A refresh used to land on Runs whatever you were looking at, which
         * is a reload discarding a navigation choice for no reason.
         *
         * Nothing else here is persisted, and that is deliberate: `runs`, `probes`, `datasets`,
         * `judgeRuns` and `report` are all SERVER state with live status in them. A persisted
         * copy would render a run as `running` after it had finished, or show a probe that has
         * since been deleted — stale data presented as current, which is worse than a spinner.
         */
        partialize: (state) => ({ activeTab: state.activeTab }),
      }
    )
  )
);

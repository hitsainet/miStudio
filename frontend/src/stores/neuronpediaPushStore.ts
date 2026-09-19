/**
 * Neuronpedia Push Store
 *
 * Zustand store for managing Neuronpedia push job state.
 * Persists push job state across modal open/close cycles,
 * allowing users to put pushes in background and resume viewing later.
 */

import { create } from 'zustand';
import { NeuronpediaPushProgress } from '../hooks/useNeuronpediaPushWebSocket';
import { listActivePushJobs } from '../api/neuronpedia';

/**
 * Active push job state.
 */
export interface ActivePushJob {
  pushJobId: string;
  saeId: string;
  saeName: string;
  startTime: number; // Date.now() when push started
  progress: NeuronpediaPushProgress | null;
  isComplete: boolean;
  error: string | null;
}

interface NeuronpediaPushState {
  // Active push jobs keyed by SAE ID (only one push per SAE at a time)
  activePushJobs: Record<string, ActivePushJob>;

  // Actions
  startPush: (saeId: string, saeName: string, pushJobId: string) => void;
  updateProgress: (saeId: string, progress: NeuronpediaPushProgress) => void;
  completePush: (saeId: string, progress: NeuronpediaPushProgress) => void;
  failPush: (saeId: string, error: string, progress?: NeuronpediaPushProgress) => void;
  clearPush: (saeId: string) => void;
  getPushJob: (saeId: string) => ActivePushJob | null;
  hasPushInProgress: (saeId: string) => boolean;
  fetchActivePushJobs: () => Promise<void>;
}

export const useNeuronpediaPushStore = create<NeuronpediaPushState>((set, get) => ({
  activePushJobs: {},

  /**
   * Start tracking a new push job.
   */
  startPush: (saeId: string, saeName: string, pushJobId: string) => {
    console.log(`[NeuronpediaPushStore] Starting push for SAE ${saeId}: ${pushJobId}`);
    set((state) => ({
      activePushJobs: {
        ...state.activePushJobs,
        [saeId]: {
          pushJobId,
          saeId,
          saeName,
          startTime: Date.now(),
          progress: null,
          isComplete: false,
          error: null,
        },
      },
    }));
  },

  /**
   * Update progress for an active push job.
   */
  updateProgress: (saeId: string, progress: NeuronpediaPushProgress) => {
    const job = get().activePushJobs[saeId];
    if (!job) {
      console.warn(`[NeuronpediaPushStore] No active push for SAE ${saeId}`);
      return;
    }

    set((state) => ({
      activePushJobs: {
        ...state.activePushJobs,
        [saeId]: {
          ...state.activePushJobs[saeId],
          progress,
        },
      },
    }));
  },

  /**
   * Mark a push job as complete.
   */
  completePush: (saeId: string, progress: NeuronpediaPushProgress) => {
    const job = get().activePushJobs[saeId];
    if (!job) {
      console.warn(`[NeuronpediaPushStore] No active push for SAE ${saeId}`);
      return;
    }

    console.log(`[NeuronpediaPushStore] Push completed for SAE ${saeId}`);
    set((state) => ({
      activePushJobs: {
        ...state.activePushJobs,
        [saeId]: {
          ...state.activePushJobs[saeId],
          progress,
          isComplete: true,
          error: null,
        },
      },
    }));
  },

  /**
   * Mark a push job as failed.
   */
  failPush: (saeId: string, error: string, progress?: NeuronpediaPushProgress) => {
    const job = get().activePushJobs[saeId];
    if (!job) {
      console.warn(`[NeuronpediaPushStore] No active push for SAE ${saeId}`);
      return;
    }

    console.log(`[NeuronpediaPushStore] Push failed for SAE ${saeId}: ${error}`);
    set((state) => ({
      activePushJobs: {
        ...state.activePushJobs,
        [saeId]: {
          ...state.activePushJobs[saeId],
          progress: progress || state.activePushJobs[saeId].progress,
          isComplete: true,
          error,
        },
      },
    }));
  },

  /**
   * Clear a push job (after user dismisses completion/error).
   */
  clearPush: (saeId: string) => {
    console.log(`[NeuronpediaPushStore] Clearing push for SAE ${saeId}`);
    set((state) => {
      const { [saeId]: _, ...rest } = state.activePushJobs;
      return { activePushJobs: rest };
    });
  },

  /**
   * Get push job for an SAE.
   */
  getPushJob: (saeId: string) => {
    return get().activePushJobs[saeId] || null;
  },

  /**
   * Check if there's a push in progress for an SAE.
   */
  hasPushInProgress: (saeId: string) => {
    const job = get().activePushJobs[saeId];
    return job !== undefined && !job.isComplete;
  },

  /**
   * Fetch active push jobs from backend and restore store state.
   * Called on page load to survive browser refreshes.
   */
  fetchActivePushJobs: async () => {
    try {
      const jobs = await listActivePushJobs();
      const currentJobs = get().activePushJobs;
      const updates: Record<string, ActivePushJob> = {};

      for (const job of jobs) {
        // Don't overwrite jobs we're already tracking (e.g. just started)
        if (currentJobs[job.saeId]) continue;

        updates[job.saeId] = {
          pushJobId: job.pushJobId,
          saeId: job.saeId,
          saeName: job.saeName || job.saeId,
          startTime: job.createdAt ? new Date(job.createdAt).getTime() : Date.now(),
          progress: {
            push_job_id: job.pushJobId,
            sae_id: job.saeId,
            stage: job.status,
            progress: job.progress,
            status: job.status as NeuronpediaPushProgress['status'],
            features_pushed: job.featuresPushed,
            total_features: job.totalFeatures,
            message: `Processing feature ${job.featuresPushed}/${job.totalFeatures}`,
          },
          isComplete: false,
          error: null,
        };
      }

      if (Object.keys(updates).length > 0) {
        set((state) => ({
          activePushJobs: { ...updates, ...state.activePushJobs },
        }));
      }
    } catch {
      // Non-critical — store stays empty, user can still start new pushes
    }
  },
}));

// Selectors
export const selectActivePushJob = (saeId: string) => (state: NeuronpediaPushState) =>
  state.activePushJobs[saeId] || null;

export const selectHasPushInProgress = (saeId: string) => (state: NeuronpediaPushState) =>
  state.activePushJobs[saeId] !== undefined && !state.activePushJobs[saeId].isComplete;

export const selectAllActivePushes = (state: NeuronpediaPushState) =>
  Object.values(state.activePushJobs).filter((job) => !job.isComplete);

/**
 * Labeling Store
 *
 * Zustand store for independent semantic labeling of SAE features.
 * Manages labeling jobs and real-time updates via WebSocket.
 *
 * Backend API Contract:
 * - POST /api/v1/labeling - Start labeling job
 * - GET /api/v1/labeling/:id - Get labeling job status
 * - GET /api/v1/labeling - List labeling jobs
 * - POST /api/v1/labeling/:id/cancel - Cancel labeling job
 * - DELETE /api/v1/labeling/:id - Delete labeling job
 * - POST /api/v1/extractions/:id/label - Convenience endpoint
 *
 * WebSocket Events:
 * - labeling/{labeling_job_id}/progress - Labeling progress updates
 */

import { create } from 'zustand';
import type {
  LabelingJob,
  LabelingConfigRequest,
} from '../types/labeling';
import { LabelingStatus, LabelingMethod } from '../types/labeling';
import * as labelingAPI from '../api/labeling';

/**
 * Labeling configuration form state.
 * Used for collecting user inputs before creating a labeling job.
 */
export interface LabelingConfig {
  // Target configuration
  extraction_job_id: string;

  // Labeling method
  labeling_method: LabelingMethod;

  // OpenAI configuration
  openai_model?: string;
  openai_api_key?: string;

  // Local model configuration
  local_model?: string;

  // Processing configuration
  batch_size?: number;
}

/**
 * Labeling store state.
 */
interface LabelingStoreState {
  // Labeling jobs list
  labelingJobs: LabelingJob[];

  // Selected labeling job for detail view
  selectedLabelingJob: LabelingJob | null;

  // Labeling configuration form
  config: LabelingConfig;

  // UI state
  isLoading: boolean;
  error: string | null;

  // Pagination
  currentPage: number;
  totalPages: number;
  totalJobs: number;
  limit: number;

  // Filters
  statusFilter: LabelingStatus | 'all';
  extractionFilter: string | null;
  methodFilter: LabelingMethod | 'all';

  // Status counts (calculated from local data)
  statusCounts: {
    all: number;
    queued: number;
    labeling: number;
    completed: number;
    failed: number;
    cancelled: number;
  };
}

/**
 * Labeling store actions.
 */
interface LabelingStoreActions {
  // Labeling CRUD operations
  fetchLabelingJobs: (page?: number, limit?: number) => Promise<void>;
  fetchLabelingJob: (labelingJobId: string) => Promise<void>;
  startLabeling: (config: LabelingConfigRequest) => Promise<LabelingJob>;
  cancelLabeling: (labelingJobId: string) => Promise<void>;
  deleteLabeling: (labelingJobId: string) => Promise<void>;
  retryLabeling: (labelingJobId: string) => Promise<LabelingJob>;

  // Batch operations
  bulkDeleteLabeling: (labelingJobIds: string[]) => Promise<void>;

  // Configuration management
  updateConfig: (updates: Partial<LabelingConfig>) => void;
  resetConfig: () => void;
  setConfigFromLabelingJob: (labelingJob: LabelingJob) => void;

  // Real-time updates (WebSocket handler)
  updateLabelingStatus: (labelingJobId: string, updates: Partial<LabelingJob>) => void;

  // UI state management
  setSelectedLabelingJob: (labelingJob: LabelingJob | null) => void;
  setStatusFilter: (status: LabelingStatus | 'all') => void;
  setExtractionFilter: (extractionId: string | null) => void;
  setMethodFilter: (method: LabelingMethod | 'all') => void;
  clearError: () => void;

  // Helper functions
  getLabelingJobsForExtraction: (extractionJobId: string) => LabelingJob[];
  getLatestLabelingJobForExtraction: (extractionJobId: string) => LabelingJob | null;
  hasActiveLabelingForExtraction: (extractionJobId: string) => boolean;
}

type LabelingStore = LabelingStoreState & LabelingStoreActions;

/**
 * Default labeling configuration.
 * Provides sensible defaults for semantic labeling.
 */
const defaultConfig: LabelingConfig = {
  extraction_job_id: '',
  labeling_method: LabelingMethod.OPENAI,
  openai_model: 'gpt-4o-mini',
  batch_size: 10,
};

/**
 * Calculate status counts from labeling jobs list.
 *
 * @param jobs - List of labeling jobs
 * @returns Status counts object
 */
const calculateStatusCounts = (jobs: LabelingJob[]) => {
  const counts = {
    all: jobs.length,
    queued: 0,
    labeling: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
  };

  jobs.forEach((job) => {
    switch (job.status) {
      case LabelingStatus.QUEUED:
        counts.queued++;
        break;
      case LabelingStatus.LABELING:
        counts.labeling++;
        break;
      case LabelingStatus.COMPLETED:
        counts.completed++;
        break;
      case LabelingStatus.FAILED:
        counts.failed++;
        break;
      case LabelingStatus.CANCELLED:
        counts.cancelled++;
        break;
    }
  });

  return counts;
};

/**
 * Labeling store using Zustand.
 *
 * This store manages the state for the independent semantic labeling feature, including:
 * - Labeling job list and filtering
 * - Labeling configuration form
 * - Real-time updates via WebSocket
 * - Labeling job lifecycle management
 *
 * Usage:
 *   const { labelingJobs, fetchLabelingJobs, startLabeling } = useLabelingStore();
 *
 *   useEffect(() => {
 *     fetchLabelingJobs();
 *   }, []);
 *
 *   const handleStart = async () => {
 *     const labelingJob = await startLabeling({
 *       extraction_job_id: 'extr_abc123',
 *       labeling_method: LabelingMethod.OPENAI,
 *       openai_model: 'gpt-4o-mini'
 *     });
 *   };
 */
export const useLabelingStore = create<LabelingStore>((set, get) => ({
  // Initial state
  labelingJobs: [],
  selectedLabelingJob: null,
  config: { ...defaultConfig },
  isLoading: false,
  error: null,
  currentPage: 1,
  totalPages: 1,
  totalJobs: 0,
  limit: 50,
  statusFilter: 'all',
  extractionFilter: null,
  methodFilter: 'all',
  statusCounts: {
    all: 0,
    queued: 0,
    labeling: 0,
    completed: 0,
    failed: 0,
    cancelled: 0,
  },

  /**
   * Fetch list of labeling jobs.
   *
   * @param page - Page number (1-indexed)
   * @param limit - Number of items per page
   */
  fetchLabelingJobs: async (page = 1, limit = 50) => {
    set({ isLoading: true, error: null });

    try {
      const { statusFilter, extractionFilter, methodFilter } = get();

      // Calculate offset from page number
      const offset = (page - 1) * limit;

      // Fetch labeling jobs
      const response = await labelingAPI.listLabelingJobs({
        extraction_job_id: extractionFilter || undefined,
        limit,
        offset,
      });

      // Client-side filtering by status and method (backend doesn't support these filters yet)
      let filteredJobs = response.data;

      if (statusFilter !== 'all') {
        filteredJobs = filteredJobs.filter((job) => job.status === statusFilter);
      }

      if (methodFilter !== 'all') {
        filteredJobs = filteredJobs.filter(
          (job) => job.labeling_method === methodFilter
        );
      }

      // Calculate pagination
      // const totalFiltered = filteredJobs.length;
      const totalPages = Math.ceil(response.meta.total / limit);

      // Calculate status counts
      const statusCounts = calculateStatusCounts(response.data);

      set({
        labelingJobs: filteredJobs,
        currentPage: page,
        totalPages,
        totalJobs: response.meta.total,
        statusCounts,
        isLoading: false,
      });
    } catch (error: any) {
      console.error('Failed to fetch labeling jobs:', error);
      set({
        error: error.message || 'Failed to fetch labeling jobs',
        isLoading: false,
      });
    }
  },

  /**
   * Fetch a single labeling job by ID.
   *
   * @param labelingJobId - Labeling job ID
   */
  fetchLabelingJob: async (labelingJobId: string) => {
    set({ isLoading: true, error: null });

    try {
      const labelingJob = await labelingAPI.getLabelingJob(labelingJobId);

      // Update the labeling job in the list
      set((state) => ({
        labelingJobs: state.labelingJobs.map((job) =>
          job.id === labelingJobId ? labelingJob : job
        ),
        selectedLabelingJob: labelingJob,
        isLoading: false,
      }));
    } catch (error: any) {
      console.error('Failed to fetch labeling job:', error);
      set({
        error: error.message || 'Failed to fetch labeling job',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Start a new semantic labeling job.
   *
   * @param config - Labeling configuration
   * @returns Created labeling job
   */
  startLabeling: async (config: LabelingConfigRequest) => {
    set({ isLoading: true, error: null });

    try {
      const labelingJob = await labelingAPI.startLabeling(config);

      // Add the new labeling job to the list
      set((state) => {
        const updatedJobs = [labelingJob, ...state.labelingJobs];
        return {
          labelingJobs: updatedJobs,
          totalJobs: state.totalJobs + 1,
          statusCounts: calculateStatusCounts(updatedJobs),
          isLoading: false,
        };
      });

      return labelingJob;
    } catch (error: any) {
      console.error('Failed to start labeling:', error);
      set({
        error: error.message || 'Failed to start labeling',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Cancel an active labeling job.
   *
   * @param labelingJobId - Labeling job ID
   */
  cancelLabeling: async (labelingJobId: string) => {
    set({ isLoading: true, error: null });

    try {
      await labelingAPI.cancelLabeling(labelingJobId);

      // Update the labeling job status in the store
      set((state) => {
        const updatedJobs = state.labelingJobs.map((job) =>
          job.id === labelingJobId
            ? { ...job, status: LabelingStatus.CANCELLED }
            : job
        );
        return {
          labelingJobs: updatedJobs,
          statusCounts: calculateStatusCounts(updatedJobs),
          isLoading: false,
        };
      });
    } catch (error: any) {
      console.error('Failed to cancel labeling:', error);
      set({
        error: error.message || 'Failed to cancel labeling',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Delete a labeling job.
   *
   * Note: This only deletes the labeling job record, not the feature labels.
   *
   * @param labelingJobId - Labeling job ID
   */
  deleteLabeling: async (labelingJobId: string) => {
    set({ isLoading: true, error: null });

    try {
      await labelingAPI.deleteLabeling(labelingJobId);

      // Remove the labeling job from the list
      set((state) => {
        const updatedJobs = state.labelingJobs.filter(
          (job) => job.id !== labelingJobId
        );
        return {
          labelingJobs: updatedJobs,
          totalJobs: state.totalJobs - 1,
          statusCounts: calculateStatusCounts(updatedJobs),
          selectedLabelingJob:
            state.selectedLabelingJob?.id === labelingJobId
              ? null
              : state.selectedLabelingJob,
          isLoading: false,
        };
      });
    } catch (error: any) {
      console.error('Failed to delete labeling:', error);
      set({
        error: error.message || 'Failed to delete labeling',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Retry a failed labeling job by creating a new labeling with the same configuration.
   *
   * @param labelingJobId - Failed labeling job ID
   * @returns New labeling job
   */
  retryLabeling: async (labelingJobId: string) => {
    set({ isLoading: true, error: null });

    try {
      // Get the failed labeling job to extract its configuration
      const state = get();
      const failedJob = state.labelingJobs.find((job) => job.id === labelingJobId);

      if (!failedJob) {
        throw new Error('Labeling job not found');
      }

      // Create new labeling request with same configuration
      const retryRequest: LabelingConfigRequest = {
        extraction_job_id: failedJob.extraction_job_id,
        labeling_method: failedJob.labeling_method,
        openai_model: failedJob.openai_model || undefined,
        local_model: failedJob.local_model || undefined,
        batch_size: 10, // Use default batch size
      };

      // Create the new labeling job
      const labelingJob = await labelingAPI.startLabeling(retryRequest);

      // Add the new labeling job to the list
      set((state) => {
        const updatedJobs = [labelingJob, ...state.labelingJobs];
        return {
          labelingJobs: updatedJobs,
          totalJobs: state.totalJobs + 1,
          statusCounts: calculateStatusCounts(updatedJobs),
          isLoading: false,
        };
      });

      return labelingJob;
    } catch (error: any) {
      console.error('Failed to retry labeling:', error);
      set({
        error: error.message || 'Failed to retry labeling',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Delete multiple labeling jobs at once.
   *
   * @param labelingJobIds - Array of labeling job IDs to delete
   */
  bulkDeleteLabeling: async (labelingJobIds: string[]) => {
    set({ isLoading: true, error: null });

    try {
      // Delete all jobs in parallel
      await Promise.all(
        labelingJobIds.map((id) => labelingAPI.deleteLabeling(id))
      );

      // Remove all deleted jobs from the list
      set((state) => {
        const updatedJobs = state.labelingJobs.filter(
          (job) => !labelingJobIds.includes(job.id)
        );
        return {
          labelingJobs: updatedJobs,
          totalJobs: state.totalJobs - labelingJobIds.length,
          statusCounts: calculateStatusCounts(updatedJobs),
          selectedLabelingJob: labelingJobIds.includes(
            state.selectedLabelingJob?.id || ''
          )
            ? null
            : state.selectedLabelingJob,
          isLoading: false,
        };
      });
    } catch (error: any) {
      console.error('Failed to bulk delete labeling:', error);
      set({
        error: error.message || 'Failed to delete labeling jobs',
        isLoading: false,
      });
      throw error;
    }
  },

  /**
   * Update labeling configuration form.
   *
   * @param updates - Partial configuration updates
   */
  updateConfig: (updates: Partial<LabelingConfig>) => {
    set((state) => ({
      config: { ...state.config, ...updates },
    }));
  },

  /**
   * Reset labeling configuration to defaults.
   */
  resetConfig: () => {
    set({ config: { ...defaultConfig } });
  },

  /**
   * Set configuration from an existing labeling job.
   * Useful for re-running or modifying previous labeling configs.
   *
   * @param labelingJob - Labeling job to copy config from
   */
  setConfigFromLabelingJob: (labelingJob: LabelingJob) => {
    set({
      config: {
        extraction_job_id: labelingJob.extraction_job_id,
        labeling_method: labelingJob.labeling_method,
        openai_model: labelingJob.openai_model || undefined,
        local_model: labelingJob.local_model || undefined,
        batch_size: 10, // Use default batch size
      },
    });
  },

  /**
   * Update labeling status in real-time (WebSocket handler).
   *
   * This function is called when WebSocket events are received
   * to update the labeling job state without refetching from the API.
   *
   * @param labelingJobId - Labeling job ID
   * @param updates - Partial labeling job updates
   */
  updateLabelingStatus: (labelingJobId: string, updates: Partial<LabelingJob>) => {
    set((state) => {
      const updatedJobs = state.labelingJobs.map((job) =>
        job.id === labelingJobId ? { ...job, ...updates } : job
      );
      return {
        labelingJobs: updatedJobs,
        statusCounts: calculateStatusCounts(updatedJobs),
        selectedLabelingJob:
          state.selectedLabelingJob?.id === labelingJobId
            ? { ...state.selectedLabelingJob, ...updates }
            : state.selectedLabelingJob,
      };
    });
  },

  /**
   * Set the selected labeling job for detail view.
   *
   * @param labelingJob - Labeling job to select
   */
  setSelectedLabelingJob: (labelingJob: LabelingJob | null) => {
    set({ selectedLabelingJob: labelingJob });
  },

  /**
   * Set status filter for labeling list.
   *
   * @param status - Status to filter by, or 'all'
   */
  setStatusFilter: (status: LabelingStatus | 'all') => {
    set({ statusFilter: status });
    // Refetch labeling jobs with new filter
    get().fetchLabelingJobs(1);
  },

  /**
   * Set extraction filter for labeling list.
   *
   * @param extractionId - Extraction ID to filter by
   */
  setExtractionFilter: (extractionId: string | null) => {
    set({ extractionFilter: extractionId });
    // Refetch labeling jobs with new filter
    get().fetchLabelingJobs(1);
  },

  /**
   * Set labeling method filter.
   *
   * @param method - Labeling method to filter by, or 'all'
   */
  setMethodFilter: (method: LabelingMethod | 'all') => {
    set({ methodFilter: method });
    // Refetch labeling jobs with new filter
    get().fetchLabelingJobs(1);
  },

  /**
   * Clear error message.
   */
  clearError: () => {
    set({ error: null });
  },

  /**
   * Get all labeling jobs for a specific extraction.
   *
   * @param extractionJobId - Extraction job ID
   * @returns Array of labeling jobs for the extraction
   */
  getLabelingJobsForExtraction: (extractionJobId: string) => {
    return get().labelingJobs.filter(
      (job) => job.extraction_job_id === extractionJobId
    );
  },

  /**
   * Get the most recent labeling job for an extraction.
   *
   * @param extractionJobId - Extraction job ID
   * @returns Latest labeling job or null if none found
   */
  getLatestLabelingJobForExtraction: (extractionJobId: string) => {
    const jobs = get().getLabelingJobsForExtraction(extractionJobId);

    if (jobs.length === 0) {
      return null;
    }

    // Sort by created_at descending and return the first
    return jobs.sort(
      (a, b) =>
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
    )[0];
  },

  /**
   * Check if an extraction has an active labeling job (queued or labeling).
   *
   * @param extractionJobId - Extraction job ID
   * @returns True if there's an active labeling job
   */
  hasActiveLabelingForExtraction: (extractionJobId: string) => {
    const jobs = get().getLabelingJobsForExtraction(extractionJobId);
    return jobs.some(
      (job) =>
        job.status === LabelingStatus.QUEUED ||
        job.status === LabelingStatus.LABELING
    );
  },
}));

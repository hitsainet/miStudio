/**
 * Features Store
 *
 * Zustand store for Feature Discovery state management.
 * Manages feature extraction, search, filtering, and real-time updates via WebSocket.
 *
 * Backend API Contract:
 * - GET /api/v1/extractions - List all extraction jobs
 * - DELETE /api/v1/extractions/:id - Delete extraction job
 * - GET /api/v1/extractions/:id/features - List/search features for an extraction
 * - GET /api/v1/features/:id - Get feature details
 * - PATCH /api/v1/features/:id - Update feature metadata
 * - POST /api/v1/features/:id/favorite - Toggle favorite status
 * - GET /api/v1/features/:id/examples - Get max-activating examples
 *
 * Note: Feature extraction is initiated via SAE API endpoints (see saes.ts).
 * This store focuses on managing and displaying extraction results.
 *
 * WebSocket Events:
 * - extraction:{extraction_id}:progress - Extraction progress updates
 * - extraction:{extraction_id}:completed - Extraction completion
 * - extraction:{extraction_id}:failed - Extraction failure
 */

import { create } from 'zustand';
import axios from 'axios';
import type {
  ExtractionStatusResponse,
  Feature,
  FeatureListResponse,
  FeatureSearchRequest,
  FeatureDetail,
  FeatureUpdateRequest,
  FeatureActivationExample,
  TokenAnalysisResponse,
} from '../types/features';

/**
 * Extract error message from API error response.
 * Handles both Pydantic validation errors (array of objects) and regular errors (string).
 */
function extractErrorMessage(error: any, fallback: string): string {
  const detail = error.response?.data?.detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail.map((e: any) => e.msg || e.message || JSON.stringify(e)).join('; ');
  } else if (typeof detail === 'string') {
    return detail;
  }
  return error.message || fallback;
}

/**
 * Features store state.
 */
interface FeaturesStoreState {
  // Extraction status by training ID
  extractionStatus: Record<string, ExtractionStatusResponse | null>;

  // All extraction jobs (for list view)
  allExtractions: ExtractionStatusResponse[];
  extractionsMetadata: {
    total: number;
    limit: number;
    offset: number;
  } | null;

  // Features by training ID
  featuresByTraining: Record<string, Feature[]>;

  // Features by extraction ID
  featuresByExtraction: Record<string, Feature[]>;

  // Feature list metadata (pagination, statistics)
  featureListMetadata: Record<string, {
    total: number;
    limit: number;
    offset: number;
    statistics: {
      total_features: number;
      interpretable_percentage: number;
      avg_activation_frequency: number;
    };
  }>;

  // Selected feature for detail view
  selectedFeature: FeatureDetail | null;

  // Max-activating examples for selected feature
  featureExamples: FeatureActivationExample[];

  // Token analysis for selected feature
  featureTokenAnalysis: TokenAnalysisResponse | null;

  // Search filters
  searchFilters: Record<string, FeatureSearchRequest>;

  // Loading states
  isLoadingExtraction: boolean;
  isLoadingExtractions: boolean;
  isLoadingFeatures: boolean;
  isLoadingFeatureDetail: boolean;
  isLoadingExamples: boolean;
  isLoadingTokenAnalysis: boolean;

  // Error states
  extractionError: string | null;
  extractionsError: string | null;
  featuresError: string | null;
  featureDetailError: string | null;

  // Actions
  deleteExtraction: (extractionId: string) => Promise<void>;
  fetchAllExtractions: (statusFilter?: string[], limit?: number, offset?: number) => Promise<void>;
  fetchFeatures: (trainingId: string, filters?: FeatureSearchRequest) => Promise<void>;
  fetchExtractionFeatures: (extractionId: string, filters?: FeatureSearchRequest) => Promise<void>;
  fetchFeatureDetail: (featureId: string) => Promise<void>;
  fetchFeatureExamples: (featureId: string, limit?: number) => Promise<void>;
  fetchFeatureTokenAnalysis: (featureId: string, filters?: {
    applyFilters?: boolean;
    filterSpecial?: boolean;
    filterSingleChar?: boolean;
    filterPunctuation?: boolean;
    filterNumbers?: boolean;
    filterFragments?: boolean;
    filterStopWords?: boolean;
  }) => Promise<void>;
  updateFeature: (featureId: string, updates: FeatureUpdateRequest) => Promise<void>;
  patchFeatureLocally: (featureId: string, fields: Partial<Pick<Feature, 'name' | 'category' | 'description' | 'notes' | 'label_source'>>) => void;
  toggleFavorite: (featureId: string, isFavorite: boolean) => Promise<void>;
  setStarColor: (featureId: string, starColor: 'yellow' | 'purple' | 'aqua' | null) => Promise<void>;
  setSearchFilters: (trainingId: string, filters: FeatureSearchRequest) => void;
  clearSelectedFeature: () => void;

  // WebSocket update handlers for allExtractions list (by extraction ID)
  updateExtractionById: (extractionId: string, updates: Partial<ExtractionStatusResponse>) => void;
}

// Module-level refs for the in-flight cleanup request.
// Stored outside the store so clearSelectedFeature() can cancel a previous
// call when the user switches features rapidly, preventing request pile-up.
let _cleanupAbortController: AbortController | null = null;
let _cleanupTimeoutId: ReturnType<typeof setTimeout> | null = null;

/**
 * Features store implementation.
 */
/**
 * Fields that exist only on the WebSocket progress payload, never on the REST
 * extractions response. A refetch must not blank them out mid-run.
 */
const TRANSIENT_EXTRACTION_FIELDS = [
  'status_message',
  'eta_seconds',
  'samples_per_second',
  'samples_processed',
  'total_samples',
  'current_batch',
  'total_batches',
  'features_in_heap',
  'heap_examples_count',
] as const;

export const useFeaturesStore = create<FeaturesStoreState>((set, get) => ({
  // Initial state
  extractionStatus: {},
  allExtractions: [],
  extractionsMetadata: null,
  featuresByTraining: {},
  featuresByExtraction: {},
  featureListMetadata: {},
  selectedFeature: null,
  featureExamples: [],
  featureTokenAnalysis: null,
  searchFilters: {},
  isLoadingExtraction: false,
  isLoadingExtractions: false,
  isLoadingFeatures: false,
  isLoadingFeatureDetail: false,
  isLoadingExamples: false,
  isLoadingTokenAnalysis: false,
  extractionError: null,
  extractionsError: null,
  featuresError: null,
  featureDetailError: null,

  /**
   * Delete an extraction job.
   */
  deleteExtraction: async (extractionId: string) => {
    set({ isLoadingExtraction: true, extractionError: null });

    try {
      await axios.delete(`/api/v1/extractions/${extractionId}`);

      // Remove from allExtractions list
      set((state) => ({
        allExtractions: state.allExtractions.filter(e => e.id !== extractionId),
        isLoadingExtraction: false,
      }));
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to delete extraction');
      set({ extractionError: errorMessage, isLoadingExtraction: false });
      throw error;
    }
  },

  /**
   * Fetch all extraction jobs with optional filtering.
   */
  fetchAllExtractions: async (statusFilter?: string[], limit: number = 50, offset: number = 0) => {
    // Only claim "loading" when there is nothing on screen yet.
    //
    // ExtractionsPanel renders the grid behind `!isLoadingExtractions`, so
    // flipping this flag UNMOUNTS every ExtractionJobCard and remounts it when
    // the response lands. Card-local state goes with it — which is why an
    // expanded "Live Metrics" panel snapped shut on every refresh.
    //
    // That was survivable while refreshes were user-initiated. It stopped being
    // survivable once this function ran on a timer (the 15s reconciliation
    // poll) and on every WebSocket completion event: reported 2026-07-26, "the
    // whole page refreshes ... and closes the expanded progress windows".
    //
    // Fixed here rather than in the poll because all four callers cause it.
    // A refresh of a list already on screen must never blank the list.
    const isFirstLoad = get().allExtractions.length === 0;
    set({ isLoadingExtractions: isFirstLoad, extractionsError: null });

    try {
      const params: Record<string, any> = {
        limit,
        offset,
      };

      if (statusFilter && statusFilter.length > 0) {
        params.status_filter = statusFilter.join(',');
      }

      const response = await axios.get<{ data: ExtractionStatusResponse[]; meta: { total: number; limit: number; offset: number } }>(
        `/api/v1/extractions`,
        { params }
      );

      // Carry forward the live-metrics fields the REST payload does not
      // contain. They arrive only over WebSocket, so a wholesale replace wipes
      // them — and with the 15s reconciliation poll in ExtractionsPanel that
      // now happens on a timer, making the ETA and status line flicker.
      //
      // Server state always wins for anything the server actually returns;
      // this only restores fields the response omits entirely, and only for a
      // job that is still running (a terminal job must not keep showing a
      // stale ETA).
      const previous = new Map(get().allExtractions.map((e) => [e.id, e]));
      const merged = response.data.data.map((incoming) => {
        const prev = previous.get(incoming.id);
        if (!prev) return incoming;

        const stillRunning =
          incoming.status === 'queued' ||
          incoming.status === 'extracting' ||
          incoming.status === 'finalizing';
        if (!stillRunning) return incoming;

        const incomingRec = incoming as unknown as Record<string, unknown>;
        const prevRec = prev as unknown as Record<string, unknown>;
        const carried: Record<string, unknown> = {};
        for (const key of TRANSIENT_EXTRACTION_FIELDS) {
          if (incomingRec[key] === undefined && prevRec[key] !== undefined) {
            carried[key] = prevRec[key];
          }
        }
        return { ...incoming, ...carried };
      });

      set({
        allExtractions: merged,
        extractionsMetadata: response.data.meta,
        isLoadingExtractions: false,
      });
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to fetch extractions');
      set({ extractionsError: errorMessage, isLoadingExtractions: false });
    }
  },

  /**
   * Fetch features for a training with optional filters.
   */
  fetchFeatures: async (trainingId: string, filters?: FeatureSearchRequest) => {
    set({ isLoadingFeatures: true, featuresError: null });

    try {
      // Use provided filters or stored filters
      const searchFilters = filters || get().searchFilters[trainingId] || {};

      const response = await axios.get<FeatureListResponse>(
        `/api/v1/trainings/${trainingId}/features`,
        { params: searchFilters }
      );

      set((state) => ({
        featuresByTraining: {
          ...state.featuresByTraining,
          [trainingId]: response.data.features,
        },
        featureListMetadata: {
          ...state.featureListMetadata,
          [trainingId]: {
            total: response.data.total,
            limit: response.data.limit,
            offset: response.data.offset,
            statistics: response.data.statistics,
          },
        },
        searchFilters: {
          ...state.searchFilters,
          [trainingId]: searchFilters,
        },
        isLoadingFeatures: false,
      }));
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to fetch features');
      set({ featuresError: errorMessage, isLoadingFeatures: false });
      throw error;
    }
  },

  /**
   * Fetch features for a specific extraction with optional filters.
   */
  fetchExtractionFeatures: async (extractionId: string, filters?: FeatureSearchRequest) => {
    set({ isLoadingFeatures: true, featuresError: null });

    try {
      // Use provided filters or stored filters
      const searchFilters = filters || get().searchFilters[extractionId] || {};

      const response = await axios.get<FeatureListResponse>(
        `/api/v1/extractions/${extractionId}/features`,
        { params: searchFilters }
      );

      set((state) => ({
        featuresByExtraction: {
          ...state.featuresByExtraction,
          [extractionId]: response.data.features,
        },
        featureListMetadata: {
          ...state.featureListMetadata,
          [extractionId]: {
            total: response.data.total,
            limit: response.data.limit,
            offset: response.data.offset,
            statistics: response.data.statistics,
          },
        },
        searchFilters: {
          ...state.searchFilters,
          [extractionId]: searchFilters,
        },
        isLoadingFeatures: false,
      }));
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to fetch extraction features');
      set({ featuresError: errorMessage, isLoadingFeatures: false });
      throw error;
    }
  },

  /**
   * Fetch detailed information for a feature.
   */
  fetchFeatureDetail: async (featureId: string) => {
    set({ isLoadingFeatureDetail: true, featureDetailError: null });

    try {
      const response = await axios.get<FeatureDetail>(`/api/v1/features/${featureId}`);

      set({
        selectedFeature: response.data,
        isLoadingFeatureDetail: false,
      });
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to fetch feature detail');
      set({ featureDetailError: errorMessage, isLoadingFeatureDetail: false });
      throw error;
    }
  },

  /**
   * Fetch max-activating examples for a feature.
   */
  fetchFeatureExamples: async (featureId: string, limit: number = 100) => {
    // Clear stale examples immediately so the UI doesn't show data from the
    // previous feature while the new request is in-flight.
    set({ isLoadingExamples: true, featureExamples: [] });

    try {
      const response = await axios.get<FeatureActivationExample[]>(
        `/api/v1/features/${featureId}/examples`,
        { params: { limit } }
      );

      set({
        featureExamples: response.data,
        isLoadingExamples: false,
      });
    } catch (error: any) {
      set({ isLoadingExamples: false });
      throw error;
    }
  },

  /**
   * Fetch token analysis for a feature.
   */
  fetchFeatureTokenAnalysis: async (
    featureId: string,
    filters?: {
      applyFilters?: boolean;
      filterSpecial?: boolean;
      filterSingleChar?: boolean;
      filterPunctuation?: boolean;
      filterNumbers?: boolean;
      filterFragments?: boolean;
      filterStopWords?: boolean;
    }
  ) => {
    // Clear stale analysis immediately so the UI doesn't show data from the
    // previous feature while the new request is in-flight.
    set({ isLoadingTokenAnalysis: true, featureTokenAnalysis: null });

    try {
      const params = {
        apply_filters: filters?.applyFilters ?? true,
        filter_special: filters?.filterSpecial ?? true,
        filter_single_char: filters?.filterSingleChar ?? true,
        filter_punctuation: filters?.filterPunctuation ?? true,
        filter_numbers: filters?.filterNumbers ?? true,
        filter_fragments: filters?.filterFragments ?? true,
        filter_stop_words: filters?.filterStopWords ?? false,
      };

      const response = await axios.get<TokenAnalysisResponse>(
        `/api/v1/features/${featureId}/token-analysis`,
        { params }
      );

      set({
        featureTokenAnalysis: response.data,
        isLoadingTokenAnalysis: false,
      });
    } catch (error: any) {
      set({ isLoadingTokenAnalysis: false });
      throw error;
    }
  },

  /**
   * Update feature metadata (name, description, notes).
   */
  updateFeature: async (featureId: string, updates: FeatureUpdateRequest) => {
    const response = await axios.patch<Feature>(`/api/v1/features/${featureId}`, updates);

    // Update feature in list if it exists
    set((state) => {
      const updatedFeaturesByTraining = { ...state.featuresByTraining };

      // Find and update feature in all training lists
      Object.keys(updatedFeaturesByTraining).forEach((trainingId) => {
        const features = updatedFeaturesByTraining[trainingId];
        const featureIndex = features.findIndex((f) => f.id === featureId);

        if (featureIndex !== -1) {
          updatedFeaturesByTraining[trainingId] = [
            ...features.slice(0, featureIndex),
            response.data,
            ...features.slice(featureIndex + 1),
          ];
        }
      });

      return {
        featuresByTraining: updatedFeaturesByTraining,
        selectedFeature: state.selectedFeature?.id === featureId
          ? { ...state.selectedFeature, ...response.data }
          : state.selectedFeature,
      };
    });
  },

  /**
   * Patch label fields on a feature in all store slices without a network call.
   * Called immediately on enhanced labeling completion so list rows and the
   * open modal reflect the new values before the user does anything.
   */
  patchFeatureLocally: (featureId, fields) => {
    const applyUpdate = (f: Feature) => f.id === featureId ? { ...f, ...fields } : f;
    set((state) => ({
      featuresByTraining: Object.fromEntries(
        Object.entries(state.featuresByTraining).map(([k, v]) => [k, v.map(applyUpdate)])
      ),
      featuresByExtraction: Object.fromEntries(
        Object.entries(state.featuresByExtraction).map(([k, v]) => [k, v.map(applyUpdate)])
      ),
      selectedFeature: state.selectedFeature?.id === featureId
        ? { ...state.selectedFeature, ...fields }
        : state.selectedFeature,
    }));
  },

  /**
   * Toggle favorite status for a feature.
   */
  toggleFavorite: async (featureId: string, isFavorite: boolean) => {
    await axios.post<{ is_favorite: boolean }>(
      `/api/v1/features/${featureId}/favorite`,
      null,
      { params: { is_favorite: isFavorite } }
    );

    // Update feature in all lists where it exists
    set((state) => {
      const updatedFeaturesByTraining = { ...state.featuresByTraining };
      const updatedFeaturesByExtraction = { ...state.featuresByExtraction };

      // Find and update feature in all training lists
      Object.keys(updatedFeaturesByTraining).forEach((trainingId) => {
        const features = updatedFeaturesByTraining[trainingId];
        const featureIndex = features.findIndex((f) => f.id === featureId);

        if (featureIndex !== -1) {
          updatedFeaturesByTraining[trainingId] = [
            ...features.slice(0, featureIndex),
            { ...features[featureIndex], is_favorite: isFavorite },
            ...features.slice(featureIndex + 1),
          ];
        }
      });

      // Find and update feature in all extraction lists
      Object.keys(updatedFeaturesByExtraction).forEach((extractionId) => {
        const features = updatedFeaturesByExtraction[extractionId];
        const featureIndex = features.findIndex((f) => f.id === featureId);

        if (featureIndex !== -1) {
          updatedFeaturesByExtraction[extractionId] = [
            ...features.slice(0, featureIndex),
            { ...features[featureIndex], is_favorite: isFavorite },
            ...features.slice(featureIndex + 1),
          ];
        }
      });

      return {
        featuresByTraining: updatedFeaturesByTraining,
        featuresByExtraction: updatedFeaturesByExtraction,
        selectedFeature: state.selectedFeature?.id === featureId
          ? { ...state.selectedFeature, is_favorite: isFavorite }
          : state.selectedFeature,
      };
    });
  },

  /**
   * Set star color for a feature.
   * 'yellow' = manually starred, 'purple' = enhanced labeling in flight,
   * 'aqua' = enhanced labeling completed (never downgraded to yellow).
   * null = unstar.
   */
  setStarColor: async (featureId: string, starColor: 'yellow' | 'purple' | 'aqua' | null) => {
    const response = await axios.post<{ is_favorite: boolean; star_color: string | null }>(
      `/api/v1/features/${featureId}/star`,
      null,
      { params: { star_color: starColor } }
    );
    const { is_favorite } = response.data;
    const star_color = (response.data.star_color ?? null) as 'yellow' | 'purple' | 'aqua' | null;

    const applyUpdate = (feature: any) =>
      feature.id === featureId ? { ...feature, is_favorite, star_color } : feature;

    set((state) => {
      const updatedByTraining = Object.fromEntries(
        Object.entries(state.featuresByTraining).map(([k, v]) => [k, v.map(applyUpdate)])
      );
      const updatedByExtraction = Object.fromEntries(
        Object.entries(state.featuresByExtraction).map(([k, v]) => [k, v.map(applyUpdate)])
      );
      return {
        featuresByTraining: updatedByTraining,
        featuresByExtraction: updatedByExtraction,
        selectedFeature: state.selectedFeature?.id === featureId
          ? { ...state.selectedFeature, is_favorite, star_color }
          : state.selectedFeature,
      };
    });
  },

  /**
   * Set search filters for a training.
   */
  setSearchFilters: (trainingId: string, filters: FeatureSearchRequest) => {
    set((state) => ({
      searchFilters: {
        ...state.searchFilters,
        [trainingId]: filters,
      },
    }));
  },

  /**
   * Clear selected feature and clean up GPU memory from analysis.
   * Calls the backend cleanup endpoint to free any GPU memory
   * allocated by logit lens or other analysis operations.
   */
  clearSelectedFeature: () => {
    set({ selectedFeature: null, featureExamples: [], featureTokenAnalysis: null });

    // Cancel any previous in-flight cleanup (user switched features rapidly).
    if (_cleanupAbortController) {
      _cleanupAbortController.abort();
      _cleanupAbortController = null;
    }
    if (_cleanupTimeoutId !== null) {
      clearTimeout(_cleanupTimeoutId);
      _cleanupTimeoutId = null;
    }

    const controller = new AbortController();
    _cleanupAbortController = controller;

    // ONLY CLEAR THE REFS IF WE STILL OWN THEM (MIS-E2E-121).
    //
    // These handlers used to null the shared refs unconditionally. An OLDER
    // request completing after a NEWER one started therefore cleared the
    // newer one's controller and timeout — so from then on the controller was
    // `null` for the rest of the session, no request could be cancelled, and
    // the 5-second hard timeout never fired again.
    //
    // Two rapid feature switches are enough, and clicking through features
    // quickly is the primary interaction of the Feature Browser. Permanent,
    // silent, and caused by ordinary use.
    const releaseIfCurrent = () => {
      if (_cleanupAbortController === controller) {
        _cleanupAbortController = null;
      }
      if (_cleanupTimeoutId === timeoutId) {
        clearTimeout(timeoutId);
        _cleanupTimeoutId = null;
      }
    };

    // Hard 5-second timeout — abort if the backend takes too long.
    const timeoutId = setTimeout(() => {
      controller.abort();
      releaseIfCurrent();
    }, 5000);
    _cleanupTimeoutId = timeoutId;

    // Fire-and-forget — result doesn't block the UI.
    axios.post('/api/v1/analysis/cleanup', undefined, { signal: controller.signal })
      .then((response) => {
        releaseIfCurrent();
        if (response.data.vram_freed_gb > 0) {
          console.log(`[Analysis Cleanup] Freed ${response.data.vram_freed_gb} GB VRAM`);
        }
      })
      .catch((error: any) => {
        releaseIfCurrent();
        // AbortError / CanceledError means a newer clearSelectedFeature() call
        // superseded this one — not a real failure, no need to log.
        if (
          error?.name === 'AbortError' ||
          error?.name === 'CanceledError' ||
          error?.code === 'ERR_CANCELED'
        ) return;
        // Log but don't surface — cleanup is best-effort.
        console.warn('[Analysis Cleanup] Failed:', error.message);
      });
  },

  /**
   * Update an extraction in the allExtractions list by extraction ID.
   * Used by WebSocket handlers to update extraction progress in real-time.
   */
  updateExtractionById: (extractionId: string, updates: Partial<ExtractionStatusResponse>) => {
    set((state) => {
      const extractionIndex = state.allExtractions.findIndex(e => e.id === extractionId);
      if (extractionIndex === -1) {
        // Extraction not in list - might need to refresh
        return state;
      }

      const updatedExtractions = [...state.allExtractions];
      updatedExtractions[extractionIndex] = {
        ...updatedExtractions[extractionIndex],
        ...updates,
      };

      // Also update extractionStatus if we have a training_id
      const extraction = updatedExtractions[extractionIndex];
      const newState: any = { allExtractions: updatedExtractions };

      if (extraction.training_id && state.extractionStatus[extraction.training_id]) {
        newState.extractionStatus = {
          ...state.extractionStatus,
          [extraction.training_id]: {
            ...state.extractionStatus[extraction.training_id],
            ...updates,
          },
        };
      }

      return newState;
    });
  },
}));

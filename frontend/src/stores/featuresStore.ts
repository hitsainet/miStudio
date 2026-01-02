/**
 * Features Store
 *
 * Zustand store for Feature Discovery state management.
 * Manages feature extraction, search, filtering, and real-time updates via WebSocket.
 *
 * Backend API Contract:
 * - POST /api/v1/trainings/:id/extract-features - Start feature extraction
 * - POST /api/v1/trainings/:id/cancel-extraction - Cancel active extraction
 * - DELETE /api/v1/extractions/:id - Delete extraction job
 * - GET /api/v1/trainings/:id/extraction-status - Get extraction status
 * - GET /api/v1/trainings/:id/features - List/search features
 * - GET /api/v1/features/:id - Get feature details
 * - PATCH /api/v1/features/:id - Update feature metadata
 * - POST /api/v1/features/:id/favorite - Toggle favorite status
 * - GET /api/v1/features/:id/examples - Get max-activating examples
 *
 * WebSocket Events:
 * - training:{training_id}/extraction:progress - Extraction progress updates
 * - training:{training_id}/extraction:completed - Extraction completion
 * - training:{training_id}/extraction:failed - Extraction failure
 */

import { create } from 'zustand';
import axios from 'axios';
import type {
  ExtractionConfigRequest,
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
  startExtraction: (trainingId: string, config: ExtractionConfigRequest) => Promise<void>;
  cancelExtraction: (trainingId: string) => Promise<void>;
  deleteExtraction: (extractionId: string, trainingId: string) => Promise<void>;
  getExtractionStatus: (trainingId: string) => Promise<void>;
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
  toggleFavorite: (featureId: string, isFavorite: boolean) => Promise<void>;
  setSearchFilters: (trainingId: string, filters: FeatureSearchRequest) => void;
  clearSelectedFeature: () => void;

  // WebSocket update handlers
  handleExtractionProgress: (trainingId: string, progress: number, featuresExtracted: number, totalFeatures: number) => void;
  handleExtractionCompleted: (trainingId: string) => void;
  handleExtractionFailed: (trainingId: string, error: string) => void;

  // WebSocket update handlers for allExtractions list (by extraction ID)
  updateExtractionById: (extractionId: string, updates: Partial<ExtractionStatusResponse>) => void;
}

/**
 * Features store implementation.
 */
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
   * Start feature extraction for a training.
   */
  startExtraction: async (trainingId: string, config: ExtractionConfigRequest) => {
    set({ isLoadingExtraction: true, extractionError: null });

    try {
      const response = await axios.post<ExtractionStatusResponse>(
        `/api/v1/trainings/${trainingId}/extract-features`,
        config
      );

      set((state) => ({
        extractionStatus: {
          ...state.extractionStatus,
          [trainingId]: response.data,
        },
        isLoadingExtraction: false,
      }));
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to start extraction');
      set({ extractionError: errorMessage, isLoadingExtraction: false });
      throw error;
    }
  },

  /**
   * Cancel an active extraction.
   */
  cancelExtraction: async (trainingId: string) => {
    set({ isLoadingExtraction: true, extractionError: null });

    try {
      await axios.post(`/api/v1/trainings/${trainingId}/cancel-extraction`);

      // Refresh status to get updated state
      await get().getExtractionStatus(trainingId);
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to cancel extraction');
      set({ extractionError: errorMessage, isLoadingExtraction: false });
      throw error;
    }
  },

  /**
   * Delete an extraction job.
   */
  deleteExtraction: async (extractionId: string, trainingId: string) => {
    set({ isLoadingExtraction: true, extractionError: null });

    try {
      await axios.delete(`/api/v1/extractions/${extractionId}`);

      // Clear extraction status for this training
      set((state) => ({
        extractionStatus: {
          ...state.extractionStatus,
          [trainingId]: null,
        },
        isLoadingExtraction: false,
      }));
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to delete extraction');
      set({ extractionError: errorMessage, isLoadingExtraction: false });
      throw error;
    }
  },

  /**
   * Get extraction status for a training.
   */
  getExtractionStatus: async (trainingId: string) => {
    set({ isLoadingExtraction: true, extractionError: null });

    try {
      const response = await axios.get<ExtractionStatusResponse | null>(
        `/api/v1/trainings/${trainingId}/extraction-status`
      );

      // Backend returns null when no extraction exists (instead of 404)
      set((state) => ({
        extractionStatus: {
          ...state.extractionStatus,
          [trainingId]: response.data,
        },
        isLoadingExtraction: false,
      }));
    } catch (error: any) {
      const errorMessage = extractErrorMessage(error, 'Failed to get extraction status');
      set({ extractionError: errorMessage, isLoadingExtraction: false });
    }
  },

  /**
   * Fetch all extraction jobs with optional filtering.
   */
  fetchAllExtractions: async (statusFilter?: string[], limit: number = 50, offset: number = 0) => {
    set({ isLoadingExtractions: true, extractionsError: null });

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

      set({
        allExtractions: response.data.data,
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
    set({ isLoadingExamples: true });

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
    set({ isLoadingTokenAnalysis: true });

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
    try {
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
    } catch (error: any) {
      throw error;
    }
  },

  /**
   * Toggle favorite status for a feature.
   */
  toggleFavorite: async (featureId: string, isFavorite: boolean) => {
    try {
      await axios.post<{ is_favorite: boolean }>(
        `/api/v1/features/${featureId}/favorite`,
        null,
        { params: { is_favorite: isFavorite } }
      );

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
              { ...features[featureIndex], is_favorite: isFavorite },
              ...features.slice(featureIndex + 1),
            ];
          }
        });

        return {
          featuresByTraining: updatedFeaturesByTraining,
          selectedFeature: state.selectedFeature?.id === featureId
            ? { ...state.selectedFeature, is_favorite: isFavorite }
            : state.selectedFeature,
        };
      });
    } catch (error: any) {
      throw error;
    }
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
    set({ selectedFeature: null, featureExamples: [] });

    // Fire-and-forget cleanup call - don't block UI
    axios.post('/api/v1/analysis/cleanup')
      .then((response) => {
        if (response.data.vram_freed_gb > 0) {
          console.log(`[Analysis Cleanup] Freed ${response.data.vram_freed_gb} GB VRAM`);
        }
      })
      .catch((error) => {
        // Log but don't fail - cleanup is best-effort
        console.warn('[Analysis Cleanup] Failed:', error.message);
      });
  },

  /**
   * Handle extraction progress WebSocket event.
   */
  handleExtractionProgress: (trainingId: string, progress: number, featuresExtracted: number, totalFeatures: number) => {
    set((state) => {
      const currentStatus = state.extractionStatus[trainingId];
      if (!currentStatus) return state;

      return {
        extractionStatus: {
          ...state.extractionStatus,
          [trainingId]: {
            ...currentStatus,
            progress,
            features_extracted: featuresExtracted,
            total_features: totalFeatures,
          },
        },
      };
    });
  },

  /**
   * Handle extraction completed WebSocket event.
   */
  handleExtractionCompleted: (trainingId: string) => {
    // Refresh extraction status to get final statistics
    get().getExtractionStatus(trainingId);
  },

  /**
   * Handle extraction failed WebSocket event.
   */
  handleExtractionFailed: (trainingId: string, error: string) => {
    set((state) => {
      const currentStatus = state.extractionStatus[trainingId];
      if (!currentStatus) return state;

      return {
        extractionStatus: {
          ...state.extractionStatus,
          [trainingId]: {
            ...currentStatus,
            status: 'failed',
            error_message: error,
          },
        },
      };
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

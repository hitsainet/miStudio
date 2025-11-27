/**
 * Zustand store for Steering operations.
 *
 * This store manages the global state for feature steering, including:
 * - Selected SAE for steering
 * - Selected features (up to 4) with colors
 * - Generation parameters
 * - Comparison requests and responses
 * - Experiment management (save/load)
 * - Real-time progress updates
 *
 * Connects to REAL backend API at /api/v1/steering
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';
import {
  SelectedFeature,
  GenerationParams,
  AdvancedGenerationParams,
  DEFAULT_GENERATION_PARAMS,
  SteeringComparisonRequest,
  SteeringComparisonResponse,
  SteeringStrengthSweepRequest,
  StrengthSweepResponse,
  SteeringExperiment,
  SteeringProgressUpdate,
  FeatureColor,
  FEATURE_COLOR_ORDER,
} from '../types/steering';
import { SAE } from '../types/sae';
import * as steeringApi from '../api/steering';

// Maximum number of features that can be selected
const MAX_SELECTED_FEATURES = 4;

// Callback for subscribing to steering progress (set by WebSocket context)
let subscribeToSteeringCallback: ((comparisonId: string) => void) | null = null;

export function setSteeringSubscriptionCallback(callback: (comparisonId: string) => void) {
  subscribeToSteeringCallback = callback;
}

interface SteeringState {
  // Selected SAE
  selectedSAE: SAE | null;

  // Selected features (up to 4)
  selectedFeatures: SelectedFeature[];

  // Prompt
  prompt: string;

  // Generation parameters
  generationParams: GenerationParams;
  advancedParams: AdvancedGenerationParams | null;
  showAdvancedParams: boolean;

  // Comparison state
  isGenerating: boolean;
  comparisonId: string | null;
  progress: number;
  progressMessage: string | null;
  currentComparison: SteeringComparisonResponse | null;

  // Strength sweep state
  isSweeping: boolean;
  sweepResults: StrengthSweepResponse | null;

  // Experiment management
  experiments: SteeringExperiment[];
  experimentsLoading: boolean;
  experimentsPagination: {
    skip: number;
    limit: number;
    total: number;
    hasMore: boolean;
  };

  // Error state
  error: string | null;

  // Actions - SAE Selection
  selectSAE: (sae: SAE | null) => void;

  // Actions - Feature Selection
  addFeature: (feature: Omit<SelectedFeature, 'color'>) => boolean;
  removeFeature: (featureIdx: number, layer: number) => void;
  updateFeatureStrength: (featureIdx: number, layer: number, strength: number) => void;
  applyStrengthPreset: (strength: number) => void;
  clearFeatures: () => void;
  reorderFeatures: (fromIndex: number, toIndex: number) => void;

  // Actions - Prompt
  setPrompt: (prompt: string) => void;

  // Actions - Generation Parameters
  setGenerationParams: (params: Partial<GenerationParams>) => void;
  setAdvancedParams: (params: Partial<AdvancedGenerationParams> | null) => void;
  toggleAdvancedParams: () => void;
  resetParams: () => void;

  // Actions - Comparison
  generateComparison: (includeUnsteered?: boolean, computeMetrics?: boolean) => Promise<SteeringComparisonResponse>;
  abortComparison: () => Promise<void>;
  clearComparison: () => void;

  // Actions - Strength Sweep
  runStrengthSweep: (featureIdx: number, layer: number, strengthValues: number[]) => Promise<StrengthSweepResponse>;
  clearSweepResults: () => void;

  // Actions - Progress Updates (WebSocket)
  updateProgress: (update: SteeringProgressUpdate) => void;

  // Actions - Experiments
  fetchExperiments: (params?: { skip?: number; limit?: number; search?: string; sae_id?: string }) => Promise<void>;
  saveExperiment: (name: string, description?: string, tags?: string[]) => Promise<SteeringExperiment>;
  loadExperiment: (experiment: SteeringExperiment) => void;
  deleteExperiment: (id: string) => Promise<void>;
  deleteExperimentsBatch: (ids: string[]) => Promise<void>;

  // Actions - Error Handling
  setError: (error: string | null) => void;
  clearError: () => void;

  // Actions - Cache Management
  clearModelCache: () => Promise<void>;
  isUnloadingCache: boolean;
}

export const useSteeringStore = create<SteeringState>()(
  devtools(
    (set, get) => ({
      // Initial state
      selectedSAE: null,
      selectedFeatures: [],
      prompt: '',
      generationParams: { ...DEFAULT_GENERATION_PARAMS },
      advancedParams: null,
      showAdvancedParams: false,
      isGenerating: false,
      comparisonId: null,
      progress: 0,
      progressMessage: null,
      currentComparison: null,
      isSweeping: false,
      sweepResults: null,
      experiments: [],
      experimentsLoading: false,
      experimentsPagination: {
        skip: 0,
        limit: 20,
        total: 0,
        hasMore: false,
      },
      error: null,
      isUnloadingCache: false,

      // Select an SAE for steering
      selectSAE: (sae: SAE | null) => {
        set({
          selectedSAE: sae,
          selectedFeatures: [], // Clear features when SAE changes
          currentComparison: null,
          sweepResults: null,
        });
      },

      // Add a feature to selection (returns false if max reached or duplicate)
      addFeature: (feature: Omit<SelectedFeature, 'color'>) => {
        const { selectedFeatures } = get();

        // Check if max features reached
        if (selectedFeatures.length >= MAX_SELECTED_FEATURES) {
          return false;
        }

        // Check for duplicate
        const isDuplicate = selectedFeatures.some(
          (f) => f.feature_idx === feature.feature_idx && f.layer === feature.layer
        );
        if (isDuplicate) {
          return false;
        }

        // Find next available color
        const usedColors = selectedFeatures.map((f) => f.color);
        const nextColor = FEATURE_COLOR_ORDER.find((c) => !usedColors.includes(c)) || FEATURE_COLOR_ORDER[0];

        // Explicitly set all properties to ensure label is preserved
        const newFeature: SelectedFeature = {
          feature_idx: feature.feature_idx,
          layer: feature.layer,
          strength: feature.strength,
          label: feature.label,
          color: nextColor,
        };

        set({ selectedFeatures: [...selectedFeatures, newFeature] });
        return true;
      },

      // Remove a feature from selection
      removeFeature: (featureIdx: number, layer: number) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.filter(
            (f) => !(f.feature_idx === featureIdx && f.layer === layer)
          ),
        }));
      },

      // Update feature strength
      updateFeatureStrength: (featureIdx: number, layer: number, strength: number) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) =>
            f.feature_idx === featureIdx && f.layer === layer
              ? { ...f, strength }
              : f
          ),
        }));
      },

      // Apply strength preset to all selected features
      applyStrengthPreset: (strength: number) => {
        set((state) => ({
          selectedFeatures: state.selectedFeatures.map((f) => ({
            ...f,
            strength,
          })),
        }));
      },

      // Clear all selected features
      clearFeatures: () => {
        set({ selectedFeatures: [], currentComparison: null });
      },

      // Reorder features (drag and drop)
      reorderFeatures: (fromIndex: number, toIndex: number) => {
        set((state) => {
          const features = [...state.selectedFeatures];
          const [removed] = features.splice(fromIndex, 1);
          features.splice(toIndex, 0, removed);
          return { selectedFeatures: features };
        });
      },

      // Set prompt
      setPrompt: (prompt: string) => {
        set({ prompt });
      },

      // Set generation parameters
      setGenerationParams: (params: Partial<GenerationParams>) => {
        set((state) => ({
          generationParams: { ...state.generationParams, ...params },
        }));
      },

      // Set advanced parameters
      setAdvancedParams: (params: Partial<AdvancedGenerationParams> | null) => {
        if (params === null) {
          set({ advancedParams: null });
        } else {
          set((state) => ({
            advancedParams: state.advancedParams
              ? { ...state.advancedParams, ...params }
              : {
                  repetition_penalty: 1.15,
                  presence_penalty: 0.0,
                  frequency_penalty: 0.0,
                  do_sample: true,
                  stop_sequences: [],
                  ...params,
                },
          }));
        }
      },

      // Toggle advanced parameters visibility
      toggleAdvancedParams: () => {
        set((state) => ({ showAdvancedParams: !state.showAdvancedParams }));
      },

      // Reset parameters to defaults
      resetParams: () => {
        set({
          generationParams: { ...DEFAULT_GENERATION_PARAMS },
          advancedParams: null,
        });
      },

      // Generate comparison
      generateComparison: async (includeUnsteered = true, computeMetrics = false) => {
        const { selectedSAE, selectedFeatures, prompt, generationParams, advancedParams } = get();

        if (!selectedSAE) {
          throw new Error('No SAE selected');
        }
        if (selectedFeatures.length === 0) {
          throw new Error('No features selected');
        }
        if (!prompt.trim()) {
          throw new Error('Prompt is required');
        }

        set({
          isGenerating: true,
          progress: 0,
          progressMessage: 'Starting comparison...',
          error: null,
        });

        try {
          const request: SteeringComparisonRequest = {
            sae_id: selectedSAE.id,
            prompt,
            selected_features: selectedFeatures,
            generation_params: generationParams,
            include_unsteered: includeUnsteered,
            compute_metrics: computeMetrics,
          };

          if (advancedParams) {
            request.advanced_params = advancedParams;
          }

          const response = await steeringApi.generateComparison(request);

          // Subscribe to progress updates
          if (subscribeToSteeringCallback && response.comparison_id) {
            console.log('[SteeringStore] Subscribing to comparison progress:', response.comparison_id);
            subscribeToSteeringCallback(response.comparison_id);
          }

          set({
            comparisonId: response.comparison_id,
            currentComparison: response,
            isGenerating: false,
            progress: 100,
            progressMessage: 'Comparison complete',
          });

          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to generate comparison';
          set({
            error: errorMessage,
            isGenerating: false,
            progress: 0,
            progressMessage: null,
          });
          throw error;
        }
      },

      // Abort an in-progress comparison
      abortComparison: async () => {
        const { comparisonId } = get();
        if (!comparisonId) return;

        try {
          await steeringApi.abortComparison(comparisonId);
          set({
            isGenerating: false,
            progress: 0,
            progressMessage: 'Comparison aborted',
            comparisonId: null,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to abort comparison';
          set({ error: errorMessage });
        }
      },

      // Clear comparison results
      clearComparison: () => {
        set({
          currentComparison: null,
          comparisonId: null,
          progress: 0,
          progressMessage: null,
        });
      },

      // Run strength sweep
      runStrengthSweep: async (featureIdx: number, layer: number, strengthValues: number[]) => {
        const { selectedSAE, prompt, generationParams } = get();

        if (!selectedSAE) {
          throw new Error('No SAE selected');
        }
        if (!prompt.trim()) {
          throw new Error('Prompt is required');
        }

        set({
          isSweeping: true,
          sweepResults: null,
          error: null,
        });

        try {
          const request: SteeringStrengthSweepRequest = {
            sae_id: selectedSAE.id,
            prompt,
            feature_idx: featureIdx,
            layer,
            strength_values: strengthValues,
            generation_params: generationParams,
          };

          const response = await steeringApi.runStrengthSweep(request);

          set({
            sweepResults: response,
            isSweeping: false,
          });

          return response;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to run strength sweep';
          set({
            error: errorMessage,
            isSweeping: false,
          });
          throw error;
        }
      },

      // Clear sweep results
      clearSweepResults: () => {
        set({ sweepResults: null });
      },

      // Update progress (called by WebSocket)
      updateProgress: (update: SteeringProgressUpdate) => {
        const { comparisonId } = get();
        if (update.comparison_id !== comparisonId) return;

        set({
          progress: update.progress,
          progressMessage: update.message,
        });

        // If completed or failed, update state
        if (update.status === 'completed' || update.status === 'failed') {
          set({ isGenerating: false });
        }
      },

      // Fetch experiments
      fetchExperiments: async (params?: { skip?: number; limit?: number; search?: string; sae_id?: string }) => {
        set({ experimentsLoading: true, error: null });
        try {
          const response = await steeringApi.getExperiments(params);
          set({
            experiments: response.data,
            experimentsPagination: {
              skip: response.pagination.skip,
              limit: response.pagination.limit,
              total: response.pagination.total,
              hasMore: response.pagination.has_more,
            },
            experimentsLoading: false,
          });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to fetch experiments';
          set({ error: errorMessage, experimentsLoading: false });
        }
      },

      // Save current comparison as experiment
      saveExperiment: async (name: string, description?: string, tags?: string[]) => {
        const { comparisonId } = get();
        if (!comparisonId) {
          throw new Error('No comparison to save');
        }

        try {
          const experiment = await steeringApi.saveExperiment({
            name,
            description,
            comparison_id: comparisonId,
            tags,
          });

          set((state) => ({
            experiments: [experiment, ...state.experiments],
            experimentsPagination: {
              ...state.experimentsPagination,
              total: state.experimentsPagination.total + 1,
            },
          }));

          return experiment;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to save experiment';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Load an experiment
      loadExperiment: (experiment: SteeringExperiment) => {
        set({
          selectedFeatures: experiment.selected_features,
          prompt: experiment.prompt,
          generationParams: experiment.generation_params,
          currentComparison: experiment.results,
          comparisonId: experiment.results.comparison_id,
        });
      },

      // Delete an experiment
      deleteExperiment: async (id: string) => {
        try {
          await steeringApi.deleteExperiment(id);
          set((state) => ({
            experiments: state.experiments.filter((e) => e.id !== id),
            experimentsPagination: {
              ...state.experimentsPagination,
              total: state.experimentsPagination.total - 1,
            },
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete experiment';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Delete multiple experiments
      deleteExperimentsBatch: async (ids: string[]) => {
        try {
          await steeringApi.deleteExperimentsBatch(ids);
          set((state) => ({
            experiments: state.experiments.filter((e) => !ids.includes(e.id)),
            experimentsPagination: {
              ...state.experimentsPagination,
              total: state.experimentsPagination.total - ids.length,
            },
          }));
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to delete experiments';
          set({ error: errorMessage });
          throw error;
        }
      },

      // Set error
      setError: (error: string | null) => {
        set({ error });
      },

      // Clear error
      clearError: () => {
        set({ error: null });
      },

      // Clear model cache
      clearModelCache: async () => {
        set({ isUnloadingCache: true, error: null });
        try {
          await steeringApi.clearSteeringCache();
          set({ isUnloadingCache: false });
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : 'Failed to clear cache';
          set({ error: errorMessage, isUnloadingCache: false });
          throw error;
        }
      },
    }),
    {
      name: 'SteeringStore',
    }
  )
);

// Selector for checking if ready to generate
export const selectCanGenerate = (state: SteeringState) =>
  state.selectedSAE !== null &&
  state.selectedFeatures.length > 0 &&
  state.prompt.trim().length > 0 &&
  !state.isGenerating;

// Selector for feature by index and layer
export const selectFeature = (featureIdx: number, layer: number) => (state: SteeringState) =>
  state.selectedFeatures.find((f) => f.feature_idx === featureIdx && f.layer === layer);

// Selector for available colors
export const selectAvailableColors = (state: SteeringState): FeatureColor[] => {
  const usedColors = state.selectedFeatures.map((f) => f.color);
  return FEATURE_COLOR_ORDER.filter((c) => !usedColors.includes(c));
};

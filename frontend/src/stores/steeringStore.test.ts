/**
 * Tests for Steering Store.
 *
 * Tests Zustand store functionality for feature steering including:
 * - Feature selection (up to 4 features)
 * - Color assignment
 * - Strength management
 * - Generation parameters
 * - Progress updates
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { act } from '@testing-library/react';
import {
  useSteeringStore,
  selectCanGenerate,
  selectFeature,
  selectAvailableColors,
} from './steeringStore';
import {
  SelectedFeature,
  FEATURE_COLOR_ORDER,
  DEFAULT_GENERATION_PARAMS,
} from '../types/steering';
import { SAE, SAESource, SAEStatus, SAEFormat } from '../types/sae';

// Mock the API module
vi.mock('../api/steering', () => ({
  generateComparison: vi.fn(),
  abortComparison: vi.fn(),
  runStrengthSweep: vi.fn(),
  getExperiments: vi.fn(),
  saveExperiment: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteExperimentsBatch: vi.fn(),
}));

// Helper to create mock SAE
function createMockSAE(overrides: Partial<SAE> = {}): SAE {
  return {
    id: 'sae-1',
    name: 'Test SAE',
    description: 'A test SAE',
    source: SAESource.HUGGINGFACE,
    status: SAEStatus.READY,
    hf_repo_id: 'test/repo',
    hf_filepath: 'model.safetensors',
    hf_revision: 'main',
    training_id: null,
    model_name: 'gpt2',
    model_id: null,
    layer: 6,
    n_features: 16384,
    d_model: 768,
    architecture: 'standard',
    format: SAEFormat.SAELENS,
    local_path: '/data/saes/sae-1',
    file_size_bytes: 100000000,
    progress: 100,
    error_message: null,
    sae_metadata: {},
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
    downloaded_at: '2024-01-01T00:00:00Z',
    ...overrides,
  };
}

// Helper to create mock feature
function createMockFeature(overrides: Partial<SelectedFeature> = {}): Omit<SelectedFeature, 'color'> {
  return {
    feature_idx: 100,
    layer: 6,
    strength: 50,
    label: 'Test Feature',
    ...overrides,
  };
}

describe('SteeringStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useSteeringStore.setState({
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
    });
  });

  describe('Initial state', () => {
    it('should have correct initial state', () => {
      const state = useSteeringStore.getState();

      expect(state.selectedSAE).toBeNull();
      expect(state.selectedFeatures).toEqual([]);
      expect(state.prompt).toBe('');
      expect(state.generationParams).toEqual(DEFAULT_GENERATION_PARAMS);
      expect(state.isGenerating).toBe(false);
    });
  });

  describe('selectSAE', () => {
    it('should select an SAE', () => {
      const sae = createMockSAE();

      act(() => {
        useSteeringStore.getState().selectSAE(sae);
      });

      expect(useSteeringStore.getState().selectedSAE).toEqual(sae);
    });

    it('should clear features when SAE changes', () => {
      const sae = createMockSAE();
      useSteeringStore.setState({
        selectedSAE: sae,
        selectedFeatures: [{ ...createMockFeature(), color: 'teal' }],
      });

      const newSAE = createMockSAE({ id: 'sae-2' });

      act(() => {
        useSteeringStore.getState().selectSAE(newSAE);
      });

      expect(useSteeringStore.getState().selectedFeatures).toEqual([]);
    });

    it('should clear comparison when SAE changes', () => {
      useSteeringStore.setState({
        currentComparison: { comparison_id: 'comp-1' } as any,
        sweepResults: { sweep_id: 'sweep-1' } as any,
      });

      act(() => {
        useSteeringStore.getState().selectSAE(createMockSAE());
      });

      expect(useSteeringStore.getState().currentComparison).toBeNull();
      expect(useSteeringStore.getState().sweepResults).toBeNull();
    });
  });

  describe('addFeature', () => {
    it('should add a feature with auto-assigned color', () => {
      const feature = createMockFeature();

      act(() => {
        const result = useSteeringStore.getState().addFeature(feature);
        expect(result).toBe(true);
      });

      const state = useSteeringStore.getState();
      expect(state.selectedFeatures).toHaveLength(1);
      expect(state.selectedFeatures[0].color).toBe(FEATURE_COLOR_ORDER[0]);
    });

    it('should assign sequential colors', () => {
      act(() => {
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 1 }));
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 2 }));
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 3 }));
      });

      const features = useSteeringStore.getState().selectedFeatures;
      expect(features[0].color).toBe(FEATURE_COLOR_ORDER[0]);
      expect(features[1].color).toBe(FEATURE_COLOR_ORDER[1]);
      expect(features[2].color).toBe(FEATURE_COLOR_ORDER[2]);
    });

    it('should reject duplicate features', () => {
      const feature = createMockFeature({ feature_idx: 100, layer: 6 });

      act(() => {
        useSteeringStore.getState().addFeature(feature);
        const result = useSteeringStore.getState().addFeature(feature);
        expect(result).toBe(false);
      });

      expect(useSteeringStore.getState().selectedFeatures).toHaveLength(1);
    });

    it('should reject when max features reached', () => {
      act(() => {
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 1 }));
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 2 }));
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 3 }));
        useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 4 }));
        const result = useSteeringStore.getState().addFeature(createMockFeature({ feature_idx: 5 }));
        expect(result).toBe(false);
      });

      expect(useSteeringStore.getState().selectedFeatures).toHaveLength(4);
    });

    it('should preserve feature label', () => {
      act(() => {
        useSteeringStore.getState().addFeature(createMockFeature({ label: 'My Feature Label' }));
      });

      expect(useSteeringStore.getState().selectedFeatures[0].label).toBe('My Feature Label');
    });
  });

  describe('removeFeature', () => {
    it('should remove a feature by idx and layer', () => {
      useSteeringStore.setState({
        selectedFeatures: [
          { ...createMockFeature({ feature_idx: 1, layer: 6 }), color: 'teal' },
          { ...createMockFeature({ feature_idx: 2, layer: 6 }), color: 'blue' },
        ],
      });

      act(() => {
        useSteeringStore.getState().removeFeature(1, 6);
      });

      const features = useSteeringStore.getState().selectedFeatures;
      expect(features).toHaveLength(1);
      expect(features[0].feature_idx).toBe(2);
    });
  });

  describe('updateFeatureStrength', () => {
    it('should update feature strength', () => {
      useSteeringStore.setState({
        selectedFeatures: [
          { ...createMockFeature({ feature_idx: 1, strength: 50 }), color: 'teal' },
        ],
      });

      act(() => {
        useSteeringStore.getState().updateFeatureStrength(1, 6, 100);
      });

      expect(useSteeringStore.getState().selectedFeatures[0].strength).toBe(100);
    });
  });

  describe('applyStrengthPreset', () => {
    it('should apply strength to all features', () => {
      useSteeringStore.setState({
        selectedFeatures: [
          { ...createMockFeature({ feature_idx: 1, strength: 10 }), color: 'teal' },
          { ...createMockFeature({ feature_idx: 2, strength: 20 }), color: 'blue' },
          { ...createMockFeature({ feature_idx: 3, strength: 30 }), color: 'purple' },
        ],
      });

      act(() => {
        useSteeringStore.getState().applyStrengthPreset(50);
      });

      const features = useSteeringStore.getState().selectedFeatures;
      expect(features.every((f) => f.strength === 50)).toBe(true);
    });
  });

  describe('clearFeatures', () => {
    it('should clear all features and comparison', () => {
      useSteeringStore.setState({
        selectedFeatures: [{ ...createMockFeature(), color: 'teal' }],
        currentComparison: { comparison_id: 'comp-1' } as any,
      });

      act(() => {
        useSteeringStore.getState().clearFeatures();
      });

      expect(useSteeringStore.getState().selectedFeatures).toEqual([]);
      expect(useSteeringStore.getState().currentComparison).toBeNull();
    });
  });

  describe('reorderFeatures', () => {
    it('should reorder features', () => {
      useSteeringStore.setState({
        selectedFeatures: [
          { ...createMockFeature({ feature_idx: 1 }), color: 'teal' },
          { ...createMockFeature({ feature_idx: 2 }), color: 'blue' },
          { ...createMockFeature({ feature_idx: 3 }), color: 'purple' },
        ],
      });

      act(() => {
        useSteeringStore.getState().reorderFeatures(0, 2);
      });

      const features = useSteeringStore.getState().selectedFeatures;
      expect(features[0].feature_idx).toBe(2);
      expect(features[1].feature_idx).toBe(3);
      expect(features[2].feature_idx).toBe(1);
    });
  });

  describe('setPrompt', () => {
    it('should set the prompt', () => {
      act(() => {
        useSteeringStore.getState().setPrompt('Once upon a time');
      });

      expect(useSteeringStore.getState().prompt).toBe('Once upon a time');
    });
  });

  describe('setGenerationParams', () => {
    it('should update generation parameters', () => {
      act(() => {
        useSteeringStore.getState().setGenerationParams({ temperature: 0.5, max_new_tokens: 200 });
      });

      const params = useSteeringStore.getState().generationParams;
      expect(params.temperature).toBe(0.5);
      expect(params.max_new_tokens).toBe(200);
    });

    it('should merge with existing params', () => {
      act(() => {
        useSteeringStore.getState().setGenerationParams({ temperature: 0.5 });
      });

      const params = useSteeringStore.getState().generationParams;
      expect(params.temperature).toBe(0.5);
      expect(params.top_p).toBe(DEFAULT_GENERATION_PARAMS.top_p);
    });
  });

  describe('setAdvancedParams', () => {
    it('should set advanced parameters', () => {
      act(() => {
        useSteeringStore.getState().setAdvancedParams({ repetition_penalty: 1.2 });
      });

      expect(useSteeringStore.getState().advancedParams?.repetition_penalty).toBe(1.2);
    });

    it('should clear advanced params when null', () => {
      useSteeringStore.setState({
        advancedParams: { repetition_penalty: 1.2 } as any,
      });

      act(() => {
        useSteeringStore.getState().setAdvancedParams(null);
      });

      expect(useSteeringStore.getState().advancedParams).toBeNull();
    });
  });

  describe('toggleAdvancedParams', () => {
    it('should toggle advanced params visibility', () => {
      expect(useSteeringStore.getState().showAdvancedParams).toBe(false);

      act(() => {
        useSteeringStore.getState().toggleAdvancedParams();
      });

      expect(useSteeringStore.getState().showAdvancedParams).toBe(true);

      act(() => {
        useSteeringStore.getState().toggleAdvancedParams();
      });

      expect(useSteeringStore.getState().showAdvancedParams).toBe(false);
    });
  });

  describe('resetParams', () => {
    it('should reset to default parameters', () => {
      useSteeringStore.setState({
        generationParams: { ...DEFAULT_GENERATION_PARAMS, temperature: 0.1 },
        advancedParams: { repetition_penalty: 2.0 } as any,
      });

      act(() => {
        useSteeringStore.getState().resetParams();
      });

      expect(useSteeringStore.getState().generationParams).toEqual(DEFAULT_GENERATION_PARAMS);
      expect(useSteeringStore.getState().advancedParams).toBeNull();
    });
  });

  describe('clearComparison', () => {
    it('should clear comparison state', () => {
      useSteeringStore.setState({
        currentComparison: { comparison_id: 'comp-1' } as any,
        comparisonId: 'comp-1',
        progress: 100,
        progressMessage: 'Done',
      });

      act(() => {
        useSteeringStore.getState().clearComparison();
      });

      const state = useSteeringStore.getState();
      expect(state.currentComparison).toBeNull();
      expect(state.comparisonId).toBeNull();
      expect(state.progress).toBe(0);
      expect(state.progressMessage).toBeNull();
    });
  });

  describe('clearSweepResults', () => {
    it('should clear sweep results', () => {
      useSteeringStore.setState({
        sweepResults: { sweep_id: 'sweep-1' } as any,
      });

      act(() => {
        useSteeringStore.getState().clearSweepResults();
      });

      expect(useSteeringStore.getState().sweepResults).toBeNull();
    });
  });

  describe('updateProgress', () => {
    it('should update progress for matching comparison', () => {
      useSteeringStore.setState({ comparisonId: 'comp-1', isGenerating: true });

      act(() => {
        useSteeringStore.getState().updateProgress({
          comparison_id: 'comp-1',
          status: 'running',
          current_config: null,
          progress: 50,
          message: 'Generating...',
        });
      });

      expect(useSteeringStore.getState().progress).toBe(50);
      expect(useSteeringStore.getState().progressMessage).toBe('Generating...');
    });

    it('should ignore progress for non-matching comparison', () => {
      useSteeringStore.setState({ comparisonId: 'comp-1', progress: 25 });

      act(() => {
        useSteeringStore.getState().updateProgress({
          comparison_id: 'comp-other',
          status: 'running',
          current_config: null,
          progress: 75,
          message: 'Different comparison',
        });
      });

      expect(useSteeringStore.getState().progress).toBe(25);
    });

    it('should set isGenerating to false when completed', () => {
      useSteeringStore.setState({ comparisonId: 'comp-1', isGenerating: true });

      act(() => {
        useSteeringStore.getState().updateProgress({
          comparison_id: 'comp-1',
          status: 'completed',
          current_config: null,
          progress: 100,
          message: 'Done',
        });
      });

      expect(useSteeringStore.getState().isGenerating).toBe(false);
    });
  });

  describe('setError / clearError', () => {
    it('should set and clear error', () => {
      act(() => {
        useSteeringStore.getState().setError('Something went wrong');
      });

      expect(useSteeringStore.getState().error).toBe('Something went wrong');

      act(() => {
        useSteeringStore.getState().clearError();
      });

      expect(useSteeringStore.getState().error).toBeNull();
    });
  });
});

describe('SteeringStore Selectors', () => {
  describe('selectCanGenerate', () => {
    it('should return true when all requirements met', () => {
      useSteeringStore.setState({
        selectedSAE: createMockSAE(),
        selectedFeatures: [{ ...createMockFeature(), color: 'teal' }],
        prompt: 'Test prompt',
        isGenerating: false,
      });

      expect(selectCanGenerate(useSteeringStore.getState())).toBe(true);
    });

    it('should return false without SAE', () => {
      useSteeringStore.setState({
        selectedSAE: null,
        selectedFeatures: [{ ...createMockFeature(), color: 'teal' }],
        prompt: 'Test prompt',
      });

      expect(selectCanGenerate(useSteeringStore.getState())).toBe(false);
    });

    it('should return false without features', () => {
      useSteeringStore.setState({
        selectedSAE: createMockSAE(),
        selectedFeatures: [],
        prompt: 'Test prompt',
      });

      expect(selectCanGenerate(useSteeringStore.getState())).toBe(false);
    });

    it('should return false without prompt', () => {
      useSteeringStore.setState({
        selectedSAE: createMockSAE(),
        selectedFeatures: [{ ...createMockFeature(), color: 'teal' }],
        prompt: '   ', // Whitespace only
      });

      expect(selectCanGenerate(useSteeringStore.getState())).toBe(false);
    });

    it('should return false while generating', () => {
      useSteeringStore.setState({
        selectedSAE: createMockSAE(),
        selectedFeatures: [{ ...createMockFeature(), color: 'teal' }],
        prompt: 'Test prompt',
        isGenerating: true,
      });

      expect(selectCanGenerate(useSteeringStore.getState())).toBe(false);
    });
  });

  describe('selectFeature', () => {
    it('should find feature by idx and layer', () => {
      const feature = { ...createMockFeature({ feature_idx: 100, layer: 6 }), color: 'teal' as const };
      useSteeringStore.setState({ selectedFeatures: [feature] });

      const found = selectFeature(100, 6)(useSteeringStore.getState());
      expect(found).toEqual(feature);
    });

    it('should return undefined for non-existent feature', () => {
      useSteeringStore.setState({
        selectedFeatures: [{ ...createMockFeature({ feature_idx: 100, layer: 6 }), color: 'teal' }],
      });

      const found = selectFeature(999, 6)(useSteeringStore.getState());
      expect(found).toBeUndefined();
    });
  });

  describe('selectAvailableColors', () => {
    it('should return all colors when none used', () => {
      useSteeringStore.setState({ selectedFeatures: [] });

      const colors = selectAvailableColors(useSteeringStore.getState());
      expect(colors).toEqual(FEATURE_COLOR_ORDER);
    });

    it('should exclude used colors', () => {
      useSteeringStore.setState({
        selectedFeatures: [
          { ...createMockFeature({ feature_idx: 1 }), color: 'teal' },
          { ...createMockFeature({ feature_idx: 2 }), color: 'purple' },
        ],
      });

      const colors = selectAvailableColors(useSteeringStore.getState());
      expect(colors).toEqual(['blue', 'amber']);
    });

    it('should return empty when all colors used', () => {
      useSteeringStore.setState({
        selectedFeatures: [
          { ...createMockFeature({ feature_idx: 1 }), color: 'teal' },
          { ...createMockFeature({ feature_idx: 2 }), color: 'blue' },
          { ...createMockFeature({ feature_idx: 3 }), color: 'purple' },
          { ...createMockFeature({ feature_idx: 4 }), color: 'amber' },
        ],
      });

      const colors = selectAvailableColors(useSteeringStore.getState());
      expect(colors).toEqual([]);
    });
  });
});

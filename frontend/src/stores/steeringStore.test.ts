/**
 * Unit tests for steeringStore.
 *
 * Tests cover:
 * - Feature selection (add, remove, duplicates, max limit)
 * - Double-submission prevention (isGenerating guard)
 * - Batch processing (sequential prompts, abort, error handling)
 * - Sweep mode (sequential sweeps on multiple features)
 * - WebSocket event handlers (completed, failed)
 * - Prompts management
 * - Validation
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { useSteeringStore, MAX_SELECTED_FEATURES, blendedTitle } from './steeringStore';
import { DEFAULT_GENERATION_PARAMS } from '../types/steering';
import type { SAE } from '../types/sae';
import type { SteeringComparisonResponse, StrengthSweepResponse } from '../types/steering';

// Mock the API module
vi.mock('../api/steering', () => ({
  submitAsyncComparison: vi.fn(),
  submitAsyncCombined: vi.fn(),
  computeClusterAllocation: vi.fn(),
  submitAsyncSweep: vi.fn(),
  cancelTask: vi.fn(),
  getExperiments: vi.fn(),
  saveExperiment: vi.fn(),
  deleteExperiment: vi.fn(),
  deleteExperimentsBatch: vi.fn(),
  getAsyncTaskResult: vi.fn(),
}));

// Import the mocked module
import * as steeringApi from '../api/steering';

// Initial state for reset
const initialState = {
  selectedSAE: null,
  selectedFeatures: [],
  prompts: [''],
  generationParams: { ...DEFAULT_GENERATION_PARAMS },
  advancedParams: null,
  showAdvancedParams: false,
  isGenerating: false,
  comparisonId: null,
  taskId: null,
  progress: 0,
  progressMessage: null,
  currentComparison: null,
  recentComparisons: [],
  batchState: null,
  clusterContext: null,
  clusterBudget: null,
  intensity: 1,
  combinedMode: false,
  isCombinedGenerating: false,
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
};

// Helper to reset store state between tests
const resetStore = () => {
  act(() => {
    useSteeringStore.setState(initialState);
  });
};

// Mock SAE for testing
const mockSAE: SAE = {
  id: 'sae-123',
  name: 'Test SAE',
  model_id: 'model-456',
  architecture: 'standard',
  layer: 6,
  d_model: 768,
  n_features: 4096,
  status: 'ready',
  path: '/path/to/sae',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
};

// Mock comparison response
const mockComparisonResponse: SteeringComparisonResponse = {
  comparison_id: 'comp-123',
  prompt: 'Test prompt',
  unsteered_output: 'Unsteered response',
  steered_outputs: [
    {
      feature_idx: 100,
      layer: 6,
      strength: 1.5,
      output: 'Steered response',
      color: 'emerald',
    },
  ],
  created_at: '2024-01-01T00:00:00Z',
};

// Mock sweep response
const mockSweepResponse: StrengthSweepResponse = {
  sweep_id: 'sweep-123',
  prompt: 'Test prompt',
  feature_idx: 100,
  layer: 6,
  strength_values: [0.5, 1.0, 1.5, 2.0],
  outputs: [
    { strength: 0.5, output: 'Output at 0.5' },
    { strength: 1.0, output: 'Output at 1.0' },
    { strength: 1.5, output: 'Output at 1.5' },
    { strength: 2.0, output: 'Output at 2.0' },
  ],
  created_at: '2024-01-01T00:00:00Z',
};

describe('steeringStore', () => {
  beforeEach(() => {
    resetStore();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Feature Selection', () => {
    it('should add a feature to selection', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      expect(result.current.selectedFeatures).toHaveLength(1);
      expect(result.current.selectedFeatures[0].feature_idx).toBe(100);
      expect(result.current.selectedFeatures[0].layer).toBe(6);
      expect(result.current.selectedFeatures[0].strength).toBe(1.5);
      expect(result.current.selectedFeatures[0].color).toBe('teal'); // First color in FEATURE_COLOR_ORDER
    });

    it('should assign different colors to each feature', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1.0 });
        result.current.addFeature({ feature_idx: 300, layer: 6, strength: 0.5 });
      });

      const colors = result.current.selectedFeatures.map((f) => f.color);
      expect(colors).toEqual(['teal', 'blue', 'purple']); // FEATURE_COLOR_ORDER: teal, blue, purple, amber
    });

    it('should enforce the maximum feature limit (Feature 011: 20)', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Fill up to the max
      act(() => {
        for (let i = 0; i < MAX_SELECTED_FEATURES; i++) {
          result.current.addFeature({ feature_idx: 100 + i, layer: 6, strength: 1.0 });
        }
      });

      expect(result.current.selectedFeatures).toHaveLength(MAX_SELECTED_FEATURES);
      expect(MAX_SELECTED_FEATURES).toBe(20);

      // One more past the limit - should be rejected
      let added: boolean | undefined;
      act(() => {
        added = result.current.addFeature({ feature_idx: 999, layer: 6, strength: 1.0 });
      });

      expect(added).toBe(false);
      expect(result.current.selectedFeatures).toHaveLength(MAX_SELECTED_FEATURES);
      expect(result.current.selectedFeatures.find((f) => f.feature_idx === 999)).toBeUndefined();
    });

    it('auto-computes a baseline strength from frequency when none is passed (Feature 011)', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        // freq 0.037 → clamp(2.9 − 2.6·0.037, 1, 3) ≈ 2.8, source 'auto'
        result.current.addFeature({
          feature_idx: 10,
          layer: 6,
          activation_frequency: 0.037,
        });
        // no frequency → default 10, source 'default'
        result.current.addFeature({ feature_idx: 11, layer: 6 });
        // explicit strength → honored verbatim, source 'manual'
        result.current.addFeature({ feature_idx: 12, layer: 6, strength: 42 });
      });

      const [auto, dflt, manual] = result.current.selectedFeatures;
      expect(auto.strength).toBeCloseTo(2.8, 1);
      expect(auto.strengthSource).toBe('auto');
      expect(dflt.strength).toBe(10);
      expect(dflt.strengthSource).toBe('default');
      expect(manual.strength).toBe(42);
      expect(manual.strengthSource).toBe('manual');
    });

    it('applyAutoBaseline recomputes every tile from its stored frequency (Feature 011)', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 20, layer: 6, strength: 100, activation_frequency: 0.484 });
        result.current.addFeature({ feature_idx: 21, layer: 6, strength: 100 });
      });

      // Both start manual at 100
      expect(result.current.selectedFeatures.every((f) => f.strength === 100)).toBe(true);

      act(() => {
        result.current.applyAutoBaseline();
      });

      const [withFreq, withoutFreq] = result.current.selectedFeatures;
      // freq 0.484 → clamp(2.9 − 2.6·0.484, 1, 3) ≈ 1.6
      expect(withFreq.strength).toBeCloseTo(1.6, 1);
      expect(withFreq.strengthSource).toBe('auto');
      expect(withoutFreq.strength).toBe(10);
      expect(withoutFreq.strengthSource).toBe('default');
    });

    it('should allow same feature at different strengths with unique instance_id', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.0 });
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 2.0 });
      });

      expect(result.current.selectedFeatures).toHaveLength(2);
      // Both should have same feature_idx but different instance_id
      expect(result.current.selectedFeatures[0].feature_idx).toBe(100);
      expect(result.current.selectedFeatures[1].feature_idx).toBe(100);
      expect(result.current.selectedFeatures[0].instance_id).not.toBe(
        result.current.selectedFeatures[1].instance_id
      );
      // Different strengths
      expect(result.current.selectedFeatures[0].strength).toBe(1.0);
      expect(result.current.selectedFeatures[1].strength).toBe(2.0);
    });

    it('should remove a feature by instance_id', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1.0 });
      });

      const instanceIdToRemove = result.current.selectedFeatures[0].instance_id;

      act(() => {
        result.current.removeFeature(instanceIdToRemove);
      });

      expect(result.current.selectedFeatures).toHaveLength(1);
      expect(result.current.selectedFeatures[0].feature_idx).toBe(200);
    });

    it('should update feature strength', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      const instanceId = result.current.selectedFeatures[0].instance_id;

      act(() => {
        result.current.updateFeatureStrength(instanceId, 3.0);
      });

      expect(result.current.selectedFeatures[0].strength).toBe(3.0);
    });

    it('should clear all features', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1.0 });
        result.current.clearFeatures();
      });

      expect(result.current.selectedFeatures).toHaveLength(0);
    });
  });

  describe('Double-Submission Prevention', () => {
    it('should set isGenerating before API call', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up SAE and prompt
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      // Mock API to never resolve (simulates in-flight request)
      vi.mocked(steeringApi.submitAsyncComparison).mockImplementation(
        () => new Promise(() => {})
      );

      // Start generation without waiting
      act(() => {
        result.current.generateComparison();
      });

      // Should be generating now
      expect(result.current.isGenerating).toBe(true);
    });

    it('should return ignored status when isGenerating is true', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up SAE and prompt
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      // Mock API to never resolve
      vi.mocked(steeringApi.submitAsyncComparison).mockImplementation(
        () => new Promise(() => {})
      );

      // Start first generation
      act(() => {
        result.current.generateComparison();
      });

      // Try second generation - should return ignored status (not throw)
      let secondResponse: unknown;
      await act(async () => {
        secondResponse = await result.current.generateComparison();
      });

      // Second call should return with status 'ignored'
      expect((secondResponse as { status: string }).status).toBe('ignored');
      // API should only have been called once
      expect(steeringApi.submitAsyncComparison).toHaveBeenCalledTimes(1);
    });

    it('should clear isGenerating on error', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up SAE and prompt
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      // Mock API to reject
      const apiError = new Error('API Error');
      vi.mocked(steeringApi.submitAsyncComparison).mockRejectedValue(apiError);

      // Call generateComparison and expect it to throw
      let thrownError: unknown;
      await act(async () => {
        try {
          await result.current.generateComparison();
        } catch (e) {
          thrownError = e;
        }
      });

      // Verify the error was thrown
      expect(thrownError).toBe(apiError);
      // Should no longer be generating
      expect(result.current.isGenerating).toBe(false);
      // Error should be set in state
      expect(result.current.error).toBe('API Error');
    });
  });

  describe('WebSocket Event Handlers', () => {
    it('should update state on handleAsyncCompleted for comparison', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set taskId to simulate active task
      act(() => {
        useSteeringStore.setState({ taskId: 'task-123', isGenerating: true });
      });

      act(() => {
        result.current.handleAsyncCompleted(mockComparisonResponse);
      });

      expect(result.current.isGenerating).toBe(false);
      expect(result.current.currentComparison).toEqual(mockComparisonResponse);
      expect(result.current.progress).toBe(100);
      // Should be added to recent comparisons
      expect(result.current.recentComparisons).toHaveLength(1);
      expect(result.current.recentComparisons[0].id).toBe('comp-123');
    });

    it('should set error on handleAsyncFailed', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        useSteeringStore.setState({ taskId: 'task-123', isGenerating: true });
      });

      act(() => {
        result.current.handleAsyncFailed('Task failed: timeout');
      });

      expect(result.current.isGenerating).toBe(false);
      expect(result.current.error).toBe('Task failed: timeout');
    });

    it('should add to recentComparisons and keep only last 10', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Add 12 comparisons
      for (let i = 0; i < 12; i++) {
        const response = {
          ...mockComparisonResponse,
          comparison_id: `comp-${i}`,
        };
        act(() => {
          result.current.handleAsyncCompleted(response);
        });
      }

      // Should only keep 10
      expect(result.current.recentComparisons).toHaveLength(10);
      // Most recent should be first
      expect(result.current.recentComparisons[0].id).toBe('comp-11');
      // Oldest kept should be comp-2 (0 and 1 were dropped)
      expect(result.current.recentComparisons[9].id).toBe('comp-2');
    });
  });

  describe('Sweep Mode', () => {
    it('should set isSweeping during sweep', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up SAE and prompt
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
      });

      // Mock API to return task ID
      vi.mocked(steeringApi.submitAsyncSweep).mockResolvedValue({
        task_id: 'sweep-task-123',
        status: 'pending',
      });

      // Start sweep without waiting (will hang on resolver)
      const sweepPromise = result.current.runStrengthSweep(100, 6, [0.5, 1.0, 1.5]);

      // Need to wait for state update
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 100));
      });

      expect(result.current.isSweeping).toBe(true);
      expect(result.current.taskId).toBe('sweep-task-123');

      // Simulate WebSocket completion
      act(() => {
        result.current.handleAsyncCompleted(mockSweepResponse);
      });

      // Wait for sweep to complete
      const sweepResult = await sweepPromise;

      expect(result.current.isSweeping).toBe(false);
      expect(result.current.sweepResults).toEqual(mockSweepResponse);
      expect(sweepResult).toEqual(mockSweepResponse);
    });

    it('should handle sweep result via handleAsyncCompleted', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up as if sweep is in progress
      act(() => {
        useSteeringStore.setState({
          isSweeping: true,
          taskId: 'sweep-task-123',
        });
      });

      // Note: Can't test the resolver directly since it's module-level
      // But we can verify the sweep detection logic
      act(() => {
        result.current.handleAsyncCompleted(mockSweepResponse);
      });

      // Sweep results should NOT be stored via handleAsyncCompleted
      // (it's handled by the resolver in runStrengthSweep)
      // But we can verify it's detected as a sweep result
      expect(result.current.currentComparison).toBeNull(); // Not a comparison
    });

    it('should handle sequential sweeps on multiple features', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up SAE and prompt
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
      });

      // First sweep on feature 100
      vi.mocked(steeringApi.submitAsyncSweep).mockResolvedValueOnce({
        task_id: 'sweep-task-1',
        status: 'pending',
      });

      const sweep1Promise = result.current.runStrengthSweep(100, 6, [0.5, 1.0]);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 100));
      });

      expect(result.current.isSweeping).toBe(true);

      // Complete first sweep
      const sweep1Response = { ...mockSweepResponse, sweep_id: 'sweep-1', feature_idx: 100 };
      act(() => {
        result.current.handleAsyncCompleted(sweep1Response);
      });

      await sweep1Promise;
      expect(result.current.isSweeping).toBe(false);

      // Second sweep on feature 200
      vi.mocked(steeringApi.submitAsyncSweep).mockResolvedValueOnce({
        task_id: 'sweep-task-2',
        status: 'pending',
      });

      const sweep2Promise = result.current.runStrengthSweep(200, 6, [1.0, 2.0]);

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 100));
      });

      expect(result.current.isSweeping).toBe(true);

      // Complete second sweep
      const sweep2Response = { ...mockSweepResponse, sweep_id: 'sweep-2', feature_idx: 200 };
      act(() => {
        result.current.handleAsyncCompleted(sweep2Response);
      });

      const result2 = await sweep2Promise;
      expect(result.current.isSweeping).toBe(false);
      expect(result2.feature_idx).toBe(200);
    });

    it('should require SAE for sweep', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // No SAE set
      act(() => {
        result.current.updatePrompt(0, 'Test prompt');
      });

      await expect(
        result.current.runStrengthSweep(100, 6, [0.5, 1.0])
      ).rejects.toThrow('No SAE selected');
    });

    it('should require prompt for sweep', async () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.selectSAE(mockSAE);
        // No prompt set
      });

      await expect(
        result.current.runStrengthSweep(100, 6, [0.5, 1.0])
      ).rejects.toThrow('Prompt is required');
    });

    it('should clear sweep results', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set sweep results
      act(() => {
        useSteeringStore.setState({ sweepResults: mockSweepResponse });
      });

      act(() => {
        result.current.clearSweepResults();
      });

      expect(result.current.sweepResults).toBeNull();
    });
  });

  describe('Batch Processing', () => {
    it('should process prompts sequentially', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up with multiple prompts
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['Prompt 1', 'Prompt 2', 'Prompt 3']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      // Mock API to return task IDs
      let callCount = 0;
      vi.mocked(steeringApi.submitAsyncComparison).mockImplementation(async () => {
        callCount++;
        return {
          task_id: `task-${callCount}`,
          status: 'pending',
        };
      });

      // Start batch
      const batchPromise = result.current.generateBatchComparison();

      // Wait a bit for first task submission
      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 150));
      });

      expect(result.current.batchState?.isRunning).toBe(true);
      expect(result.current.batchState?.results).toHaveLength(3);

      // Complete each prompt via WebSocket
      for (let i = 1; i <= 3; i++) {
        const response = {
          ...mockComparisonResponse,
          comparison_id: `comp-${i}`,
          prompt: `Prompt ${i}`,
        };
        act(() => {
          result.current.handleAsyncCompleted(response);
        });
        // Small delay between completions
        await act(async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
        });
      }

      await batchPromise;

      expect(result.current.batchState?.isRunning).toBe(false);
      expect(result.current.isGenerating).toBe(false);
    });

    it('runs Blended per-prompt in batch mode (combinedMode) and completes all', async () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['Prompt 1', 'Prompt 2', 'Prompt 3']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 2.0 });
        result.current.setCombinedMode(true); // Blended
      });

      let callCount = 0;
      vi.mocked(steeringApi.submitAsyncCombined).mockImplementation(async () => {
        callCount++;
        return { task_id: `ctask-${callCount}`, status: 'pending' } as any;
      });

      const batchPromise = result.current.generateBatchComparison();
      await act(async () => {
        await new Promise((r) => setTimeout(r, 150));
      });

      // Must use the COMBINED endpoint, not comparison
      expect(steeringApi.submitAsyncCombined).toHaveBeenCalled();
      expect(steeringApi.submitAsyncComparison).not.toHaveBeenCalled();
      expect(result.current.batchState?.isRunning).toBe(true);
      expect(result.current.batchState?.results).toHaveLength(3);

      // Complete each prompt with a COMBINED result (has combined_id).
      for (let i = 1; i <= 3; i++) {
        const combined = {
          combined_id: `cmb-${i}`,
          sae_id: mockSAE.id,
          model_id: 'm',
          prompt: `Prompt ${i}`,
          combined_output: `blended output ${i}`,
          features_applied: [
            { feature_idx: 100, layer: 6, strength: 1.5, label: null, color: 'teal' },
            { feature_idx: 200, layer: 6, strength: 2.0, label: null, color: 'blue' },
          ],
          baseline_output: `baseline ${i}`,
          combined_metrics: null,
          baseline_metrics: null,
          total_steering_strength: 3.5,
          total_time_ms: 1000,
          created_at: new Date().toISOString(),
        };
        act(() => {
          result.current.handleAsyncCompleted(combined as any);
        });
        await act(async () => {
          await new Promise((r) => setTimeout(r, 100));
        });
      }

      await batchPromise;

      expect(result.current.batchState?.isRunning).toBe(false);
      expect(result.current.isGenerating).toBe(false);
      // All three prompts completed, each carrying an adapted comparison result.
      const statuses = result.current.batchState?.results.map((r) => r.status);
      expect(statuses).toEqual(['completed', 'completed', 'completed']);
      // The adapter produced a single "blended" steered variation per prompt.
      const first = result.current.batchState?.results[0].comparison;
      expect(first?.steered).toHaveLength(1);
      expect(first?.steered[0].text).toBe('blended output 1');
    });

    it('should continue on individual prompt failure', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up with 2 prompts
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['Prompt 1', 'Prompt 2']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      let callCount = 0;
      vi.mocked(steeringApi.submitAsyncComparison).mockImplementation(async () => {
        callCount++;
        return {
          task_id: `task-${callCount}`,
          status: 'pending',
        };
      });

      const batchPromise = result.current.generateBatchComparison();

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 150));
      });

      // Fail first prompt
      act(() => {
        result.current.handleAsyncFailed('First prompt failed');
      });

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 150));
      });

      // Complete second prompt
      const response2 = {
        ...mockComparisonResponse,
        comparison_id: 'comp-2',
        prompt: 'Prompt 2',
      };
      act(() => {
        result.current.handleAsyncCompleted(response2);
      });

      await batchPromise;

      // First should be failed, second should be completed
      expect(result.current.batchState?.results[0].status).toBe('failed');
      expect(result.current.batchState?.results[0].error).toBe('First prompt failed');
      expect(result.current.batchState?.results[1].status).toBe('completed');
    });

    it('should stop when aborted', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set up with multiple prompts
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['Prompt 1', 'Prompt 2', 'Prompt 3']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
      });

      let callCount = 0;
      vi.mocked(steeringApi.submitAsyncComparison).mockImplementation(async () => {
        callCount++;
        return {
          task_id: `task-${callCount}`,
          status: 'pending',
        };
      });

      vi.mocked(steeringApi.cancelTask).mockResolvedValue({
        task_id: 'task-1',
        status: 'cancelled',
        message: 'Task cancelled',
      });

      // Start batch
      const batchPromise = result.current.generateBatchComparison();

      await act(async () => {
        await new Promise((resolve) => setTimeout(resolve, 150));
      });

      expect(result.current.batchState?.isRunning).toBe(true);

      // Abort
      act(() => {
        result.current.abortBatch();
      });

      await batchPromise;

      expect(result.current.batchState?.isRunning).toBe(false);
      expect(result.current.batchState?.aborted).toBe(true);
    });

    it('should clear batch results', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set batch state
      act(() => {
        useSteeringStore.setState({
          batchState: {
            isRunning: false,
            aborted: false,
            currentIndex: 2,
            results: [
              { prompt: 'P1', status: 'completed', comparison: mockComparisonResponse },
              { prompt: 'P2', status: 'completed', comparison: mockComparisonResponse },
            ],
          },
        });
      });

      act(() => {
        result.current.clearBatchResults();
      });

      expect(result.current.batchState).toBeNull();
    });
  });

  describe('Prompts Management', () => {
    it('should set a single prompt', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.updatePrompt(0, 'Hello world');
      });

      expect(result.current.prompts[0]).toBe('Hello world');
    });

    it('should set multiple prompts', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setPrompts(['Prompt 1', 'Prompt 2', 'Prompt 3']);
      });

      expect(result.current.prompts).toEqual(['Prompt 1', 'Prompt 2', 'Prompt 3']);
    });

    it('should add a new prompt', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setPrompts(['Prompt 1']);
        result.current.addPrompt();
      });

      expect(result.current.prompts).toHaveLength(2);
      expect(result.current.prompts[1]).toBe('');
    });

    it('should remove a prompt', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setPrompts(['Prompt 1', 'Prompt 2', 'Prompt 3']);
        result.current.removePrompt(1);
      });

      expect(result.current.prompts).toEqual(['Prompt 1', 'Prompt 3']);
    });

    it('should not remove the last prompt', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setPrompts(['Only prompt']);
        result.current.removePrompt(0);
      });

      // Should still have one prompt
      expect(result.current.prompts).toHaveLength(1);
    });
  });

  describe('Validation', () => {
    it('should require SAE for generation', async () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.updatePrompt(0, 'Test prompt');
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        // No SAE set
      });

      await expect(result.current.generateComparison()).rejects.toThrow('No SAE selected');
    });

    it('should require prompt for generation', async () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        // Empty prompt
        result.current.updatePrompt(0, '');
      });

      await expect(result.current.generateComparison()).rejects.toThrow('Prompt is required');
    });

    it('should require at least one feature for generation', async () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
        // No features
      });

      await expect(result.current.generateComparison()).rejects.toThrow(
        'No features selected'
      );
    });
  });

  describe('SAE Selection', () => {
    it('should set selected SAE', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.selectSAE(mockSAE);
      });

      expect(result.current.selectedSAE).toEqual(mockSAE);
    });

    it('should clear selected SAE', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.selectSAE(null);
      });

      expect(result.current.selectedSAE).toBeNull();
    });
  });

  describe('Generation Parameters', () => {
    it('should update generation params', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setGenerationParams({
          max_new_tokens: 100,
          temperature: 0.8,
        });
      });

      expect(result.current.generationParams.max_new_tokens).toBe(100);
      expect(result.current.generationParams.temperature).toBe(0.8);
    });

    it('should preserve other params when updating', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Get initial value
      const initialTopP = result.current.generationParams.top_p;

      act(() => {
        result.current.setGenerationParams({
          max_new_tokens: 200,
        });
      });

      // top_p should be unchanged
      expect(result.current.generationParams.top_p).toBe(initialTopP);
      expect(result.current.generationParams.max_new_tokens).toBe(200);
    });
  });

  describe('Error Handling', () => {
    it('should set error', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setError('Something went wrong');
      });

      expect(result.current.error).toBe('Something went wrong');
    });

    it('should clear error', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.setError('Something went wrong');
        result.current.clearError();
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('State Reset via setState', () => {
    it('should reset state via useSteeringStore.setState', () => {
      const { result } = renderHook(() => useSteeringStore());

      // Set various state
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.updatePrompt(0, 'Test prompt');
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.setError('Error');
      });

      // Verify state was set
      expect(result.current.selectedSAE).toEqual(mockSAE);
      expect(result.current.selectedFeatures).toHaveLength(1);

      // Reset state using setState (the way the test helper does it)
      act(() => {
        useSteeringStore.setState({
          selectedSAE: null,
          selectedFeatures: [],
          prompts: [''],
          error: null,
        });
      });

      expect(result.current.selectedSAE).toBeNull();
      expect(result.current.prompts).toEqual(['']);
      expect(result.current.selectedFeatures).toHaveLength(0);
      expect(result.current.error).toBeNull();
    });
  });

  describe('Cluster Context (Feature 012)', () => {
    const ctx = { group_id: 'g1', display_token: 'fear' };

    it('setClusterContext stores and clears', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.setClusterContext(ctx));
      expect(result.current.clusterContext).toEqual(ctx);
      act(() => result.current.setClusterContext(null));
      expect(result.current.clusterContext).toBeNull();
    });

    it('is cleared by every selection mutation', () => {
      const { result } = renderHook(() => useSteeringStore());

      // addFeature clears
      act(() => {
        result.current.setClusterContext(ctx);
        result.current.addFeature({ feature_idx: 1, layer: 6, strength: 1 });
      });
      expect(result.current.clusterContext).toBeNull();

      // removeFeature clears
      act(() => result.current.setClusterContext(ctx));
      const inst = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.removeFeature(inst));
      expect(result.current.clusterContext).toBeNull();

      // duplicateFeature clears
      act(() => {
        result.current.addFeature({ feature_idx: 2, layer: 6, strength: 1 });
      });
      const inst2 = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.setClusterContext(ctx));
      act(() => result.current.duplicateFeature(inst2));
      expect(result.current.clusterContext).toBeNull();

      // clearFeatures clears
      act(() => result.current.setClusterContext(ctx));
      act(() => result.current.clearFeatures());
      expect(result.current.clusterContext).toBeNull();

      // selectSAE clears
      act(() => result.current.setClusterContext(ctx));
      act(() => result.current.selectSAE(mockSAE));
      expect(result.current.clusterContext).toBeNull();
    });

    it('survives strength edits (not a selection mutation)', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.addFeature({ feature_idx: 1, layer: 6, strength: 1 });
        result.current.setClusterContext(ctx);
      });
      const inst = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.updateFeatureStrength(inst, 2.5));
      expect(result.current.clusterContext).toEqual(ctx);
    });

    it('blendedTitle: token tier, generic tier, and honest single-feature titles', () => {
      expect(blendedTitle(null, 3)).toBe('Blended (3 features)');
      expect(blendedTitle(ctx, 3)).toBe('fear — Blended (3 features)');
      // Single feature: the member's own identity, cluster-prefixed when known
      expect(blendedTitle(ctx, 1, 'anxiety spike')).toBe('fear — anxiety spike');
      expect(blendedTitle(null, 1, 'anxiety spike')).toBe('anxiety spike');
      expect(blendedTitle(null, 1)).toBe('Blended (1 feature)');
    });

    it('loadExperiment and resetSession clear stale cluster context + combined results', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.setClusterContext(ctx);
        useSteeringStore.setState({
          combinedResults: { combined_id: 'x' } as any,
          combinedResultsTitle: 'fear — Blended (3 features)',
        });
      });
      act(() =>
        result.current.loadExperiment({
          id: 'e1',
          name: 'exp',
          selected_features: [],
          prompt: 'p',
          generation_params: { ...DEFAULT_GENERATION_PARAMS },
          results: { comparison_id: 'c1' } as any,
          created_at: new Date().toISOString(),
        } as any),
      );
      expect(result.current.clusterContext).toBeNull();
      expect(result.current.combinedResults).toBeNull();
      expect(result.current.combinedResultsTitle).toBeNull();

      // resetSession (first click) also clears combined results
      act(() => {
        result.current.setClusterContext(ctx);
        useSteeringStore.setState({
          combinedResults: { combined_id: 'y' } as any,
          combinedResultsTitle: 't',
        });
      });
      act(() => result.current.resetSession());
      expect(result.current.combinedResults).toBeNull();
      expect(result.current.combinedResultsTitle).toBeNull();
    });

    it('combined completion carries applied_features + cluster title into batch results', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['P1']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 2.0 });
        result.current.setCombinedMode(true);
        result.current.setPrompts(['P1', 'P2']); // batch
        result.current.setClusterContext(ctx);
      });

      vi.mocked(steeringApi.submitAsyncCombined).mockImplementation(async () => ({
        task_id: 'ctask-ctx', status: 'pending',
      }) as any);

      const batchPromise = result.current.generateBatchComparison();
      await act(async () => { await new Promise((r) => setTimeout(r, 120)); });

      const combined = {
        combined_id: 'cmb-ctx', sae_id: mockSAE.id, model_id: 'm', prompt: 'P1',
        combined_output: 'out',
        features_applied: [
          { feature_idx: 100, layer: 6, strength: 1.5, label: null, color: 'teal' },
          { feature_idx: 200, layer: 6, strength: 2.0, label: null, color: 'blue' },
        ],
        baseline_output: null, combined_metrics: null, baseline_metrics: null,
        total_steering_strength: 3.5, total_time_ms: 10, created_at: new Date().toISOString(),
      };
      act(() => { result.current.handleAsyncCompleted(combined as any); });
      await act(async () => { await new Promise((r) => setTimeout(r, 80)); });
      // finish prompt 2
      act(() => { result.current.handleAsyncCompleted({ ...combined, combined_id: 'cmb-ctx2', prompt: 'P2' } as any); });
      await act(async () => { await new Promise((r) => setTimeout(r, 80)); });
      await batchPromise;

      const first = result.current.batchState?.results[0].comparison;
      expect(first?.applied_features).toHaveLength(2);
      expect(first?.steered[0].feature_config.label).toBe('fear — Blended (2 features)');
    });
  });

  describe('Cluster Strength Budget (Feature 013)', () => {
    // weightsByInstance content only matters for rebalance; the intensity test
    // needs mere truthiness.
    const budget = {
      B: 2.4, B_dir: 2.4, G: 1.0, flags: [], approximate: false,
      weightsByInstance: {} as Record<string, number>,
    };

    const setupCluster = (result: any) => {
      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.2 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 0.7 });
        result.current.addFeature({ feature_idx: 300, layer: 6, strength: 0.5 });
      });
      const [a, b, c] = result.current.selectedFeatures.map((f: any) => f.instance_id);
      act(() => {
        useSteeringStore.setState({
          clusterBudget: { ...budget, weightsByInstance: { [a]: 0.5, [b]: 0.3, [c]: 0.2 } },
        });
      });
    };

    it('rebalanceStrength pins the edit and preserves the total budget', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      const inst = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.rebalanceStrength(inst, 1.6));
      const feats = result.current.selectedFeatures;
      expect(feats[0].strength).toBe(1.6);
      expect(feats[0].pinned).toBe(true);
      const total = feats.reduce((s, f) => s + Math.abs(f.strength), 0);
      expect(total).toBeCloseTo(2.4, 1);
      // Unpinned redistribute by renormalized weights (0.3:0.2 of 0.8)
      expect(feats[1].strength).toBeCloseTo(0.5, 1);
      expect(feats[2].strength).toBeCloseTo(0.3, 1);
    });

    it('property: random edit sequences keep Σ|strength| within grain of B', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      let seed = 42;
      const rand = () => (seed = (seed * 1103515245 + 12345) % 2 ** 31) / 2 ** 31;
      for (let i = 0; i < 25; i++) {
        const feats = result.current.selectedFeatures;
        const unpinnedIdx = feats.map((f, j) => (f.pinned ? -1 : j)).filter((j) => j >= 0);
        if (unpinnedIdx.length <= 1) break; // all-but-one pinned: rebalance can no longer conserve
        const pick = unpinnedIdx[Math.floor(rand() * unpinnedIdx.length)];
        const value = Math.round(rand() * 15) / 10; // 0..1.5
        act(() => result.current.rebalanceStrength(feats[pick].instance_id, value));
        const total = result.current.selectedFeatures.reduce((s, f) => s + Math.abs(f.strength), 0);
        const pinnedTotal = result.current.selectedFeatures
          .filter((f) => f.pinned).reduce((s, f) => s + Math.abs(f.strength), 0);
        if (pinnedTotal <= 2.4) {
          expect(total).toBeGreaterThanOrEqual(2.4 - 0.15);
          expect(total).toBeLessThanOrEqual(2.4 + 0.15);
        }
      }
    });

    it('over-budget pin zeroes unpinned members, never rescales pins', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      const inst = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.rebalanceStrength(inst, 5.0));
      const feats = result.current.selectedFeatures;
      expect(feats[0].strength).toBe(5.0);
      expect(feats[1].strength).toBe(0);
      expect(feats[2].strength).toBe(0);
    });

    it('rebalanceStrength without a budget is a plain edit', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.addFeature({ feature_idx: 1, layer: 6, strength: 1 }));
      const inst = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.rebalanceStrength(inst, 7));
      expect(result.current.selectedFeatures[0].strength).toBe(7);
      expect(result.current.selectedFeatures[0].pinned).toBeFalsy();
    });

    it('applyStrengthPreset exits cluster mode', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      act(() => result.current.applyStrengthPreset(50));
      expect(result.current.clusterBudget).toBeNull();
      expect(result.current.selectedFeatures.every((f) => f.strength === 50)).toBe(true);
    });

    it('selection mutations clear the budget', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      act(() => result.current.addFeature({ feature_idx: 400, layer: 6, strength: 1 }));
      expect(result.current.clusterBudget).toBeNull();
    });

    it('intensity is applied once, at request-build time', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['P']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 2.0 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1.0 });
        useSteeringStore.setState({ clusterBudget: budget, intensity: 0.5 });
      });
      let captured: any = null;
      vi.mocked(steeringApi.submitAsyncCombined).mockImplementation(async (req: any) => {
        captured = req;
        return { task_id: 'ti', status: 'pending' } as any;
      });
      const p = result.current.generateCombined(true, false).catch(() => {});
      await act(async () => { await new Promise((r) => setTimeout(r, 80)); });
      expect(captured.selected_features.map((f: any) => f.strength)).toEqual([1.0, 0.5]);
      // Tiles still show pre-λ values
      expect(result.current.selectedFeatures.map((f) => f.strength)).toEqual([2.0, 1.0]);
      act(() => { useSteeringStore.setState({ error: null }); });
      void p;
    });

    it('requestClusterAllocation applies strengths + drops stale responses', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 10, activation_frequency: 0.2, similarity: 0.8 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 10, activation_frequency: 0.2, similarity: 0.8 });
      });
      vi.mocked(steeringApi.computeClusterAllocation).mockResolvedValue({
        B: 2.4, B_dir: 2.4, G: 1.0, f_eff: 0.2,
        weights: [0.5, 0.5], strengths: [1.2, 1.2], flags: [],
        cancellation_pair: null, constants_used: {}, formula_id: 'freq-budget/sim-alloc@1',
        approximate: false,
      } as any);
      await act(async () => { await result.current.requestClusterAllocation(0.8); });
      const feats = result.current.selectedFeatures;
      expect(feats.map((f) => f.strength)).toEqual([1.2, 1.2]);
      expect(feats.every((f) => f.strengthSource === 'cluster')).toBe(true);
      expect(result.current.clusterBudget?.B).toBe(2.4);

      // Stale: selection changes while request in flight
      let resolveFn: any;
      vi.mocked(steeringApi.computeClusterAllocation).mockImplementation(
        () => new Promise((res) => { resolveFn = res; }),
      );
      const pending = result.current.requestClusterAllocation(0.8);
      act(() => { result.current.addFeature({ feature_idx: 300, layer: 6, strength: 5 }); });
      resolveFn({ B: 9, B_dir: 9, G: 1, f_eff: 0.1, weights: [0.5, 0.5], strengths: [4.5, 4.5],
        flags: [], cancellation_pair: null, constants_used: {}, formula_id: 'x', approximate: false });
      await act(async () => { await pending; });
      // Budget was cleared by the mutation and the stale response dropped
      expect(result.current.clusterBudget).toBeNull();
      expect(result.current.selectedFeatures.find((f) => f.feature_idx === 100)?.strength).not.toBe(4.5);
    });

    it('batch titles use the ctx snapshot even when context clears mid-batch (TE-3)', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['P1', 'P2']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1 });
        result.current.setCombinedMode(true);
        result.current.setClusterContext({ group_id: 'g1', display_token: 'fear' });
      });
      vi.mocked(steeringApi.submitAsyncCombined).mockImplementation(async () => ({
        task_id: `t-${Math.random()}`, status: 'pending',
      }) as any);
      const batch = result.current.generateBatchComparison();
      await act(async () => { await new Promise((r) => setTimeout(r, 120)); });

      const combined = (id: string, prompt: string) => ({
        combined_id: id, sae_id: mockSAE.id, model_id: 'm', prompt,
        combined_output: 'o',
        features_applied: [
          { feature_idx: 100, layer: 6, strength: 1, label: null, color: 'teal' },
          { feature_idx: 200, layer: 6, strength: 1, label: null, color: 'blue' },
        ],
        baseline_output: null, combined_metrics: null, baseline_metrics: null,
        total_steering_strength: 2, total_time_ms: 1, created_at: new Date().toISOString(),
      });
      // Prompt 1 completes with ctx intact
      act(() => { result.current.handleAsyncCompleted(combined('c1', 'P1') as any); });
      await act(async () => { await new Promise((r) => setTimeout(r, 80)); });
      // Context cleared mid-batch (e.g. another hand-off elsewhere)
      act(() => { result.current.setClusterContext(null); });
      // Prompt 2 completes AFTER the clear — must still use the snapshot title
      act(() => { result.current.handleAsyncCompleted(combined('c2', 'P2') as any); });
      await act(async () => { await new Promise((r) => setTimeout(r, 80)); });
      await batch;

      const titles = result.current.batchState?.results.map(
        (r) => r.comparison?.steered[0].feature_config.label,
      );
      expect(titles).toEqual([
        'fear — Blended (2 features)',
        'fear — Blended (2 features)',
      ]);
    });

    it('low_cohesion keeps solo baselines (gate)', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.addFeature({ feature_idx: 100, layer: 6, activation_frequency: 0.2 });
        result.current.addFeature({ feature_idx: 200, layer: 6, activation_frequency: 0.2 });
      });
      const before = result.current.selectedFeatures.map((f) => f.strength);
      vi.mocked(steeringApi.computeClusterAllocation).mockResolvedValue({
        B: 2.4, B_dir: 2.4, G: 1.0, f_eff: 0.2, weights: [0.5, 0.5], strengths: [1.2, 1.2],
        flags: ['low_cohesion'], cancellation_pair: null, constants_used: {},
        formula_id: 'x', approximate: false,
      } as any);
      await act(async () => { await result.current.requestClusterAllocation(0.2); });
      expect(result.current.selectedFeatures.map((f) => f.strength)).toEqual(before);
      // Gated: no governing budget (rebalance/λ/bar must not engage), notice set.
      expect(result.current.clusterBudget).toBeNull();
      expect(result.current.clusterNotice).toMatch(/cohesion/i);
    });

    it('duplicate feature indices refuse allocation (v1 single-membership)', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 10 });
      });
      act(() => { result.current.duplicateFeature(result.current.selectedFeatures[0].instance_id); });
      vi.mocked(steeringApi.computeClusterAllocation).mockClear();
      await act(async () => { await result.current.requestClusterAllocation(0.9); });
      expect(steeringApi.computeClusterAllocation).not.toHaveBeenCalled();
      expect(result.current.clusterBudget).toBeNull();
    });

    it('allocation maps strengths by instance from the REQUEST order (reorder-safe)', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 10 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 10 });
      });
      let resolveFn: any;
      vi.mocked(steeringApi.computeClusterAllocation).mockImplementation(
        () => new Promise((res) => { resolveFn = res; }),
      );
      const pending = result.current.requestClusterAllocation(0.9);
      // User drags tiles into reverse order while the request is in flight —
      // same instance set, so the stale guard passes.
      act(() => {
        useSteeringStore.setState({
          selectedFeatures: [...useSteeringStore.getState().selectedFeatures].reverse(),
        });
      });
      // Response arrays are aligned with the request order: [100, 200]
      resolveFn({ B: 3, B_dir: 3, G: 1, f_eff: 0.1, weights: [0.6, 0.4], strengths: [1.8, -1.2],
        flags: [], cancellation_pair: null, constants_used: {}, formula_id: 'x', approximate: false });
      await act(async () => { await pending; });
      const byIdx = Object.fromEntries(
        result.current.selectedFeatures.map((f) => [f.feature_idx, f.strength]),
      );
      // 100 got 1.8, 200 got −1.2 (sign carried), regardless of tile order.
      expect(byIdx[100]).toBe(1.8);
      expect(byIdx[200]).toBe(-1.2);
    });

    it('λ applies in Compare requests too (parity with Blended)', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => {
        result.current.selectSAE(mockSAE);
        result.current.setPrompts(['P']);
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 2.0 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1.0 });
        useSteeringStore.setState({ clusterBudget: budget, intensity: 0.5 });
      });
      let captured: any = null;
      vi.mocked(steeringApi.submitAsyncComparison).mockImplementation(async (req: any) => {
        captured = req;
        return { task_id: 'tc', status: 'pending' } as any;
      });
      const p = result.current.generateComparison(true, false).catch(() => {});
      await act(async () => { await new Promise((r) => setTimeout(r, 80)); });
      expect(captured.selected_features.map((f: any) => f.strength)).toEqual([1.0, 0.5]);
      expect(result.current.selectedFeatures.map((f) => f.strength)).toEqual([2.0, 1.0]);
      act(() => { useSteeringStore.setState({ error: null }); });
      void p;
    });

    it('persist partialize strips pinned and demotes cluster→manual', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      act(() => {
        useSteeringStore.setState({
          selectedFeatures: useSteeringStore.getState().selectedFeatures.map((f, i) => ({
            ...f,
            pinned: i === 0,
            strengthSource: 'cluster' as const,
          })),
        });
      });
      const raw = localStorage.getItem('miStudio-steering');
      expect(raw).toBeTruthy();
      const persisted = JSON.parse(raw!).state.selectedFeatures;
      expect(persisted.every((f: any) => f.pinned === false)).toBe(true);
      expect(persisted.every((f: any) => f.strengthSource === 'manual')).toBe(true);
      // Live state untouched
      expect(result.current.selectedFeatures[0].pinned).toBe(true);
    });

    it('togglePin unpins a pinned member', () => {
      const { result } = renderHook(() => useSteeringStore());
      setupCluster(result);
      const inst = result.current.selectedFeatures[0].instance_id;
      act(() => result.current.rebalanceStrength(inst, 1.6));
      expect(result.current.selectedFeatures[0].pinned).toBe(true);
      act(() => result.current.togglePin(inst));
      expect(result.current.selectedFeatures[0].pinned).toBe(false);
    });

    // ── Feature 014: profile hydration ──────────────────────────────────

    const mockProfile = {
      id: 'clp_1',
      sae_id: mockSAE.id,
      model_id: null,
      extraction_id: null,
      source_group_id: 'grp_9',
      name: 'Fear response',
      narrative: null,
      display_token: 'fear',
      members: [
        { feature_idx: 100, strength: 1.4, pinned: true, similarity: 0.9,
          activation_frequency: 0.2, max_activation: 5, label: 'fear', sign: 1 as const },
        { feature_idx: 200, strength: -0.6, pinned: false, similarity: 0.7,
          activation_frequency: null, max_activation: null, label: null, sign: -1 as const },
      ],
      budget: { B: 2.0, B_dir: 2.2, G: 1.1, intensity: 1.5 },
      schema_version: '1',
      imported_from: null,
      created_at: '2026-07-16T00:00:00Z',
      updated_at: '2026-07-16T00:00:00Z',
    };

    it('loadProfileIntoSteering hydrates explicit strengths, budget, λ, title context', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.selectSAE(mockSAE));
      let ok = false;
      act(() => { ok = result.current.loadProfileIntoSteering(mockProfile as any); });
      expect(ok).toBe(true);
      const feats = result.current.selectedFeatures;
      // Explicit tuned strengths — NOT auto-baselines (0.2 freq would give ~2.4)
      expect(feats.map((f) => f.strength)).toEqual([1.4, -0.6]);
      expect(feats.every((f) => f.strengthSource === 'manual')).toBe(true);
      expect(feats[0].pinned).toBe(true);
      expect(result.current.clusterBudget?.B).toBe(2.0);
      expect(result.current.intensity).toBe(1.5);
      // Profiles are blended artifacts — loading one enters Blended mode
      expect(result.current.combinedMode).toBe(true);
      // Label tier 1: the authored profile NAME titles blended results
      expect(result.current.clusterContext?.display_token).toBe('Fear response');
      expect(result.current.activeProfile).toEqual({ id: 'clp_1', name: 'Fear response' });
      // Tile meta (2026-07-18): every member carries its source profile name
      expect(feats.every((f) => f.sourceProfileName === 'Fear response')).toBe(true);
    });

    it('loadProfileIntoSteering binds UNBOUND profiles at load when they fit (2026-07-17 fix: refusal was a dead end)', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.selectSAE(mockSAE));
      let ok = false;
      act(() => { ok = result.current.loadProfileIntoSteering({ ...mockProfile, sae_id: null } as any); });
      expect(ok).toBe(true);
      expect(result.current.selectedFeatures.length).toBeGreaterThan(0);
      expect(result.current.combinedMode).toBe(true);
    });

    it('loadProfileIntoSteering still refuses SAE-mismatched and oversized-space profiles', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.selectSAE(mockSAE));
      let ok = true;
      act(() => { ok = result.current.loadProfileIntoSteering({ ...mockProfile, sae_id: 'sae-other' } as any); });
      expect(ok).toBe(false);
      // unbound but a member exceeds the SAE's feature space -> cannot bind
      const oversized = {
        ...mockProfile,
        sae_id: null,
        members: [{ feature_idx: 10_000_000, strength: 0.2, sign: 1 }],
      };
      act(() => { ok = result.current.loadProfileIntoSteering(oversized as any); });
      expect(ok).toBe(false);
      expect(result.current.selectedFeatures).toHaveLength(0);
    });

    it('requestClusterAllocation is a no-op while a profile is active (explicit strengths win)', async () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.selectSAE(mockSAE));
      act(() => { result.current.loadProfileIntoSteering(mockProfile as any); });
      vi.mocked(steeringApi.computeClusterAllocation).mockClear();
      await act(async () => { await result.current.requestClusterAllocation(0.9); });
      expect(steeringApi.computeClusterAllocation).not.toHaveBeenCalled();
      expect(result.current.selectedFeatures.map((f) => f.strength)).toEqual([1.4, -0.6]);
    });

    it('selection mutations clear the active profile (stale titles are dishonest)', () => {
      const { result } = renderHook(() => useSteeringStore());
      act(() => result.current.selectSAE(mockSAE));
      act(() => { result.current.loadProfileIntoSteering(mockProfile as any); });
      expect(result.current.activeProfile).not.toBeNull();
      act(() => { result.current.addFeature({ feature_idx: 999, layer: 6, strength: 1 }); });
      expect(result.current.activeProfile).toBeNull();
      expect(result.current.clusterContext).toBeNull();
    });

    // Parity with the backend reference implementation (rebalance() in
    // cluster_allocation_service.py) — shared vectors, shared expectations.
    it('rebalance parity: backend reference vectors', () => {
      const { result } = renderHook(() => useSteeringStore());
      // Vector set 1: [1.2 pinned-by-edit, 0.8, 0.8], w=1/3 each, B=2.4
      act(() => {
        result.current.addFeature({ feature_idx: 1, layer: 6, strength: 0.8 });
        result.current.addFeature({ feature_idx: 2, layer: 6, strength: 0.8 });
        result.current.addFeature({ feature_idx: 3, layer: 6, strength: 0.8 });
      });
      const ids = result.current.selectedFeatures.map((f) => f.instance_id);
      act(() => {
        useSteeringStore.setState({
          clusterBudget: {
            B: 2.4, B_dir: 2.4, G: 1.0, flags: [], approximate: false,
            weightsByInstance: { [ids[0]]: 1 / 3, [ids[1]]: 1 / 3, [ids[2]]: 1 / 3 },
          },
        });
      });
      act(() => result.current.rebalanceStrength(ids[0], 1.2));
      // Backend: out=[1.2, 0.6, 0.6], Σ=2.4
      const out = result.current.selectedFeatures.map((f) => f.strength);
      expect(out[0]).toBe(1.2);
      expect(out[1]).toBeCloseTo(0.6, 1);
      expect(out[2]).toBeCloseTo(0.6, 1);
      expect(out.reduce((s, v) => s + Math.abs(v), 0)).toBeCloseTo(2.4, 1);
    });
  });

  describe('loadCircuitIntoSteering (Feature 015)', () => {
    // A promoted 2-layer circuit with a per-layer SAE for each of L13 and L14.
    const twoLayerCircuit = {
      id: 'circ-1',
      name: 'induction head',
      granularity: 'feature' as const,
      layers: [13, 14],
      member_count: 2,
      edge_count: 1,
      rung: 2,
      rung_language: 'validated',
      rung_next_step: '',
      promoted: true,
      model_id: 'model-456',
      version: 1,
      updated_at: '2024-01-01T00:00:00Z',
      narrative: null,
      saes: [
        { mistudio_sae_id: 'sae-l13', layer: 13 },
        { mistudio_sae_id: 'sae-l14', layer: 14 },
      ],
      members: [
        { layer: 13, member_kind: 'feature_ref' as const,
          feature: { feature_idx: 100, label: 'prev-token', strength: 1.2 } },
        { layer: 14, member_kind: 'feature_ref' as const,
          feature: { feature_idx: 200, label: 'copy', strength: 2.0 } },
      ],
      edges: [],
      budget: null,
      faithfulness: null,
      discovery: null,
      created_at: '2024-01-01T00:00:00Z',
    } as any;

    const primarySAE: SAE = { ...mockSAE, id: 'sae-l13', layer: 13 };

    it('builds one SelectedFeature per layer with the correct layer + per-layer sae_id, sets circuit_id, and calls requestClusterAllocation', async () => {
      const { result } = renderHook(() => useSteeringStore());

      // Multi-layer allocation shape (Feature 015): keyed by layer.
      vi.mocked(steeringApi.computeClusterAllocation).mockResolvedValue({
        formula_id: 'freq-budget/sim-alloc/per-layer@1',
        layers: {
          13: { B: 2.4, B_dir: 2.4, G: 1, weights: [1], strengths: [1.2],
                flags: [], constants_used: {}, approximate: false },
          14: { B: 3.0, B_dir: 3.0, G: 1, weights: [1], strengths: [2.0],
                flags: [], constants_used: {}, approximate: false },
        },
        hazards: [],
        strengths: [1.2, 2.0],
      } as any);

      let ok = false;
      await act(async () => {
        ok = result.current.loadCircuitIntoSteering(twoLayerCircuit, primarySAE);
        await new Promise((r) => setTimeout(r, 0));
      });

      expect(ok).toBe(true);
      const feats = result.current.selectedFeatures;
      expect(feats).toHaveLength(2);

      const l13 = feats.find((f) => f.feature_idx === 100);
      const l14 = feats.find((f) => f.feature_idx === 200);
      expect(l13?.layer).toBe(13);
      expect(l13?.sae_id).toBe('sae-l13');
      expect(l14?.layer).toBe(14);
      expect(l14?.sae_id).toBe('sae-l14');

      // Panel SAE + circuit provenance
      expect(result.current.selectedSAE?.id).toBe('sae-l13');
      expect(result.current.clusterContext?.circuit_id).toBe('circ-1');
      expect(result.current.clusterContext?.display_token).toBe('induction head');
      expect(result.current.combinedMode).toBe(true);
      // Not an authored profile — allocation must be allowed to run.
      expect(result.current.activeProfile).toBeNull();

      // requestClusterAllocation fired and threaded circuit_id.
      expect(steeringApi.computeClusterAllocation).toHaveBeenCalledTimes(1);
      const req = vi.mocked(steeringApi.computeClusterAllocation).mock.calls[0][0] as any;
      expect(req.circuit_id).toBe('circ-1');
      expect(req.members).toHaveLength(2);

      // Hydration guard is released.
      expect(result.current.isHydratingProfile).toBe(false);
    });

    it('expands cluster-ref members and skips those without expanded_members', () => {
      const { result } = renderHook(() => useSteeringStore());
      const circuit = {
        ...twoLayerCircuit,
        members: [
          { layer: 13, member_kind: 'cluster_ref' as const, cluster_name: 'A',
            expanded_members: [
              { feature_idx: 11, label: 'a1', strength: 0.5 },
              { feature_idx: 12, label: 'a2', strength: 0.7 },
            ] },
          { layer: 14, member_kind: 'cluster_ref' as const, cluster_name: 'B',
            expanded_members: null },
        ],
      } as any;
      vi.mocked(steeringApi.computeClusterAllocation).mockResolvedValue({
        formula_id: 'x', layers: {}, hazards: [], strengths: [],
      } as any);

      let ok = false;
      act(() => { ok = result.current.loadCircuitIntoSteering(circuit, primarySAE); });
      expect(ok).toBe(true);
      const feats = result.current.selectedFeatures;
      // 2 from the expanded cluster; the unexpanded cluster-ref is skipped.
      expect(feats.map((f) => f.feature_idx).sort()).toEqual([11, 12]);
      expect(feats.every((f) => f.layer === 13 && f.sae_id === 'sae-l13')).toBe(true);
    });

    it('returns false when the circuit has no loadable feature members', () => {
      const { result } = renderHook(() => useSteeringStore());
      const empty = { ...twoLayerCircuit, members: [] } as any;
      let ok = true;
      act(() => { ok = result.current.loadCircuitIntoSteering(empty, primarySAE); });
      expect(ok).toBe(false);
      expect(result.current.selectedFeatures).toHaveLength(0);
    });
  });
});

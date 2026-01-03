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
import { useSteeringStore } from './steeringStore';
import { DEFAULT_GENERATION_PARAMS } from '../types/steering';
import type { SAE } from '../types/sae';
import type { SteeringComparisonResponse, StrengthSweepResponse } from '../types/steering';

// Mock the API module
vi.mock('../api/steering', () => ({
  submitAsyncComparison: vi.fn(),
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

    it('should enforce maximum of 4 features', () => {
      const { result } = renderHook(() => useSteeringStore());

      act(() => {
        result.current.addFeature({ feature_idx: 100, layer: 6, strength: 1.5 });
        result.current.addFeature({ feature_idx: 200, layer: 6, strength: 1.0 });
        result.current.addFeature({ feature_idx: 300, layer: 6, strength: 0.5 });
        result.current.addFeature({ feature_idx: 400, layer: 6, strength: 2.0 });
      });

      expect(result.current.selectedFeatures).toHaveLength(4);

      // Try to add a 5th feature - should be ignored
      act(() => {
        result.current.addFeature({ feature_idx: 500, layer: 6, strength: 1.0 });
      });

      expect(result.current.selectedFeatures).toHaveLength(4);
      // Should not contain feature 500
      expect(result.current.selectedFeatures.find((f) => f.feature_idx === 500)).toBeUndefined();
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
});

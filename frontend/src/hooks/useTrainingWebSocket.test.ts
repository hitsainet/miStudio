/**
 * Unit tests for useTrainingWebSocket hook.
 *
 * Tests the hook lifecycle, subscription logic, and store updates for training WebSocket events.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import { useTrainingWebSocket } from './useTrainingWebSocket';
import { TrainingStatus } from '../types/training';

// Mock WebSocketContext
const mockOn = vi.fn();
const mockOff = vi.fn();
const mockSubscribe = vi.fn();
const mockUnsubscribe = vi.fn();
let mockIsConnected = true; // Default to connected

vi.mock('../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({
    on: mockOn,
    off: mockOff,
    subscribe: mockSubscribe,
    unsubscribe: mockUnsubscribe,
    isConnected: mockIsConnected,
  }),
}));

// Mock trainingsStore
const mockUpdateTrainingStatus = vi.fn();

vi.mock('../stores/trainingsStore', () => ({
  useTrainingsStore: () => ({
    updateTrainingStatus: mockUpdateTrainingStatus,
  }),
}));

describe('useTrainingWebSocket', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsConnected = true; // Reset to connected state
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  describe('Event Handler Registration', () => {
    it('should register all event handlers on mount', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Should register 5 event handlers
      expect(mockOn).toHaveBeenCalledTimes(5);
      expect(mockOn).toHaveBeenCalledWith('training:progress', expect.any(Function));
      expect(mockOn).toHaveBeenCalledWith('training:status_changed', expect.any(Function));
      expect(mockOn).toHaveBeenCalledWith('training:completed', expect.any(Function));
      expect(mockOn).toHaveBeenCalledWith('training:failed', expect.any(Function));
      expect(mockOn).toHaveBeenCalledWith('checkpoint:created', expect.any(Function));
    });

    it('should register event handlers only once', () => {
      const { rerender } = renderHook(() => useTrainingWebSocket([]));

      // Initial render
      expect(mockOn).toHaveBeenCalledTimes(5);

      // Clear mocks and rerender
      vi.clearAllMocks();
      rerender();

      // Should not register again
      expect(mockOn).not.toHaveBeenCalled();
    });

    it('should cleanup event handlers on unmount', () => {
      const { unmount } = renderHook(() => useTrainingWebSocket([]));

      // Clear registration calls
      vi.clearAllMocks();

      unmount();

      // Should unregister 5 event handlers
      expect(mockOff).toHaveBeenCalledTimes(5);
      expect(mockOff).toHaveBeenCalledWith('training:progress', expect.any(Function));
      expect(mockOff).toHaveBeenCalledWith('training:status_changed', expect.any(Function));
      expect(mockOff).toHaveBeenCalledWith('training:completed', expect.any(Function));
      expect(mockOff).toHaveBeenCalledWith('training:failed', expect.any(Function));
      expect(mockOff).toHaveBeenCalledWith('checkpoint:created', expect.any(Function));
    });
  });

  describe('Channel Subscription', () => {
    it('should subscribe to progress and checkpoint channels for each training ID', () => {
      renderHook(() => useTrainingWebSocket(['training-1', 'training-2']));

      // Should subscribe to 4 channels (2 trainings × 2 channels each)
      expect(mockSubscribe).toHaveBeenCalledTimes(4);
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-1/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-1/checkpoints');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-2/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-2/checkpoints');
    });

    it('should not subscribe when trainingIds is empty', () => {
      renderHook(() => useTrainingWebSocket([]));

      expect(mockSubscribe).not.toHaveBeenCalled();
    });

    it('should not subscribe when not connected', () => {
      // Set not connected state
      mockIsConnected = false;

      renderHook(() => useTrainingWebSocket(['training-1']));

      expect(mockSubscribe).not.toHaveBeenCalled();
    });

    it('should unsubscribe from all channels on unmount', () => {
      const { unmount } = renderHook(() =>
        useTrainingWebSocket(['training-1', 'training-2'])
      );

      // Clear subscription calls
      vi.clearAllMocks();

      unmount();

      // Should unsubscribe from 4 channels
      expect(mockUnsubscribe).toHaveBeenCalledTimes(4);
      expect(mockUnsubscribe).toHaveBeenCalledWith('trainings/training-1/progress');
      expect(mockUnsubscribe).toHaveBeenCalledWith('trainings/training-1/checkpoints');
      expect(mockUnsubscribe).toHaveBeenCalledWith('trainings/training-2/progress');
      expect(mockUnsubscribe).toHaveBeenCalledWith('trainings/training-2/checkpoints');
    });

    it('should resubscribe when trainingIds change', () => {
      const { rerender } = renderHook(({ ids }) => useTrainingWebSocket(ids), {
        initialProps: { ids: ['training-1'] },
      });

      // Initial subscription (2 channels)
      expect(mockSubscribe).toHaveBeenCalledTimes(2);
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-1/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-1/checkpoints');

      // Clear mocks
      vi.clearAllMocks();

      // Rerender with different training IDs
      rerender({ ids: ['training-2', 'training-3'] });

      // Should unsubscribe from old channels (2 channels)
      expect(mockUnsubscribe).toHaveBeenCalledTimes(2);
      expect(mockUnsubscribe).toHaveBeenCalledWith('trainings/training-1/progress');
      expect(mockUnsubscribe).toHaveBeenCalledWith('trainings/training-1/checkpoints');

      // Should subscribe to new channels (4 channels)
      expect(mockSubscribe).toHaveBeenCalledTimes(4);
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-2/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-2/checkpoints');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-3/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-3/checkpoints');
    });
  });

  describe('Event Handler: training:progress', () => {
    it('should update training status with progress data', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Get the progress handler
      const progressHandler = mockOn.mock.calls.find(
        (call) => call[0] === 'training:progress'
      )?.[1];

      expect(progressHandler).toBeDefined();

      // Simulate progress event
      const progressData = {
        training_id: 'training-1',
        current_step: 500,
        progress: 50.0,
        loss: 0.25,
        l0_sparsity: 15.0,
        dead_neurons: 2,
        learning_rate: 0.0001,
      };

      progressHandler(progressData);

      // Should call updateTrainingStatus with correct data
      expect(mockUpdateTrainingStatus).toHaveBeenCalledTimes(1);
      expect(mockUpdateTrainingStatus).toHaveBeenCalledWith('training-1', {
        current_step: 500,
        progress: 50.0,
        current_loss: 0.25,
        current_l0_sparsity: 15.0,
        current_dead_neurons: 2,
        current_learning_rate: 0.0001,
      });
    });
  });

  describe('Event Handler: training:status_changed', () => {
    it('should update training status with new status', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Get the status_changed handler
      const statusHandler = mockOn.mock.calls.find(
        (call) => call[0] === 'training:status_changed'
      )?.[1];

      expect(statusHandler).toBeDefined();

      // Simulate status changed event
      const statusData = {
        training_id: 'training-1',
        status: TrainingStatus.PAUSED,
        current_step: 500,
        progress: 50.0,
      };

      statusHandler(statusData);

      // Should call updateTrainingStatus with correct data
      expect(mockUpdateTrainingStatus).toHaveBeenCalledTimes(1);
      expect(mockUpdateTrainingStatus).toHaveBeenCalledWith('training-1', {
        status: TrainingStatus.PAUSED,
        current_step: 500,
        progress: 50.0,
      });
    });
  });

  describe('Event Handler: training:completed', () => {
    it('should update training status to COMPLETED with 100% progress', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Get the completed handler
      const completedHandler = mockOn.mock.calls.find(
        (call) => call[0] === 'training:completed'
      )?.[1];

      expect(completedHandler).toBeDefined();

      // Simulate completed event
      const completedData = {
        training_id: 'training-1',
      };

      completedHandler(completedData);

      // Should call updateTrainingStatus with COMPLETED status
      expect(mockUpdateTrainingStatus).toHaveBeenCalledTimes(1);
      expect(mockUpdateTrainingStatus).toHaveBeenCalledWith('training-1', {
        status: TrainingStatus.COMPLETED,
        progress: 100.0,
        completed_at: expect.any(String),
      });

      // Verify completed_at is a valid ISO string
      const updateCall = mockUpdateTrainingStatus.mock.calls[0][1];
      expect(new Date(updateCall.completed_at).toISOString()).toBe(updateCall.completed_at);
    });
  });

  describe('Event Handler: training:failed', () => {
    it('should update training status to FAILED with error message', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Get the failed handler
      const failedHandler = mockOn.mock.calls.find(
        (call) => call[0] === 'training:failed'
      )?.[1];

      expect(failedHandler).toBeDefined();

      // Simulate failed event with error_message
      const failedData = {
        training_id: 'training-1',
        error_message: 'CUDA out of memory',
      };

      failedHandler(failedData);

      // Should call updateTrainingStatus with FAILED status
      expect(mockUpdateTrainingStatus).toHaveBeenCalledTimes(1);
      expect(mockUpdateTrainingStatus).toHaveBeenCalledWith('training-1', {
        status: TrainingStatus.FAILED,
        error_message: 'CUDA out of memory',
      });
    });

    it('should fallback to message field if error_message not present', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Get the failed handler
      const failedHandler = mockOn.mock.calls.find(
        (call) => call[0] === 'training:failed'
      )?.[1];

      expect(failedHandler).toBeDefined();

      // Simulate failed event with message instead of error_message
      const failedData = {
        training_id: 'training-1',
        message: 'Training job terminated',
      };

      failedHandler(failedData);

      // Should call updateTrainingStatus with FAILED status and message
      expect(mockUpdateTrainingStatus).toHaveBeenCalledTimes(1);
      expect(mockUpdateTrainingStatus).toHaveBeenCalledWith('training-1', {
        status: TrainingStatus.FAILED,
        error_message: 'Training job terminated',
      });
    });
  });

  describe('Event Handler: checkpoint:created', () => {
    it('should handle checkpoint created event without error', () => {
      renderHook(() => useTrainingWebSocket([]));

      // Get the checkpoint handler
      const checkpointHandler = mockOn.mock.calls.find(
        (call) => call[0] === 'checkpoint:created'
      )?.[1];

      expect(checkpointHandler).toBeDefined();

      // Simulate checkpoint created event
      const checkpointData = {
        training_id: 'training-1',
        checkpoint_id: 'checkpoint-1',
        step: 1000,
        loss: 0.15,
      };

      // Should not throw error
      expect(() => checkpointHandler(checkpointData)).not.toThrow();

      // Currently doesn't update store (TODO in code)
      expect(mockUpdateTrainingStatus).not.toHaveBeenCalled();
    });
  });

  describe('Subscription Memoization', () => {
    it('should not resubscribe when trainingIds array changes reference but has same IDs', () => {
      const { rerender } = renderHook(({ ids }) => useTrainingWebSocket(ids), {
        initialProps: { ids: ['training-1', 'training-2'] },
      });

      // Initial subscription (4 channels)
      expect(mockSubscribe).toHaveBeenCalledTimes(4);

      // Clear mocks
      vi.clearAllMocks();

      // Rerender with new array reference but same IDs
      rerender({ ids: ['training-1', 'training-2'] });

      // Should not resubscribe (memoization working)
      expect(mockUnsubscribe).not.toHaveBeenCalled();
      expect(mockSubscribe).not.toHaveBeenCalled();
    });

    it('should handle trainingIds in different order without resubscribing', () => {
      const { rerender } = renderHook(({ ids }) => useTrainingWebSocket(ids), {
        initialProps: { ids: ['training-1', 'training-2'] },
      });

      // Initial subscription (4 channels)
      expect(mockSubscribe).toHaveBeenCalledTimes(4);

      // Clear mocks
      vi.clearAllMocks();

      // Rerender with same IDs but different order
      rerender({ ids: ['training-2', 'training-1'] });

      // Should not resubscribe (sorted for stable key)
      expect(mockUnsubscribe).not.toHaveBeenCalled();
      expect(mockSubscribe).not.toHaveBeenCalled();
    });
  });

  describe('Multiple Training IDs', () => {
    it('should handle multiple training IDs correctly', () => {
      renderHook(() =>
        useTrainingWebSocket(['training-1', 'training-2', 'training-3'])
      );

      // Should subscribe to 6 channels (3 trainings × 2 channels each)
      expect(mockSubscribe).toHaveBeenCalledTimes(6);
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-1/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-1/checkpoints');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-2/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-2/checkpoints');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-3/progress');
      expect(mockSubscribe).toHaveBeenCalledWith('trainings/training-3/checkpoints');
    });
  });
});

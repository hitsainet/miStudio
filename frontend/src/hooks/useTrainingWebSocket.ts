/**
 * Training WebSocket Hook
 *
 * React hook for subscribing to training progress and checkpoint events via WebSocket.
 * Uses WebSocketContext for proper connection management and event queuing.
 *
 * WebSocket Channels:
 * - trainings/{training_id}/progress - Progress updates
 * - trainings/{training_id}/checkpoints - Checkpoint creation
 *
 * Events:
 * - training:progress - Progress update (every 100 steps)
 * - training:status_changed - Status changed (pause/resume/stop)
 * - training:completed - Training completed
 * - training:failed - Training failed
 * - checkpoint:created - Checkpoint saved
 *
 * Usage:
 *   useTrainingWebSocket(trainingIds);
 *
 *   // Automatically subscribes/unsubscribes based on trainingIds array
 *   // Updates are handled by trainingsStore.updateTrainingStatus()
 */

import { useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useTrainingsStore } from '../stores/trainingsStore';
import { TrainingStatus } from '../types/training';
import type {
  TrainingProgressEvent,
  CheckpointCreatedEvent,
} from '../types/training';

export const useTrainingWebSocket = (trainingIds: string[]) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const updateTrainingStatus = useTrainingsStore((state) => state.updateTrainingStatus);

  // Use refs to store handler references for cleanup
  const handlersRef = useRef<{
    progress: ((data: TrainingProgressEvent) => void) | null;
    statusChanged: ((data: any) => void) | null;
    completed: ((data: any) => void) | null;
    failed: ((data: any) => void) | null;
    checkpoint: ((data: CheckpointCreatedEvent) => void) | null;
  }>({
    progress: null,
    statusChanged: null,
    completed: null,
    failed: null,
    checkpoint: null,
  });

  // Create stable handler using useCallback - this ensures the same function reference
  // is used across re-renders, which is critical for proper cleanup
  const handleProgress = useCallback((data: TrainingProgressEvent) => {
    console.log('[Training WS] 📊 Progress event received:', data);
    updateTrainingStatus(data.training_id, {
      current_step: data.current_step,
      progress: data.progress,
      current_loss: data.loss,
      current_l0_sparsity: data.l0_sparsity,
      current_dead_neurons: data.dead_neurons,
      current_learning_rate: data.learning_rate,
    });
  }, [updateTrainingStatus]);

  const handleStatusChanged = useCallback((data: any) => {
    console.log('[Training WS] 🔄 Status changed:', data);
    updateTrainingStatus(data.training_id, {
      status: data.status,
      current_step: data.current_step,
      progress: data.progress,
    });
  }, [updateTrainingStatus]);

  const handleCompleted = useCallback((data: any) => {
    console.log('[Training WS] ✅ Training completed:', data);
    updateTrainingStatus(data.training_id, {
      status: TrainingStatus.COMPLETED,
      progress: 100.0,
      completed_at: new Date().toISOString(),
    });
  }, [updateTrainingStatus]);

  const handleFailed = useCallback((data: any) => {
    console.log('[Training WS] ❌ Training failed:', data);
    updateTrainingStatus(data.training_id, {
      status: TrainingStatus.FAILED,
      error_message: data.error_message || data.message,
    });
  }, [updateTrainingStatus]);

  const handleCheckpointCreated = useCallback((data: CheckpointCreatedEvent) => {
    console.log('[Training WS] 💾 Checkpoint created:', data);
    // TODO: Update checkpoints in store when we add checkpoint state management
  }, []);

  // Register event handlers
  useEffect(() => {
    console.log('[Training WS] Setting up event handlers');

    // Store refs for cleanup
    handlersRef.current = {
      progress: handleProgress,
      statusChanged: handleStatusChanged,
      completed: handleCompleted,
      failed: handleFailed,
      checkpoint: handleCheckpointCreated,
    };

    // Register event handlers
    on('training:progress', handleProgress);
    on('training:status_changed', handleStatusChanged);
    on('training:completed', handleCompleted);
    on('training:failed', handleFailed);
    on('checkpoint:created', handleCheckpointCreated);

    console.log('[Training WS] ✓ Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Training WS] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('training:progress', handlers.progress);
      if (handlers.statusChanged) off('training:status_changed', handlers.statusChanged);
      if (handlers.completed) off('training:completed', handlers.completed);
      if (handlers.failed) off('training:failed', handlers.failed);
      if (handlers.checkpoint) off('checkpoint:created', handlers.checkpoint);

      // Clear refs
      handlersRef.current = {
        progress: null,
        statusChanged: null,
        completed: null,
        failed: null,
        checkpoint: null,
      };
    };
  }, [on, off, handleProgress, handleStatusChanged, handleCompleted, handleFailed, handleCheckpointCreated]);

  // Subscribe to channels for trainings
  useEffect(() => {
    if (!isConnected) {
      console.log('[Training WS] ⚠️ Not connected, skipping channel subscriptions');
      return;
    }

    if (trainingIds.length === 0) {
      console.log('[Training WS] No trainings to subscribe to');
      return;
    }

    console.log('[Training WS] 📡 Subscribing to', trainingIds.length, 'training channel(s)');

    // Subscribe to progress and checkpoint channels for each training
    const subscribedChannels: string[] = [];
    trainingIds.forEach((trainingId) => {
      const progressChannel = `trainings/${trainingId}/progress`;
      const checkpointChannel = `trainings/${trainingId}/checkpoints`;

      console.log(`[Training WS] → Subscribing to ${progressChannel}`);
      subscribe(progressChannel);
      subscribedChannels.push(progressChannel);

      console.log(`[Training WS] → Subscribing to ${checkpointChannel}`);
      subscribe(checkpointChannel);
      subscribedChannels.push(checkpointChannel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[Training WS] 📡 Unsubscribing from training channels');
      subscribedChannels.forEach((channel) => {
        console.log(`[Training WS] ← Unsubscribing from ${channel}`);
        unsubscribe(channel);
      });
    };
  }, [trainingIds.join(','), isConnected, subscribe, unsubscribe]);
};

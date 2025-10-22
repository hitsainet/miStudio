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

import { useEffect, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useTrainingsStore } from '../stores/trainingsStore';
import { TrainingStatus } from '../types/training';
import type {
  TrainingProgressEvent,
  CheckpointCreatedEvent,
} from '../types/training';

export const useTrainingWebSocket = (trainingIds: string[]) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateTrainingStatus } = useTrainingsStore();
  const handlersRegisteredRef = useRef(false);

  // Set up global event handlers (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[Training WS] Setting up training event handlers');

    // Handler for 'training:progress' events
    const handleProgress = (data: TrainingProgressEvent) => {
      console.log('[Training WS] Progress event:', data);
      updateTrainingStatus(data.training_id, {
        current_step: data.current_step,
        progress: data.progress,
        current_loss: data.loss,
        current_l0_sparsity: data.l0_sparsity,
        current_dead_neurons: data.dead_neurons,
        current_learning_rate: data.learning_rate,
      });
    };

    // Handler for 'training:status_changed' events
    const handleStatusChanged = (data: any) => {
      console.log('[Training WS] Status changed:', data);
      updateTrainingStatus(data.training_id, {
        status: data.status,
        current_step: data.current_step,
        progress: data.progress,
      });
    };

    // Handler for 'training:completed' events
    const handleCompleted = (data: any) => {
      console.log('[Training WS] Training completed:', data);
      updateTrainingStatus(data.training_id, {
        status: TrainingStatus.COMPLETED,
        progress: 100.0,
        completed_at: new Date().toISOString(),
      });
    };

    // Handler for 'training:failed' events
    const handleFailed = (data: any) => {
      console.log('[Training WS] Training failed:', data);
      updateTrainingStatus(data.training_id, {
        status: TrainingStatus.FAILED,
        error_message: data.error_message || data.message,
      });
    };

    // Handler for 'checkpoint:created' events
    const handleCheckpointCreated = (data: CheckpointCreatedEvent) => {
      console.log('[Training WS] Checkpoint created:', data);
      // TODO: Update checkpoints in store when we add checkpoint state management
    };

    // Register event handlers
    on('training:progress', handleProgress);
    on('training:status_changed', handleStatusChanged);
    on('training:completed', handleCompleted);
    on('training:failed', handleFailed);
    on('checkpoint:created', handleCheckpointCreated);

    handlersRegisteredRef.current = true;
    console.log('[Training WS] Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Training WS] Cleaning up event handlers');
      off('training:progress', handleProgress);
      off('training:status_changed', handleStatusChanged);
      off('training:completed', handleCompleted);
      off('training:failed', handleFailed);
      off('checkpoint:created', handleCheckpointCreated);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, updateTrainingStatus]);

  // Subscribe to channels for active trainings
  useEffect(() => {
    if (!isConnected) {
      console.log('[Training WS] Not connected, skipping channel subscriptions');
      return;
    }

    if (trainingIds.length === 0) {
      console.log('[Training WS] No trainings to subscribe to');
      return;
    }

    console.log('[Training WS] Subscribing to', trainingIds.length, 'training channels');

    // Subscribe to progress and checkpoint channels for each training
    trainingIds.forEach((trainingId) => {
      const progressChannel = `trainings/${trainingId}/progress`;
      const checkpointChannel = `trainings/${trainingId}/checkpoints`;

      console.log(`[Training WS] Subscribing to ${progressChannel}`);
      subscribe(progressChannel);

      console.log(`[Training WS] Subscribing to ${checkpointChannel}`);
      subscribe(checkpointChannel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[Training WS] Unsubscribing from training channels');
      trainingIds.forEach((trainingId) => {
        unsubscribe(`trainings/${trainingId}/progress`);
        unsubscribe(`trainings/${trainingId}/checkpoints`);
      });
    };
  }, [trainingIds, isConnected, subscribe, unsubscribe]);
};

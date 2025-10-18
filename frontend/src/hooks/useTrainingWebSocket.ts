/**
 * Training WebSocket Hook
 *
 * React hook for subscribing to training progress and checkpoint events via WebSocket.
 *
 * WebSocket Channels:
 * - trainings/{training_id}/progress - Progress updates
 * - trainings/{training_id}/checkpoints - Checkpoint creation
 *
 * Events:
 * - training:created - New training job created
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

import { useEffect } from 'react';
import { websocketClient } from '../api/websocket';
import { useTrainingsStore } from '../stores/trainingsStore';
import type {
  TrainingProgressEvent,
  CheckpointCreatedEvent,
} from '../types/training';

export const useTrainingWebSocket = (trainingIds: string[]) => {
  const { updateTrainingStatus } = useTrainingsStore();

  useEffect(() => {
    // Connect WebSocket
    const socket = websocketClient.connect();

    if (!socket.connected) {
      console.warn('WebSocket not connected, training updates will not be real-time');
      return;
    }

    // Subscribe to each training's progress updates
    const progressHandlers = trainingIds.map((trainingId) => {
      const handler = (event: { event: string; data: any }) => {
        console.log(`[WS] Training ${trainingId} event:`, event.event, event.data);

        switch (event.event) {
          case 'created':
            // New training created - could refresh list
            console.log('Training created:', event.data);
            break;

          case 'progress':
            // Progress update
            const progressData = event.data as TrainingProgressEvent;
            updateTrainingStatus(progressData.training_id, {
              current_step: progressData.current_step,
              progress: progressData.progress,
              current_loss: progressData.loss,
              current_l0_sparsity: progressData.l0_sparsity,
              current_dead_neurons: progressData.dead_neurons,
              current_learning_rate: progressData.learning_rate,
            });
            break;

          case 'status_changed':
            // Status changed (pause/resume/stop)
            updateTrainingStatus(event.data.training_id, {
              status: event.data.status,
              current_step: event.data.current_step,
              progress: event.data.progress,
            });
            break;

          case 'completed':
            // Training completed
            updateTrainingStatus(event.data.training_id, {
              status: 'completed',
              progress: 100.0,
              completed_at: new Date().toISOString(),
            });
            break;

          case 'failed':
            // Training failed
            updateTrainingStatus(event.data.training_id, {
              status: 'failed',
              error_message: event.data.error_message,
            });
            break;

          default:
            console.warn('Unknown training event:', event.event);
        }
      };

      websocketClient.subscribeToTrainingProgress(trainingId, handler);
      return { trainingId, handler };
    });

    // Subscribe to checkpoint events for each training
    const checkpointHandlers = trainingIds.map((trainingId) => {
      const handler = (event: { event: string; data: CheckpointCreatedEvent }) => {
        console.log(`[WS] Checkpoint event for ${trainingId}:`, event.event, event.data);

        if (event.event === 'checkpoint_created') {
          // Checkpoint created - could update checkpoint list
          console.log('Checkpoint created:', event.data);
          // TODO: Update checkpoints in store when we add checkpoint state management
        }
      };

      websocketClient.subscribeToTrainingCheckpoints(trainingId, handler);
      return { trainingId, handler };
    });

    // Cleanup on unmount or when trainingIds change
    return () => {
      progressHandlers.forEach(({ trainingId }) => {
        websocketClient.unsubscribeFromTrainingProgress(trainingId);
      });

      checkpointHandlers.forEach(({ trainingId }) => {
        websocketClient.unsubscribeFromTrainingCheckpoints(trainingId);
      });
    };
  }, [trainingIds, updateTrainingStatus]);
};

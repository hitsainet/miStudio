/**
 * Deletion Progress WebSocket Hook
 *
 * React hook for subscribing to training deletion progress events via WebSocket.
 * Tracks real-time progress as deletion sub-tasks complete.
 *
 * WebSocket Channel:
 * - trainings/{training_id}/deletion - Deletion progress updates
 *
 * Events:
 * - task_update - Deletion task completed (extractions, checkpoints, metrics, features, database, files)
 *
 * Usage:
 *   useDeletionProgressWebSocket(trainingId, onTaskUpdate);
 *
 *   // Automatically subscribes/unsubscribes based on trainingId
 *   // Calls onTaskUpdate callback when tasks complete
 */

import { useEffect, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';

interface DeletionTaskUpdate {
  training_id: string;
  task: string;
  status: 'in_progress' | 'completed';
  message?: string;
  count?: number;
}

type TaskUpdateCallback = (update: DeletionTaskUpdate) => void;

export const useDeletionProgressWebSocket = (
  trainingId: string | null,
  onTaskUpdate: TaskUpdateCallback
) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const handlersRegisteredRef = useRef(false);
  const callbackRef = useRef(onTaskUpdate);

  // Keep callback ref up to date
  useEffect(() => {
    callbackRef.current = onTaskUpdate;
  }, [onTaskUpdate]);

  // Set up global event handler (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[Deletion WS] Setting up deletion event handlers');

    // Handler for 'task_update' events
    const handleTaskUpdate = (data: DeletionTaskUpdate) => {
      console.log('[Deletion WS] ✅ Task update received:', {
        training_id: data.training_id,
        task: data.task,
        status: data.status,
        message: data.message,
        count: data.count,
      });
      callbackRef.current(data);
    };

    // Register event handler
    on('task_update', handleTaskUpdate);

    handlersRegisteredRef.current = true;
    console.log('[Deletion WS] Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Deletion WS] Cleaning up event handlers');
      off('task_update', handleTaskUpdate);
      handlersRegisteredRef.current = false;
    };
  }, [on, off]);

  // Subscribe to deletion channel
  useEffect(() => {
    if (!isConnected) {
      console.log('[Deletion WS] Not connected, skipping channel subscription');
      return;
    }

    if (!trainingId) {
      console.log('[Deletion WS] No training ID, skipping subscription');
      return;
    }

    const channel = `trainings/${trainingId}/deletion`;
    console.log(`[Deletion WS] Subscribing to ${channel}`);
    subscribe(channel);

    // Cleanup subscription
    return () => {
      console.log(`[Deletion WS] Unsubscribing from ${channel}`);
      unsubscribe(channel);
    };
  }, [trainingId, isConnected, subscribe, unsubscribe]);
};

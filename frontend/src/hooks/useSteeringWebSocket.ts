/**
 * Steering WebSocket Hook
 *
 * React hook for subscribing to steering task progress via WebSocket.
 * Used with the async Celery-based steering implementation.
 *
 * WebSocket Channel:
 * - steering/{task_id} - Progress updates for a specific steering task
 *
 * Events:
 * - steering:progress - Progress update (percent, message, current feature/strength)
 * - steering:completed - Task completed with result
 * - steering:failed - Task failed with error
 *
 * Usage:
 *   useSteeringWebSocket(taskId, {
 *     onProgress: (data) => console.log('Progress:', data.percent),
 *     onCompleted: (data) => console.log('Result:', data.result),
 *     onFailed: (data) => console.log('Error:', data.error),
 *   });
 */

import { useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import type { SteeringComparisonResponse } from '../types/steering';

/**
 * Progress event data from steering worker.
 */
export interface SteeringProgressEvent {
  task_id: string;
  percent: number;
  message: string;
  current_feature?: number;
  current_strength?: number;
}

/**
 * Completed event data from steering worker.
 */
export interface SteeringCompletedEvent {
  task_id: string;
  percent: number;
  message: string;
  result: SteeringComparisonResponse;
}

/**
 * Failed event data from steering worker.
 */
export interface SteeringFailedEvent {
  task_id: string;
  percent: number;
  message: string;
  error: string;
}

/**
 * Callbacks for steering WebSocket events.
 */
export interface SteeringWebSocketCallbacks {
  onProgress?: (data: SteeringProgressEvent) => void;
  onCompleted?: (data: SteeringCompletedEvent) => void;
  onFailed?: (data: SteeringFailedEvent) => void;
}

/**
 * Hook for subscribing to steering task progress via WebSocket.
 *
 * @param taskId - The Celery task ID to subscribe to (null to skip subscription)
 * @param callbacks - Event callbacks for progress, completed, and failed events
 */
export const useSteeringWebSocket = (
  taskId: string | null,
  callbacks: SteeringWebSocketCallbacks = {}
) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();

  // Use refs for handlers to ensure stable cleanup
  const handlersRef = useRef<{
    progress: ((data: SteeringProgressEvent) => void) | null;
    completed: ((data: SteeringCompletedEvent) => void) | null;
    failed: ((data: SteeringFailedEvent) => void) | null;
  }>({
    progress: null,
    completed: null,
    failed: null,
  });

  // Use ref to track current taskId for filtering events
  const taskIdRef = useRef<string | null>(taskId);
  taskIdRef.current = taskId;

  // Use ref for callbacks to avoid re-registering handlers
  const callbacksRef = useRef(callbacks);
  callbacksRef.current = callbacks;

  // Create stable handlers
  const handleProgress = useCallback((data: SteeringProgressEvent) => {
    // Filter by task_id to only handle events for our task
    if (data.task_id !== taskIdRef.current) return;

    console.log('[Steering WS] 📊 Progress event:', data);
    callbacksRef.current.onProgress?.(data);
  }, []);

  const handleCompleted = useCallback((data: SteeringCompletedEvent) => {
    if (data.task_id !== taskIdRef.current) return;

    console.log('[Steering WS] ✅ Completed event:', data);
    callbacksRef.current.onCompleted?.(data);
  }, []);

  const handleFailed = useCallback((data: SteeringFailedEvent) => {
    if (data.task_id !== taskIdRef.current) return;

    console.log('[Steering WS] ❌ Failed event:', data);
    callbacksRef.current.onFailed?.(data);
  }, []);

  // Register event handlers
  useEffect(() => {
    console.log('[Steering WS] Setting up event handlers');

    // Store refs for cleanup
    handlersRef.current = {
      progress: handleProgress,
      completed: handleCompleted,
      failed: handleFailed,
    };

    // Register event handlers
    on('steering:progress', handleProgress);
    on('steering:completed', handleCompleted);
    on('steering:failed', handleFailed);

    console.log('[Steering WS] ✓ Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Steering WS] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('steering:progress', handlers.progress);
      if (handlers.completed) off('steering:completed', handlers.completed);
      if (handlers.failed) off('steering:failed', handlers.failed);

      handlersRef.current = {
        progress: null,
        completed: null,
        failed: null,
      };
    };
  }, [on, off, handleProgress, handleCompleted, handleFailed]);

  // Subscribe to steering channel when we have a task ID
  useEffect(() => {
    if (!isConnected) {
      console.log('[Steering WS] ⚠️ Not connected, skipping channel subscription');
      return;
    }

    if (!taskId) {
      console.log('[Steering WS] No task ID, skipping subscription');
      return;
    }

    const channel = `steering/${taskId}`;
    console.log(`[Steering WS] 📡 Subscribing to ${channel}`);
    subscribe(channel);

    // Cleanup subscription
    return () => {
      console.log(`[Steering WS] 📡 Unsubscribing from ${channel}`);
      unsubscribe(channel);
    };
  }, [taskId, isConnected, subscribe, unsubscribe]);

  return { isConnected };
};

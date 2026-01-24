/**
 * Neuronpedia Push WebSocket Hook
 *
 * React hook for subscribing to Neuronpedia push progress events via WebSocket.
 * Uses WebSocketContext for proper connection management and event queuing.
 *
 * WebSocket Channels:
 * - neuronpedia/push/{push_job_id} - Progress updates for specific push job
 *
 * Events:
 * - neuronpedia:push_progress - Progress update (emitted during push)
 * - neuronpedia:push_completed - Push completed
 * - neuronpedia:push_failed - Push failed
 *
 * Usage:
 *   const { progress, isComplete, error } = useNeuronpediaPushWebSocket(pushJobId);
 */

import { useEffect, useCallback, useState, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';

export interface NeuronpediaPushProgress {
  push_job_id: string;
  sae_id: string;
  stage: string;
  progress: number;
  status: 'pending' | 'pushing' | 'completed' | 'failed';
  message?: string;
  features_pushed?: number;
  total_features?: number;
  activations_pushed?: number;
  explanations_pushed?: number;
  elapsed_seconds?: number;
  eta_seconds?: number;
  error?: string;
}

export interface NeuronpediaPushResult {
  progress: NeuronpediaPushProgress | null;
  isComplete: boolean;
  error: string | null;
  reset: () => void;
}

export const useNeuronpediaPushWebSocket = (
  pushJobId: string | null
): NeuronpediaPushResult => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();

  const [progress, setProgress] = useState<NeuronpediaPushProgress | null>(null);
  const [isComplete, setIsComplete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Use refs to store handler references for cleanup
  const handlersRef = useRef<{
    progress: ((data: NeuronpediaPushProgress) => void) | null;
    completed: ((data: NeuronpediaPushProgress) => void) | null;
    failed: ((data: NeuronpediaPushProgress) => void) | null;
  }>({
    progress: null,
    completed: null,
    failed: null,
  });

  // Reset function to clear state
  const reset = useCallback(() => {
    setProgress(null);
    setIsComplete(false);
    setError(null);
  }, []);

  // Create stable handlers using useCallback
  const handleProgress = useCallback((data: NeuronpediaPushProgress) => {
    console.log('[Neuronpedia Push WS] 📊 Progress event received:', data);
    setProgress(data);
  }, []);

  const handleCompleted = useCallback((data: NeuronpediaPushProgress) => {
    console.log('[Neuronpedia Push WS] ✅ Push completed:', data);
    setProgress(data);
    setIsComplete(true);
    setError(null);
  }, []);

  const handleFailed = useCallback((data: NeuronpediaPushProgress) => {
    console.log('[Neuronpedia Push WS] ❌ Push failed:', data);
    setProgress(data);
    setIsComplete(true);
    setError(data.error || data.message || 'Push failed');
  }, []);

  // Effect for event handlers
  useEffect(() => {
    if (!pushJobId) return;

    console.log('[Neuronpedia Push WS] Setting up event handlers');

    // Store handlers in ref
    handlersRef.current = {
      progress: handleProgress,
      completed: handleCompleted,
      failed: handleFailed,
    };

    // Register event listeners
    on('neuronpedia:push_progress', handleProgress);
    on('neuronpedia:push_completed', handleCompleted);
    on('neuronpedia:push_failed', handleFailed);

    console.log('[Neuronpedia Push WS] ✓ Event handlers registered');

    return () => {
      console.log('[Neuronpedia Push WS] Cleaning up event handlers');
      if (handlersRef.current.progress) {
        off('neuronpedia:push_progress', handlersRef.current.progress);
      }
      if (handlersRef.current.completed) {
        off('neuronpedia:push_completed', handlersRef.current.completed);
      }
      if (handlersRef.current.failed) {
        off('neuronpedia:push_failed', handlersRef.current.failed);
      }
    };
  }, [pushJobId, on, off, handleProgress, handleCompleted, handleFailed]);

  // Effect for channel subscription
  useEffect(() => {
    if (!pushJobId || !isConnected) {
      if (pushJobId && !isConnected) {
        console.log('[Neuronpedia Push WS] Not connected, skipping subscription');
      }
      return;
    }

    const channel = `neuronpedia/push/${pushJobId}`;
    console.log(`[Neuronpedia Push WS] 📡 Subscribing to channel: ${channel}`);
    subscribe(channel);

    return () => {
      console.log(`[Neuronpedia Push WS] 📡 Unsubscribing from channel: ${channel}`);
      unsubscribe(channel);
    };
  }, [pushJobId, isConnected, subscribe, unsubscribe]);

  return { progress, isComplete, error, reset };
};

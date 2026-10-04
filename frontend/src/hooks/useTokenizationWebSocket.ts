/**
 * Tokenization WebSocket Hook
 *
 * React hook for subscribing to tokenization progress events via WebSocket.
 * Uses WebSocketContext for proper connection management and event queuing.
 *
 * WebSocket Channels:
 * - datasets/{dataset_id}/tokenization/{tokenization_id} - Detailed progress updates
 *
 * Events:
 * - tokenization:progress - Detailed progress with stage, samples, filter stats
 * - tokenization:status - Status changes (error, cancelled, deleted, ready)
 *
 * Usage:
 *   useTokenizationWebSocket(tokenizationIds);
 *
 *   // Automatically subscribes/unsubscribes based on tokenizationIds array
 *   // Updates are handled by datasetsStore
 */

import { useEffect, useRef, useMemo } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useDatasetsStore } from '../stores/datasetsStore';
import type { DatasetTokenizationProgress } from '../types/dataset';

interface TokenizationStatusEvent {
  dataset_id: string;
  tokenization_id: string;
  status: string;
  error_message?: string;
  model_id?: string;
}

export const useTokenizationWebSocket = (tokenizationIds: Array<{ datasetId: string; tokenizationId: string }>) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateTokenizationProgress, updateTokenizationStatus } = useDatasetsStore();
  const handlersRegisteredRef = useRef(false);

  // Set up global event handlers (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[Tokenization WS] Setting up tokenization event handlers');

    // Handler for 'tokenization:progress' events
    const handleProgress = (data: DatasetTokenizationProgress) => {
      console.log('[Tokenization WS] Progress event:', data);
      updateTokenizationProgress(data.dataset_id, data.tokenization_id, data);
    };

    // Handler for 'tokenization:status' events (cancel, error, deleted)
    const handleStatus = (data: TokenizationStatusEvent) => {
      console.log('[Tokenization WS] Status event:', data);
      updateTokenizationStatus(data.dataset_id, data.tokenization_id, data.status, data.error_message);
    };

    // Register event handlers
    on('tokenization:progress', handleProgress);
    on('tokenization:status', handleStatus);

    handlersRegisteredRef.current = true;
    console.log('[Tokenization WS] Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Tokenization WS] Cleaning up event handlers');
      off('tokenization:progress', handleProgress);
      off('tokenization:status', handleStatus);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, updateTokenizationProgress, updateTokenizationStatus]);

  // Create a stable key from tokenizationIds to prevent unnecessary re-subscriptions
  const tokenizationIdsKey = useMemo(
    () => tokenizationIds.map(t => `${t.datasetId}:${t.tokenizationId}`).sort().join(','),
    [tokenizationIds.map(t => `${t.datasetId}:${t.tokenizationId}`).join(',')]
  );

  // Subscribe to channels for active tokenizations
  useEffect(() => {
    if (!isConnected) {
      console.log('[Tokenization WS] Not connected, skipping channel subscriptions');
      return;
    }

    if (tokenizationIds.length === 0) {
      console.log('[Tokenization WS] No tokenizations to subscribe to');
      return;
    }

    console.log('[Tokenization WS] Subscribing to', tokenizationIds.length, 'tokenization channels');

    // Subscribe to progress channel for each tokenization
    tokenizationIds.forEach(({ datasetId, tokenizationId }) => {
      const channel = `datasets/${datasetId}/tokenization/${tokenizationId}`;
      console.log(`[Tokenization WS] Subscribing to ${channel}`);
      subscribe(channel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[Tokenization WS] Unsubscribing from tokenization channels');
      tokenizationIds.forEach(({ datasetId, tokenizationId }) => {
        unsubscribe(`datasets/${datasetId}/tokenization/${tokenizationId}`);
      });
    };
  }, [tokenizationIdsKey, isConnected, subscribe, unsubscribe]);
};

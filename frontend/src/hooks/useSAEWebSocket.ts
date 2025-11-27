/**
 * SAE WebSocket Hook
 *
 * React hook for subscribing to SAE download/upload progress via WebSocket.
 * Uses WebSocketContext for proper connection management.
 *
 * WebSocket Channels:
 * - sae/{sae_id}/download - Download progress
 * - sae/{sae_id}/upload - Upload progress
 *
 * Events:
 * - sae:download - Download progress update
 * - sae:upload - Upload progress update
 *
 * Usage:
 *   useSAEWebSocket(saeIds);
 *
 *   // Automatically subscribes/unsubscribes based on saeIds array
 *   // Updates are handled by saesStore
 */

import { useEffect, useRef, useMemo } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useSAEsStore } from '../stores/saesStore';
import { SAEStatus } from '../types/sae';

interface SAEProgressEvent {
  sae_id: string;
  progress: number;
  status: string;
  message?: string;
  stage?: string;
  repo_url?: string;
}

// Map backend status strings to SAEStatus enum
function mapStatus(status: string): SAEStatus {
  switch (status) {
    case 'downloading':
      return SAEStatus.DOWNLOADING;
    case 'converting':
      return SAEStatus.CONVERTING;
    case 'ready':
    case 'completed':
      return SAEStatus.READY;
    case 'failed':
    case 'error':
      return SAEStatus.ERROR;
    case 'uploading':
      // Use CONVERTING as a proxy since there's no UPLOADING status
      return SAEStatus.CONVERTING;
    default:
      return SAEStatus.PENDING;
  }
}

export const useSAEWebSocket = (saeIds: string[]) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateDownloadProgress, fetchSAEs } = useSAEsStore();
  const handlersRegisteredRef = useRef(false);

  // Set up global event handlers (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[SAE WS] Setting up SAE event handlers');

    // Handler for 'sae:download' events
    const handleDownloadProgress = (data: SAEProgressEvent) => {
      console.log('[SAE WS] Download progress:', data);
      updateDownloadProgress(
        data.sae_id,
        data.progress,
        mapStatus(data.status),
        data.message,
      );

      // Auto-refresh list on completion
      if (data.status === 'ready') {
        console.log('[SAE WS] Download complete, refreshing SAE list');
        fetchSAEs();
      }
    };

    // Handler for 'sae:upload' events
    const handleUploadProgress = (data: SAEProgressEvent) => {
      console.log('[SAE WS] Upload progress:', data);
      updateDownloadProgress(
        data.sae_id,
        data.progress,
        mapStatus(data.status),
        data.message,
      );

      // Auto-refresh list on completion
      if (data.status === 'completed') {
        console.log('[SAE WS] Upload complete, refreshing SAE list');
        fetchSAEs();
      }
    };

    // Register event handlers
    on('sae:download', handleDownloadProgress);
    on('sae:upload', handleUploadProgress);

    handlersRegisteredRef.current = true;

    return () => {
      console.log('[SAE WS] Cleaning up SAE event handlers');
      off('sae:download', handleDownloadProgress);
      off('sae:upload', handleUploadProgress);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, updateDownloadProgress, fetchSAEs]);

  // Memoize sorted IDs for stable comparison
  const sortedSAEIds = useMemo(() => [...saeIds].sort(), [saeIds]);

  // Subscribe to channels for each SAE
  useEffect(() => {
    if (!isConnected || sortedSAEIds.length === 0) return;

    console.log('[SAE WS] Subscribing to SAE channels:', sortedSAEIds);

    // Subscribe to download and upload channels for each SAE
    sortedSAEIds.forEach((saeId) => {
      subscribe(`sae/${saeId}/download`);
      subscribe(`sae/${saeId}/upload`);
    });

    return () => {
      console.log('[SAE WS] Unsubscribing from SAE channels:', sortedSAEIds);
      sortedSAEIds.forEach((saeId) => {
        unsubscribe(`sae/${saeId}/download`);
        unsubscribe(`sae/${saeId}/upload`);
      });
    };
  }, [isConnected, sortedSAEIds, subscribe, unsubscribe]);

  return {
    isConnected,
  };
};

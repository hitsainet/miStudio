/**
 * Robust Dataset Progress Hook (V2)
 *
 * This hook provides bulletproof progress tracking for datasets with:
 * - Immediate subscription on mount
 * - Persistent event handlers across reconnects
 * - Automatic cleanup
 * - Proactive channel subscription
 */

import { useEffect, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useDatasetsStore } from '../stores/datasetsStore';
import { DatasetStatus } from '../types/dataset';

/**
 * Global hook that manages progress updates for ALL datasets.
 * Should be used once at the app root level.
 */
export function useGlobalDatasetProgress() {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { datasets, updateDatasetProgress, updateDatasetStatus, fetchDatasets } = useDatasetsStore();
  const handlersRegisteredRef = useRef(false);

  // Set up global event handlers (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[Progress] Setting up global progress event handlers');

    // Handler for 'progress' events
    const handleProgress = (data: any) => {
      console.log('[Progress] Received progress event:', data);

      const datasetId = data.dataset_id || data.id;
      if (datasetId && data.progress !== undefined) {
        console.log(`[Progress] Updating dataset ${datasetId} progress to ${data.progress}%`);
        updateDatasetProgress(datasetId, data.progress);

        // If status is provided in the progress event, update that too
        if (data.status) {
          console.log(`[Progress] Updating dataset ${datasetId} status to ${data.status}`);
          updateDatasetStatus(datasetId, data.status as DatasetStatus);
        }
      }
    };

    // Handler for 'completed' events
    const handleCompleted = (data: any) => {
      console.log('[Progress] Received completed event:', data);

      const datasetId = data.dataset_id || data.id;
      if (datasetId) {
        updateDatasetStatus(datasetId, DatasetStatus.READY);
        // Refresh datasets to get final state
        setTimeout(() => fetchDatasets(), 500);
      }
    };

    // Handler for 'error' events
    const handleError = (data: any) => {
      console.error('[Progress] Received error event:', data);

      const datasetId = data.dataset_id || data.id;
      if (datasetId) {
        updateDatasetStatus(
          datasetId,
          DatasetStatus.ERROR,
          data.message || data.error || 'An error occurred'
        );
      }
    };

    // Register event handlers with namespace prefix for proper WebSocket routing
    on('dataset:progress', handleProgress);
    on('dataset:completed', handleCompleted);
    on('dataset:error', handleError);

    handlersRegisteredRef.current = true;

    console.log('[Progress] Global event handlers registered');

    // Cleanup
    return () => {
      console.log('[Progress] Cleaning up global event handlers');
      off('dataset:progress', handleProgress);
      off('dataset:completed', handleCompleted);
      off('dataset:error', handleError);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, updateDatasetProgress, updateDatasetStatus, fetchDatasets]);

  // Subscribe to channels for active datasets
  useEffect(() => {
    if (!isConnected) {
      console.log('[Progress] Not connected, skipping channel subscriptions');
      return;
    }

    console.log('[Progress] Checking datasets:', datasets.length, 'total datasets');
    console.log('[Progress] Dataset statuses:', datasets.map(d => ({ id: d.id, status: d.status })));

    // Find all datasets that might have active operations
    const activeDatasets = datasets.filter(
      (d) => {
        // Normalize status to string for consistent comparison
        const statusString = String(d.status).toLowerCase();
        const isActive = statusString === 'downloading' || statusString === 'processing';
        console.log(`[Progress] Dataset ${d.id} status: "${d.status}" (${typeof d.status}), normalized: "${statusString}", isActive: ${isActive}`);
        return isActive;
      }
    );

    if (activeDatasets.length === 0) {
      console.log('[Progress] No active datasets to subscribe to');
      return;
    }

    console.log('[Progress] Subscribing to', activeDatasets.length, 'active dataset channels');

    // Subscribe to each dataset's progress channel
    const channels: string[] = [];
    activeDatasets.forEach((dataset) => {
      const channel = `datasets/${dataset.id}/progress`;
      console.log('[Progress] Subscribing to channel:', channel);
      channels.push(channel);
      subscribe(channel);
    });

    // Cleanup: unsubscribe when datasets change or component unmounts
    return () => {
      console.log('[Progress] Unsubscribing from', channels.length, 'channels');
      channels.forEach(channel => unsubscribe(channel));
    };
  }, [datasets, subscribe, unsubscribe, isConnected]);
}

/**
 * Hook to subscribe to a specific dataset's progress.
 * Use this to proactively subscribe BEFORE starting a download/tokenization.
 */
export function useDatasetProgressSubscription(datasetId: string | null) {
  const { subscribe, unsubscribe, isConnected } = useWebSocketContext();

  useEffect(() => {
    if (!datasetId || !isConnected) return;

    const channel = `datasets/${datasetId}/progress`;
    console.log('[Progress] Proactively subscribing to channel:', channel);
    subscribe(channel);

    return () => {
      console.log('[Progress] Unsubscribing from channel:', channel);
      unsubscribe(channel);
    };
  }, [datasetId, subscribe, unsubscribe, isConnected]);
}

/**
 * Helper function to subscribe to a dataset's progress channel immediately.
 * Call this right after initiating a download or tokenization.
 */
export function subscribeToDatasetProgress(datasetId: string, ws: ReturnType<typeof useWebSocketContext>) {
  const channel = `datasets/${datasetId}/progress`;
  console.log('[Progress] Immediate subscription to channel:', channel);
  ws.subscribe(channel);
}

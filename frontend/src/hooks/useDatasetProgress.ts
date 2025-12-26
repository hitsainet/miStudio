/**
 * Hook for dataset progress updates via WebSocket.
 *
 * This hook subscribes to real-time progress updates for datasets.
 *
 * IMPORTANT: This hook now uses WebSocketContext which properly separates:
 * - subscribe(channel): Joins a Socket.IO room to receive events for that room
 * - on(event, handler): Listens for events with a specific event name
 *
 * This is the correct pattern - events are named (e.g., 'dataset:progress')
 * and rooms determine who receives them (e.g., 'datasets/{id}/progress').
 */

import { useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useDatasetsStore } from '../stores/datasetsStore';
import { DatasetStatus } from '../types/dataset';

interface ProgressEvent {
  type: 'progress' | 'completed' | 'error';
  dataset_id: string;
  progress?: number;
  status?: DatasetStatus;
  error?: string;
  message?: string;
}

export function useDatasetProgress(datasetId?: string) {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateDatasetProgress, updateDatasetStatus } = useDatasetsStore();
  const handlerRef = useRef<((event: ProgressEvent) => void) | null>(null);

  useEffect(() => {
    if (!datasetId || !isConnected) return;

    const channel = `datasets/${datasetId}/progress`;

    const handleProgressUpdate = (event: ProgressEvent) => {
      console.log('[useDatasetProgress] Progress update:', event);

      // Only process events for this specific dataset
      if (event.dataset_id !== datasetId) {
        return;
      }

      switch (event.type) {
        case 'progress':
          if (event.progress !== undefined) {
            updateDatasetProgress(event.dataset_id, event.progress);
          }
          break;

        case 'completed':
          updateDatasetStatus(event.dataset_id, DatasetStatus.READY);
          break;

        case 'error':
          updateDatasetStatus(
            event.dataset_id,
            DatasetStatus.ERROR,
            event.error || 'An error occurred'
          );
          break;
      }
    };

    // Store handler ref for cleanup
    handlerRef.current = handleProgressUpdate;

    // Join the room (channel) to receive events for this dataset
    console.log(`[useDatasetProgress] Subscribing to room: ${channel}`);
    subscribe(channel);

    // Listen for dataset:progress events
    console.log('[useDatasetProgress] Listening for dataset:progress events');
    on('dataset:progress', handleProgressUpdate);

    return () => {
      console.log(`[useDatasetProgress] Unsubscribing from room: ${channel}`);
      unsubscribe(channel);
      if (handlerRef.current) {
        off('dataset:progress', handlerRef.current);
        handlerRef.current = null;
      }
    };
  }, [datasetId, isConnected, subscribe, unsubscribe, on, off, updateDatasetProgress, updateDatasetStatus]);
}

/**
 * Hook to monitor progress for all downloading/processing datasets
 */
export function useAllDatasetsProgress() {
  const { datasets } = useDatasetsStore();
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateDatasetProgress, updateDatasetStatus, fetchDatasets } = useDatasetsStore();

  // Store handler refs for cleanup
  const handlersRef = useRef<{
    progress: ((data: any) => void) | null;
    completed: ((data: any) => void) | null;
    error: ((data: any) => void) | null;
  }>({
    progress: null,
    completed: null,
    error: null,
  });

  // Handler for 'progress' events
  const handleProgress = useCallback((data: any) => {
    console.log('[useAllDatasetsProgress] Progress update:', data);

    if (data.progress !== undefined) {
      // Extract dataset ID from the data
      const datasetId = data.dataset_id || data.id;
      if (datasetId) {
        updateDatasetProgress(datasetId, data.progress);
      }
    }
  }, [updateDatasetProgress]);

  // Handler for 'completed' events
  const handleCompleted = useCallback((data: any) => {
    console.log('[useAllDatasetsProgress] Dataset completed:', data);
    const datasetId = data.dataset_id || data.id;
    if (datasetId) {
      updateDatasetStatus(datasetId, DatasetStatus.READY);
      // Refresh all datasets to get the latest data
      fetchDatasets();
    }
  }, [updateDatasetStatus, fetchDatasets]);

  // Handler for 'error' events
  const handleError = useCallback((data: any) => {
    console.error('[useAllDatasetsProgress] Dataset error:', data);
    const datasetId = data.dataset_id || data.id;
    if (datasetId) {
      updateDatasetStatus(
        datasetId,
        DatasetStatus.ERROR,
        data.message || data.error || 'An error occurred'
      );
    }
  }, [updateDatasetStatus]);

  // Register event handlers
  useEffect(() => {
    console.log('[useAllDatasetsProgress] Setting up event handlers');

    // Store refs for cleanup
    handlersRef.current = {
      progress: handleProgress,
      completed: handleCompleted,
      error: handleError,
    };

    // Register event handlers
    on('dataset:progress', handleProgress);
    on('dataset:completed', handleCompleted);
    on('dataset:error', handleError);

    console.log('[useAllDatasetsProgress] Event handlers registered');

    return () => {
      console.log('[useAllDatasetsProgress] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('dataset:progress', handlers.progress);
      if (handlers.completed) off('dataset:completed', handlers.completed);
      if (handlers.error) off('dataset:error', handlers.error);

      // Clear refs
      handlersRef.current = {
        progress: null,
        completed: null,
        error: null,
      };
    };
  }, [on, off, handleProgress, handleCompleted, handleError]);

  // Subscribe to channels
  useEffect(() => {
    if (!isConnected) {
      console.log('[useAllDatasetsProgress] Not connected, skipping channel subscriptions');
      return;
    }

    // Subscribe to progress updates for all downloading/processing datasets
    const activeDatasets = datasets.filter(
      (d) =>
        d.status === DatasetStatus.DOWNLOADING || d.status === DatasetStatus.PROCESSING
    );

    // Subscribe to each active dataset's channel (joins the rooms)
    const channels = activeDatasets.map((dataset) => `datasets/${dataset.id}/progress`);
    channels.forEach((channel) => {
      console.log(`[useAllDatasetsProgress] Subscribing to ${channel}`);
      subscribe(channel);
    });

    return () => {
      console.log('[useAllDatasetsProgress] Unsubscribing from channels');

      // Leave all rooms
      channels.forEach((channel) => {
        unsubscribe(channel);
      });
    };
  }, [datasets.map(d => `${d.id}:${d.status}`).join(','), isConnected, subscribe, unsubscribe]);
}

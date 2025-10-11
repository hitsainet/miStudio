/**
 * Hook for dataset progress updates via WebSocket.
 *
 * This hook subscribes to real-time progress updates for datasets.
 */

import { useEffect } from 'react';
import { useWebSocket } from './useWebSocket';
import { useDatasetsStore } from '../stores/datasetsStore';
import { DatasetStatus } from '../types/dataset';

interface ProgressEvent {
  type: 'progress' | 'completed' | 'error';
  dataset_id: string;
  progress?: number;
  status?: DatasetStatus;
  error?: string;
}

export function useDatasetProgress(datasetId?: string) {
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateDatasetProgress, updateDatasetStatus } = useDatasetsStore();

  useEffect(() => {
    if (!datasetId) return;

    const channel = `datasets/${datasetId}/progress`;

    const handleProgressUpdate = (event: ProgressEvent) => {
      console.log('Progress update:', event);

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

    subscribe(channel, handleProgressUpdate);

    return () => {
      unsubscribe(channel, handleProgressUpdate);
    };
  }, [datasetId, subscribe, unsubscribe, updateDatasetProgress, updateDatasetStatus]);
}

/**
 * Hook to monitor progress for all downloading/processing datasets
 */
export function useAllDatasetsProgress() {
  const { datasets } = useDatasetsStore();
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateDatasetProgress, updateDatasetStatus, fetchDatasets } = useDatasetsStore();

  useEffect(() => {
    // Subscribe to progress updates for all downloading/processing datasets
    const activeDatasets = datasets.filter(
      (d) =>
        d.status === DatasetStatus.DOWNLOADING || d.status === DatasetStatus.PROCESSING
    );

    // Handler for 'progress' events
    const handleProgress = (data: any) => {
      console.log('All datasets progress update:', data);

      if (data.progress !== undefined) {
        // Extract dataset ID from the data or context
        // We need to track which dataset this progress is for
        const datasetId = data.dataset_id || data.id;
        if (datasetId) {
          updateDatasetProgress(datasetId, data.progress);
        }
      }
    };

    // Handler for 'completed' events
    const handleCompleted = (data: any) => {
      console.log('Dataset completed:', data);
      const datasetId = data.dataset_id || data.id;
      if (datasetId) {
        updateDatasetStatus(datasetId, DatasetStatus.READY);
        // Refresh all datasets to get the latest data
        fetchDatasets();
      }
    };

    // Handler for 'error' events
    const handleError = (data: any) => {
      console.error('Dataset error:', data);
      const datasetId = data.dataset_id || data.id;
      if (datasetId) {
        updateDatasetStatus(
          datasetId,
          DatasetStatus.ERROR,
          data.message || data.error || 'An error occurred'
        );
      }
    };

    // Subscribe to the event types (these are global, but only room members receive them)
    subscribe('progress', handleProgress);
    subscribe('completed', handleCompleted);
    subscribe('error', handleError);

    // Subscribe to each active dataset's channel (joins the rooms)
    activeDatasets.forEach((dataset) => {
      const channel = `datasets/${dataset.id}/progress`;
      subscribe(channel, () => {
        console.log(`Joined channel for dataset: ${dataset.id}`);
      });
    });

    return () => {
      // Unsubscribe from event handlers
      unsubscribe('progress', handleProgress);
      unsubscribe('completed', handleCompleted);
      unsubscribe('error', handleError);

      // Leave all rooms
      activeDatasets.forEach((dataset) => {
        const channel = `datasets/${dataset.id}/progress`;
        unsubscribe(channel);
      });
    };
  }, [datasets, subscribe, unsubscribe, updateDatasetProgress, updateDatasetStatus, fetchDatasets]);
}

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
  const { updateDatasetProgress, updateDatasetStatus } = useDatasetsStore();

  useEffect(() => {
    // Subscribe to progress updates for all downloading/processing datasets
    const activeDatasets = datasets.filter(
      (d) =>
        d.status === DatasetStatus.DOWNLOADING || d.status === DatasetStatus.PROCESSING
    );

    const handlers: Array<{ channel: string; handler: (event: ProgressEvent) => void }> = [];

    activeDatasets.forEach((dataset) => {
      const channel = `datasets/${dataset.id}/progress`;

      const handler = (event: ProgressEvent) => {
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

      subscribe(channel, handler);
      handlers.push({ channel, handler });
    });

    return () => {
      handlers.forEach(({ channel, handler }) => {
        unsubscribe(channel, handler);
      });
    };
  }, [datasets, subscribe, unsubscribe, updateDatasetProgress, updateDatasetStatus]);
}

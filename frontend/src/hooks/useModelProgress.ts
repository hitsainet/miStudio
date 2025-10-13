/**
 * Hook for model progress updates via WebSocket.
 *
 * This hook subscribes to real-time progress updates for models.
 * Handles both model download/quantization progress and activation extraction progress.
 */

import { useEffect } from 'react';
import { useWebSocket } from './useWebSocket';
import { useModelsStore } from '../stores/modelsStore';
import { ModelStatus } from '../types/model';

interface ModelProgressEvent {
  type: 'progress' | 'completed' | 'error' | 'extraction_progress' | 'extraction_completed';
  model_id: string;
  progress?: number;
  status?: ModelStatus | string;
  error?: string;
  // Extraction-specific fields
  samples_processed?: number;
  eta_seconds?: number;
  extraction_id?: string;
  output_path?: string;
  statistics?: Record<string, any>;
}

/**
 * Hook for monitoring progress of a single model
 */
export function useModelProgress(modelId?: string, channel: 'progress' | 'extraction' = 'progress') {
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateModelProgress, updateModelStatus } = useModelsStore();

  useEffect(() => {
    if (!modelId) return;

    const channelPath = `models/${modelId}/${channel}`;

    const handleProgressUpdate = (event: ModelProgressEvent) => {
      console.log(`[useModelProgress] ${channel} update:`, event);

      switch (event.type) {
        case 'progress':
          if (event.progress !== undefined) {
            updateModelProgress(event.model_id, event.progress);
          }
          if (event.status) {
            const status = typeof event.status === 'string' ? event.status.toLowerCase() as ModelStatus : event.status;
            updateModelStatus(event.model_id, status);
          }
          break;

        case 'completed':
          updateModelStatus(event.model_id, ModelStatus.READY);
          updateModelProgress(event.model_id, 100);
          break;

        case 'error':
          updateModelStatus(
            event.model_id,
            ModelStatus.ERROR,
            event.error || 'An error occurred'
          );
          break;

        case 'extraction_progress':
          console.log(`[useModelProgress] Extraction progress: ${event.progress}% (${event.samples_processed} samples)`);
          break;

        case 'extraction_completed':
          console.log(`[useModelProgress] Extraction completed: ${event.extraction_id}`);
          break;
      }
    };

    subscribe(channelPath, handleProgressUpdate);

    return () => {
      unsubscribe(channelPath, handleProgressUpdate);
    };
  }, [modelId, channel, subscribe, unsubscribe, updateModelProgress, updateModelStatus]);
}

/**
 * Hook to monitor progress for all downloading/loading/quantizing models
 */
export function useAllModelsProgress() {
  const { models } = useModelsStore();
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateModelProgress, updateModelStatus, updateExtractionProgress, fetchModels } = useModelsStore();

  useEffect(() => {
    // Subscribe to progress updates for all active models
    const activeModels = models.filter(
      (m) =>
        m.status === ModelStatus.DOWNLOADING ||
        m.status === ModelStatus.LOADING ||
        m.status === ModelStatus.QUANTIZING
    );

    // Subscribe to extraction updates for all ready models
    const readyModels = models.filter((m) => m.status === ModelStatus.READY);

    // Handler for 'progress' events
    const handleProgress = (data: ModelProgressEvent) => {
      console.log('[useAllModelsProgress] Progress update:', data);

      if (data.progress !== undefined) {
        const modelId = data.model_id;
        if (modelId) {
          updateModelProgress(modelId, data.progress);
        }
      }

      if (data.status) {
        const modelId = data.model_id;
        const status = typeof data.status === 'string' ? data.status.toLowerCase() as ModelStatus : data.status;
        if (modelId) {
          updateModelStatus(modelId, status);
        }
      }
    };

    // Handler for 'completed' events
    const handleCompleted = (data: ModelProgressEvent) => {
      console.log('[useAllModelsProgress] Model completed:', data);
      const modelId = data.model_id;
      if (modelId) {
        updateModelStatus(modelId, ModelStatus.READY);
        updateModelProgress(modelId, 100);
        // Refresh all models to get the latest data
        fetchModels();
      }
    };

    // Handler for 'error' events
    const handleError = (data: ModelProgressEvent) => {
      console.error('[useAllModelsProgress] Model error:', data);
      const modelId = data.model_id;
      if (modelId) {
        updateModelStatus(
          modelId,
          ModelStatus.ERROR,
          data.error || 'An error occurred'
        );
      }
    };

    // Handler for extraction progress events
    const handleExtractionProgress = (event: any) => {
      console.log('[useAllModelsProgress] Extraction progress:', event);
      const data = event.data || event;

      if (data.model_id && data.extraction_id && data.progress !== undefined && data.status && data.message) {
        updateExtractionProgress(
          data.model_id,
          data.extraction_id,
          data.progress,
          data.status,
          data.message
        );
      }
    };

    // Subscribe to the event types (these are global, but only room members receive them)
    subscribe('progress', handleProgress);
    subscribe('completed', handleCompleted);
    subscribe('error', handleError);

    // Subscribe to each active model's progress channel (joins the rooms)
    activeModels.forEach((model) => {
      const progressChannel = `models/${model.id}/progress`;
      subscribe(progressChannel, () => {
        console.log(`[useAllModelsProgress] Joined progress channel for model: ${model.id}`);
      });
    });

    // Subscribe to extraction channels for all ready models
    readyModels.forEach((model) => {
      const extractionChannel = `models/${model.id}/extraction`;
      subscribe(extractionChannel, handleExtractionProgress);
      console.log(`[useAllModelsProgress] Subscribed to extraction channel for model: ${model.id}`);
    });

    return () => {
      // Unsubscribe from event handlers
      unsubscribe('progress', handleProgress);
      unsubscribe('completed', handleCompleted);
      unsubscribe('error', handleError);

      // Leave all progress rooms
      activeModels.forEach((model) => {
        const progressChannel = `models/${model.id}/progress`;
        unsubscribe(progressChannel);
      });

      // Leave all extraction rooms
      readyModels.forEach((model) => {
        const extractionChannel = `models/${model.id}/extraction`;
        unsubscribe(extractionChannel, handleExtractionProgress);
      });
    };
  }, [models, subscribe, unsubscribe, updateModelProgress, updateModelStatus, updateExtractionProgress, fetchModels]);
}

/**
 * Hook for monitoring activation extraction progress for a specific model
 */
export function useModelExtractionProgress(modelId?: string) {
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateExtractionProgress } = useModelsStore();

  useEffect(() => {
    if (!modelId) return;

    const extractionChannel = `models/${modelId}/extraction`;

    const handleExtractionUpdate = (event: any) => {
      console.log('[useModelExtractionProgress] Extraction update:', event);

      // Backend sends { type: 'extraction_progress', model_id, extraction_id, progress, status, message }
      const data = event.data || event;

      if (data.model_id && data.extraction_id && data.progress !== undefined && data.status && data.message) {
        updateExtractionProgress(
          data.model_id,
          data.extraction_id,
          data.progress,
          data.status,
          data.message
        );
      }
    };

    subscribe(extractionChannel, handleExtractionUpdate);

    return () => {
      unsubscribe(extractionChannel, handleExtractionUpdate);
    };
  }, [modelId, subscribe, unsubscribe, updateExtractionProgress]);
}

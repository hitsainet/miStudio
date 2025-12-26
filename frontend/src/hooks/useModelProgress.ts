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
  type: 'progress' | 'completed' | 'error' | 'extraction_progress' | 'extraction_completed' | 'extraction_failed';
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
  // Failure-specific fields
  error_type?: string;
  error_message?: string;
  suggested_retry_params?: Record<string, any>;
  retry_available?: boolean;
  cancel_available?: boolean;
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
 *
 * This hook also checks for active extractions on mount to restore state after page refresh.
 */
export function useAllModelsProgress() {
  const { models } = useModelsStore();
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateModelProgress, updateModelStatus, updateExtractionProgress, clearExtractionProgress, updateExtractionFailure, fetchModels, checkActiveExtraction } = useModelsStore();

  useEffect(() => {
    // On mount, check all ready models for active extractions
    const readyModels = models.filter((m) => m.status === ModelStatus.READY);

    readyModels.forEach(async (model) => {
      const hasActiveExtraction = await checkActiveExtraction(model.id);
      if (hasActiveExtraction) {
        console.log('[useAllModelsProgress] Restored active extraction for model:', model.id);
      }
    });

    // Subscribe to progress updates for all active models
    const activeModels = models.filter(
      (m) =>
        m.status === ModelStatus.DOWNLOADING ||
        m.status === ModelStatus.LOADING ||
        m.status === ModelStatus.QUANTIZING
    );

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

        // On completion, keep the 'complete' status visible so users see the completion message
        // Auto-clear after 5 seconds to reset for next extraction
        if (data.status === 'complete' || data.status === 'completed') {
          console.log('[useAllModelsProgress] Extraction completed, showing completion state');
          setTimeout(() => {
            console.log('[useAllModelsProgress] Auto-clearing completion state after 5s');
            clearExtractionProgress(data.model_id);
          }, 5000);
        }
        // For failed/cancelled, keep state visible indefinitely for user to see error
        // State will be cleared when user starts a new extraction
      }
    };

    // Handler for extraction failed events (dedicated failure with retry options)
    const handleExtractionFailed = (event: any) => {
      console.log('[useAllModelsProgress] Extraction failed:', event);
      const data = event.data || event;

      if (data.model_id && data.extraction_id && data.error_message) {
        updateExtractionFailure(
          data.model_id,
          data.extraction_id,
          data.error_type || 'UNKNOWN',
          data.error_message,
          data.suggested_retry_params
        );
      }
    };

    // Subscribe to the event types with namespace prefixes for proper WebSocket routing
    subscribe('model:progress', handleProgress);
    subscribe('model:completed', handleCompleted);
    subscribe('model:error', handleError);

    // Subscribe to each active model's progress channel (joins the rooms)
    activeModels.forEach((model) => {
      const progressChannel = `models/${model.id}/progress`;
      subscribe(progressChannel, () => {
        console.log(`[useAllModelsProgress] Joined progress channel for model: ${model.id}`);
      });
    });

    // Join extraction rooms for all ready models (so server sends us events for these rooms)
    readyModels.forEach((model) => {
      const extractionChannel = `models/${model.id}/extraction`;
      // Join the room - use empty handler since we listen on the event name below
      subscribe(extractionChannel, () => {});
      console.log(`[useAllModelsProgress] Joined extraction room for model: ${model.id}`);
    });

    // Subscribe to 'extraction:progress' event - this is the actual event name the server emits
    // The room membership ensures we only receive events for our subscribed models
    subscribe('extraction:progress', handleExtractionProgress);

    // Subscribe to 'extraction:failed' event for extraction failures
    subscribe('extraction:failed', handleExtractionFailed);

    return () => {
      // Unsubscribe from event handlers
      unsubscribe('model:progress', handleProgress);
      unsubscribe('model:completed', handleCompleted);
      unsubscribe('model:error', handleError);
      unsubscribe('extraction:failed', handleExtractionFailed);

      // Leave all progress rooms
      activeModels.forEach((model) => {
        const progressChannel = `models/${model.id}/progress`;
        unsubscribe(progressChannel);
      });

      // Leave all extraction rooms
      readyModels.forEach((model) => {
        const extractionChannel = `models/${model.id}/extraction`;
        unsubscribe(extractionChannel);
      });

      // Unsubscribe from extraction:progress event
      unsubscribe('extraction:progress', handleExtractionProgress);
    };
  }, [models, subscribe, unsubscribe, updateModelProgress, updateModelStatus, updateExtractionProgress, updateExtractionFailure, fetchModels, checkActiveExtraction]);
}

/**
 * Hook for monitoring activation extraction progress for a specific model
 *
 * This hook:
 * 1. On mount, checks for any active extraction (restores state after page refresh)
 * 2. Subscribes to WebSocket for real-time updates
 * 3. Handles extraction completion by clearing active extraction state
 */
export function useModelExtractionProgress(modelId?: string) {
  const { subscribe, unsubscribe } = useWebSocket();
  const { updateExtractionProgress, clearExtractionProgress, checkActiveExtraction } = useModelsStore();

  useEffect(() => {
    if (!modelId) return;

    // On mount, check for active extraction to restore state
    const initializeExtraction = async () => {
      console.log('[useModelExtractionProgress] Checking for active extraction on mount for model:', modelId);
      const hasActiveExtraction = await checkActiveExtraction(modelId);

      if (hasActiveExtraction) {
        console.log('[useModelExtractionProgress] Active extraction found and restored');
      } else {
        console.log('[useModelExtractionProgress] No active extraction found');
      }
    };

    initializeExtraction();

    const extractionChannel = `models/${modelId}/extraction`;

    const handleExtractionUpdate = (event: any) => {
      console.log('[useModelExtractionProgress] Extraction update:', event);

      // Backend sends { type: 'extraction_progress', model_id, extraction_id, progress, status, message }
      const data = event.data || event;

      // Only process events for this specific model
      if (data.model_id !== modelId) {
        return;
      }

      if (data.model_id && data.extraction_id && data.progress !== undefined && data.status && data.message) {
        updateExtractionProgress(
          data.model_id,
          data.extraction_id,
          data.progress,
          data.status,
          data.message
        );
      }

      // On completion, keep the 'complete' status visible so users see the completion message
      // Auto-clear after 5 seconds to reset for next extraction
      if (data.status === 'complete' || data.status === 'completed') {
        console.log('[useModelExtractionProgress] Extraction completed, showing completion state');
        setTimeout(() => {
          console.log('[useModelExtractionProgress] Auto-clearing completion state after 5s');
          clearExtractionProgress(data.model_id);
        }, 5000);
      }
      // For failed/cancelled, keep state visible indefinitely for user to see error
      // State will be cleared when user starts a new extraction
    };

    // Join the extraction room for this model
    subscribe(extractionChannel, () => {});

    // Listen for extraction:progress events (the actual event name the server emits)
    subscribe('extraction:progress', handleExtractionUpdate);

    return () => {
      unsubscribe(extractionChannel);
      unsubscribe('extraction:progress', handleExtractionUpdate);
    };
  }, [modelId, subscribe, unsubscribe, updateExtractionProgress, clearExtractionProgress, checkActiveExtraction]);
}

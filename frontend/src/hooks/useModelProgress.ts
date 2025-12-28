/**
 * Hook for model progress updates via WebSocket.
 *
 * This hook subscribes to real-time progress updates for models.
 * Handles both model download/quantization progress and activation extraction progress.
 *
 * IMPORTANT: This hook now uses WebSocketContext which properly separates:
 * - subscribe(channel): Joins a Socket.IO room to receive events for that room
 * - on(event, handler): Listens for events with a specific event name
 *
 * This is the correct pattern - events are named (e.g., 'extraction:progress')
 * and rooms determine who receives them (e.g., 'models/{id}/extraction').
 */

import { useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
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
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateModelProgress, updateModelStatus } = useModelsStore();
  const handlerRef = useRef<((event: ModelProgressEvent) => void) | null>(null);

  useEffect(() => {
    if (!modelId || !isConnected) return;

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

    // Store handler ref for cleanup
    handlerRef.current = handleProgressUpdate;

    // Join the room (channel) to receive events for this model
    subscribe(channelPath);

    // Listen for specific event types based on channel
    const eventName = channel === 'extraction' ? 'extraction:progress' : 'model:progress';
    on(eventName, handleProgressUpdate);

    return () => {
      unsubscribe(channelPath);
      if (handlerRef.current) {
        off(eventName, handlerRef.current);
        handlerRef.current = null;
      }
    };
  }, [modelId, channel, isConnected, subscribe, unsubscribe, on, off, updateModelProgress, updateModelStatus]);
}

/**
 * Hook to monitor progress for all downloading/loading/quantizing models
 *
 * This hook also checks for active extractions on mount to restore state after page refresh.
 */
export function useAllModelsProgress() {
  const { models } = useModelsStore();
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateModelProgress, updateModelStatus, updateExtractionProgress, clearExtractionProgress, updateExtractionFailure, fetchModels, checkActiveExtraction } = useModelsStore();

  // Store handler refs for cleanup
  const handlersRef = useRef<{
    progress: ((data: ModelProgressEvent) => void) | null;
    completed: ((data: ModelProgressEvent) => void) | null;
    error: ((data: ModelProgressEvent) => void) | null;
    extractionProgress: ((event: any) => void) | null;
    extractionFailed: ((event: any) => void) | null;
  }>({
    progress: null,
    completed: null,
    error: null,
    extractionProgress: null,
    extractionFailed: null,
  });

  // Handler for 'progress' events
  const handleProgress = useCallback((data: ModelProgressEvent) => {
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
  }, [updateModelProgress, updateModelStatus]);

  // Handler for 'completed' events
  const handleCompleted = useCallback((data: ModelProgressEvent) => {
    console.log('[useAllModelsProgress] Model completed:', data);
    const modelId = data.model_id;
    if (modelId) {
      updateModelStatus(modelId, ModelStatus.READY);
      updateModelProgress(modelId, 100);
      // Refresh all models to get the latest data
      fetchModels();
    }
  }, [updateModelStatus, updateModelProgress, fetchModels]);

  // Handler for 'error' events
  const handleError = useCallback((data: ModelProgressEvent) => {
    console.error('[useAllModelsProgress] Model error:', data);
    const modelId = data.model_id;
    if (modelId) {
      updateModelStatus(
        modelId,
        ModelStatus.ERROR,
        data.error || 'An error occurred'
      );
    }
  }, [updateModelStatus]);

  // Handler for extraction progress events
  const handleExtractionProgress = useCallback((event: any) => {
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
          // Pass true to indicate successful completion - this will set has_completed_extractions
          clearExtractionProgress(data.model_id, true);
        }, 5000);
      }
      // For failed/cancelled, keep state visible indefinitely for user to see error
      // State will be cleared when user starts a new extraction
    }
  }, [updateExtractionProgress, clearExtractionProgress]);

  // Handler for extraction failed events (dedicated failure with retry options)
  const handleExtractionFailed = useCallback((event: any) => {
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
  }, [updateExtractionFailure]);

  // Check for active extractions on mount
  useEffect(() => {
    // On mount, check all ready models for active extractions
    const readyModels = models.filter((m) => m.status === ModelStatus.READY);

    readyModels.forEach(async (model) => {
      const hasActiveExtraction = await checkActiveExtraction(model.id);
      if (hasActiveExtraction) {
        console.log('[useAllModelsProgress] Restored active extraction for model:', model.id);
      }
    });
  }, [models.map(m => m.id).join(','), checkActiveExtraction]);

  // Register event handlers
  useEffect(() => {
    console.log('[useAllModelsProgress] Setting up event handlers');

    // Store refs for cleanup
    handlersRef.current = {
      progress: handleProgress,
      completed: handleCompleted,
      error: handleError,
      extractionProgress: handleExtractionProgress,
      extractionFailed: handleExtractionFailed,
    };

    // Register event handlers for model events
    on('model:progress', handleProgress);
    on('model:completed', handleCompleted);
    on('model:error', handleError);

    // Register event handlers for extraction events
    on('extraction:progress', handleExtractionProgress);
    on('extraction:failed', handleExtractionFailed);

    console.log('[useAllModelsProgress] Event handlers registered');

    return () => {
      console.log('[useAllModelsProgress] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('model:progress', handlers.progress);
      if (handlers.completed) off('model:completed', handlers.completed);
      if (handlers.error) off('model:error', handlers.error);
      if (handlers.extractionProgress) off('extraction:progress', handlers.extractionProgress);
      if (handlers.extractionFailed) off('extraction:failed', handlers.extractionFailed);

      // Clear refs
      handlersRef.current = {
        progress: null,
        completed: null,
        error: null,
        extractionProgress: null,
        extractionFailed: null,
      };
    };
  }, [on, off, handleProgress, handleCompleted, handleError, handleExtractionProgress, handleExtractionFailed]);

  // Subscribe to channels
  useEffect(() => {
    if (!isConnected) {
      console.log('[useAllModelsProgress] Not connected, skipping channel subscriptions');
      return;
    }

    // Subscribe to progress updates for all active models (downloading/loading/quantizing)
    const activeModels = models.filter(
      (m) =>
        m.status === ModelStatus.DOWNLOADING ||
        m.status === ModelStatus.LOADING ||
        m.status === ModelStatus.QUANTIZING
    );

    // Subscribe to each active model's progress channel (joins the rooms)
    const progressChannels = activeModels.map((model) => `models/${model.id}/progress`);
    progressChannels.forEach((channel) => {
      console.log(`[useAllModelsProgress] Subscribing to ${channel}`);
      subscribe(channel);
    });

    // Join extraction rooms for all ready models (so server sends us events for these rooms)
    const readyModels = models.filter((m) => m.status === ModelStatus.READY);
    const extractionChannels = readyModels.map((model) => `models/${model.id}/extraction`);
    extractionChannels.forEach((channel) => {
      console.log(`[useAllModelsProgress] Subscribing to extraction room: ${channel}`);
      subscribe(channel);
    });

    return () => {
      console.log('[useAllModelsProgress] Unsubscribing from channels');

      // Leave all progress rooms
      progressChannels.forEach((channel) => {
        unsubscribe(channel);
      });

      // Leave all extraction rooms
      extractionChannels.forEach((channel) => {
        unsubscribe(channel);
      });
    };
  }, [models.map(m => `${m.id}:${m.status}`).join(','), isConnected, subscribe, unsubscribe]);
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
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateExtractionProgress, clearExtractionProgress, checkActiveExtraction } = useModelsStore();
  const handlerRef = useRef<((event: any) => void) | null>(null);

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
  }, [modelId, checkActiveExtraction]);

  useEffect(() => {
    if (!modelId || !isConnected) return;

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
          // Pass true to indicate successful completion - this will set has_completed_extractions
          clearExtractionProgress(data.model_id, true);
        }, 5000);
      }
      // For failed/cancelled, keep state visible indefinitely for user to see error
      // State will be cleared when user starts a new extraction
    };

    // Store handler ref for cleanup
    handlerRef.current = handleExtractionUpdate;

    // Join the extraction room for this model
    console.log(`[useModelExtractionProgress] Subscribing to room: ${extractionChannel}`);
    subscribe(extractionChannel);

    // Listen for extraction:progress events (the actual event name the server emits)
    console.log('[useModelExtractionProgress] Listening for extraction:progress events');
    on('extraction:progress', handleExtractionUpdate);

    return () => {
      console.log(`[useModelExtractionProgress] Unsubscribing from room: ${extractionChannel}`);
      unsubscribe(extractionChannel);
      if (handlerRef.current) {
        off('extraction:progress', handlerRef.current);
        handlerRef.current = null;
      }
    };
  }, [modelId, isConnected, subscribe, unsubscribe, on, off, updateExtractionProgress, clearExtractionProgress]);
}

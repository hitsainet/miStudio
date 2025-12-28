/**
 * Extraction WebSocket Hook
 *
 * React hook for subscribing to extraction progress events via WebSocket.
 * Uses WebSocketContext for proper connection management and event queuing.
 *
 * WebSocket Channels:
 * - extraction/{extraction_id} - Progress updates for specific extraction
 *
 * Events:
 * - extraction:progress - Progress update (emitted during extraction)
 * - extraction:completed - Extraction completed
 * - extraction:failed - Extraction failed
 *
 * Usage:
 *   useExtractionWebSocket(extractionIds);
 *
 *   // Automatically subscribes/unsubscribes based on extractionIds array
 *   // Updates are handled by featuresStore.updateExtractionById()
 */

import { useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useFeaturesStore } from '../stores/featuresStore';
import type { ExtractionStatus } from '../types/features';

export interface ExtractionProgressEvent {
  extraction_id: string;
  status: ExtractionStatus;
  progress: number;
  features_extracted: number | null;
  total_features: number | null;
  error_message?: string;
  message?: string;  // Status message (e.g., "Saving features to database...")
  // Detailed progress metrics (from emit_extraction_job_progress)
  current_batch?: number;
  total_batches?: number;
  samples_processed?: number;
  total_samples?: number;
  samples_per_second?: number;
  eta_seconds?: number;
  features_in_heap?: number;
  heap_examples_count?: number;
}

export const useExtractionWebSocket = (extractionIds: string[]) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const updateExtractionById = useFeaturesStore((state) => state.updateExtractionById);
  const fetchAllExtractions = useFeaturesStore((state) => state.fetchAllExtractions);

  // Use refs to store handler references for cleanup
  const handlersRef = useRef<{
    progress: ((data: ExtractionProgressEvent) => void) | null;
    completed: ((data: ExtractionProgressEvent) => void) | null;
    failed: ((data: ExtractionProgressEvent) => void) | null;
  }>({
    progress: null,
    completed: null,
    failed: null,
  });

  // Create stable handler using useCallback
  const handleProgress = useCallback((data: ExtractionProgressEvent) => {
    console.log('[Extraction WS] 📊 Progress event received:', data);
    // Build update object, only including defined values to avoid overwriting with undefined
    const update: Partial<ExtractionProgressEvent> = {
      status: data.status,
      progress: data.progress,
    };
    // Only include optional fields if they are defined
    if (data.features_extracted !== undefined) update.features_extracted = data.features_extracted;
    if (data.total_features !== undefined) update.total_features = data.total_features;
    if (data.current_batch !== undefined) update.current_batch = data.current_batch;
    if (data.total_batches !== undefined) update.total_batches = data.total_batches;
    if (data.samples_processed !== undefined) update.samples_processed = data.samples_processed;
    if (data.total_samples !== undefined) update.total_samples = data.total_samples;
    if (data.samples_per_second !== undefined) update.samples_per_second = data.samples_per_second;
    if (data.eta_seconds !== undefined) update.eta_seconds = data.eta_seconds;
    if (data.features_in_heap !== undefined) update.features_in_heap = data.features_in_heap;
    if (data.heap_examples_count !== undefined) update.heap_examples_count = data.heap_examples_count;
    // Map 'message' from WebSocket to 'status_message' in store
    const storeUpdate: Record<string, unknown> = { ...update };
    if (data.message !== undefined) storeUpdate.status_message = data.message;

    updateExtractionById(data.extraction_id, storeUpdate);

    // If extraction completed or failed, refresh to get final statistics
    if (data.status === 'completed' || data.status === 'failed') {
      console.log(`[Extraction WS] ${data.status === 'completed' ? '✅' : '❌'} Extraction ${data.status}, refreshing...`);
      fetchAllExtractions();
    }
  }, [updateExtractionById, fetchAllExtractions]);

  const handleCompleted = useCallback((data: ExtractionProgressEvent) => {
    console.log('[Extraction WS] ✅ Extraction completed:', data);
    updateExtractionById(data.extraction_id, {
      status: 'completed',
      progress: 1.0,
      features_extracted: data.features_extracted,
      total_features: data.total_features,
    });
    // Refresh to get final statistics
    fetchAllExtractions();
  }, [updateExtractionById, fetchAllExtractions]);

  const handleFailed = useCallback((data: ExtractionProgressEvent) => {
    console.log('[Extraction WS] ❌ Extraction failed:', data);
    updateExtractionById(data.extraction_id, {
      status: 'failed',
      error_message: data.error_message,
    });
  }, [updateExtractionById]);

  // Register event handlers
  useEffect(() => {
    console.log('[Extraction WS] Setting up event handlers');

    // Store refs for cleanup
    handlersRef.current = {
      progress: handleProgress,
      completed: handleCompleted,
      failed: handleFailed,
    };

    // Register event handlers
    on('extraction:progress', handleProgress);
    on('extraction:completed', handleCompleted);
    on('extraction:failed', handleFailed);

    console.log('[Extraction WS] ✓ Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Extraction WS] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('extraction:progress', handlers.progress);
      if (handlers.completed) off('extraction:completed', handlers.completed);
      if (handlers.failed) off('extraction:failed', handlers.failed);

      // Clear refs
      handlersRef.current = {
        progress: null,
        completed: null,
        failed: null,
      };
    };
  }, [on, off, handleProgress, handleCompleted, handleFailed]);

  // Subscribe to channels for extractions
  useEffect(() => {
    if (!isConnected) {
      console.log('[Extraction WS] ⚠️ Not connected, skipping channel subscriptions');
      return;
    }

    if (extractionIds.length === 0) {
      console.log('[Extraction WS] No extractions to subscribe to');
      return;
    }

    console.log('[Extraction WS] 📡 Subscribing to', extractionIds.length, 'extraction channel(s)');

    // Subscribe to extraction channel for each extraction
    const subscribedChannels: string[] = [];
    extractionIds.forEach((extractionId) => {
      const channel = `extraction/${extractionId}`;

      console.log(`[Extraction WS] → Subscribing to ${channel}`);
      subscribe(channel);
      subscribedChannels.push(channel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[Extraction WS] 📡 Unsubscribing from extraction channels');
      subscribedChannels.forEach((channel) => {
        console.log(`[Extraction WS] ← Unsubscribing from ${channel}`);
        unsubscribe(channel);
      });
    };
  }, [extractionIds.join(','), isConnected, subscribe, unsubscribe]);
};

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
  status: ExtractionStatus | 'deleted' | 'deleting';
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
  // Deletion-specific fields
  feature_count?: number;
  features_deleted?: number;  // For deletion progress
  // NLP auto-trigger fields (sent with extraction:completed when auto_nlp=true)
  nlp_status?: string;  // 'pending' when NLP is about to start
  nlp_progress?: number;
}

export interface ExtractionDeletedEvent {
  extraction_id: string;
  feature_count: number;
  status: 'deleted';
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
    deleted: ((data: ExtractionDeletedEvent) => void) | null;
    deletionProgress: ((data: ExtractionProgressEvent) => void) | null;
  }>({
    progress: null,
    completed: null,
    failed: null,
    deleted: null,
    deletionProgress: null,
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

    // Build update object including NLP status if present (auto-NLP enabled)
    const update: Record<string, unknown> = {
      status: 'completed',
      progress: 1.0,
      features_extracted: data.features_extracted,
      total_features: data.total_features,
    };

    // If auto-NLP is enabled, the backend sends nlp_status='pending' in the completion event
    // This ensures frontend immediately knows NLP will start and can subscribe to updates
    if (data.nlp_status !== undefined) {
      update.nlp_status = data.nlp_status;
      update.nlp_progress = data.nlp_progress || 0;
      console.log('[Extraction WS] 🔬 NLP auto-triggered, status:', data.nlp_status);
    }

    updateExtractionById(data.extraction_id, update);
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

  const handleDeleted = useCallback((data: ExtractionDeletedEvent) => {
    console.log('[Extraction WS] 🗑️ Extraction deleted:', data);
    // Refresh the extraction list to remove the deleted extraction
    fetchAllExtractions();
  }, [fetchAllExtractions]);

  const handleDeletionProgress = useCallback((data: ExtractionProgressEvent) => {
    console.log('[Extraction WS] 🗑️ Deletion progress:', data);
    // Update extraction with deletion progress
    updateExtractionById(data.extraction_id, {
      status: 'deleting',  // 'deleting' is now a valid ExtractionStatus
      deletion_progress: data.progress,
      deletion_features_deleted: data.features_deleted,
      deletion_total_features: data.total_features ?? undefined,  // Convert null to undefined
      status_message: data.message,
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
      deleted: handleDeleted,
      deletionProgress: handleDeletionProgress,
    };

    // Register event handlers
    on('extraction:progress', handleProgress);
    on('extraction:completed', handleCompleted);
    on('extraction:failed', handleFailed);
    on('extraction:deleted', handleDeleted);
    on('extraction:deletion_progress', handleDeletionProgress);

    console.log('[Extraction WS] ✓ Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Extraction WS] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('extraction:progress', handlers.progress);
      if (handlers.completed) off('extraction:completed', handlers.completed);
      if (handlers.failed) off('extraction:failed', handlers.failed);
      if (handlers.deleted) off('extraction:deleted', handlers.deleted);
      if (handlers.deletionProgress) off('extraction:deletion_progress', handlers.deletionProgress);

      // Clear refs
      handlersRef.current = {
        progress: null,
        completed: null,
        failed: null,
        deleted: null,
        deletionProgress: null,
      };
    };
  }, [on, off, handleProgress, handleCompleted, handleFailed, handleDeleted, handleDeletionProgress]);

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

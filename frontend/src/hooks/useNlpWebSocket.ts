/**
 * NLP Analysis WebSocket Hook
 *
 * React hook for subscribing to NLP analysis progress events via WebSocket.
 * Uses WebSocketContext for proper connection management and event queuing.
 *
 * WebSocket Channels:
 * - nlp_analysis/{extraction_id} - Progress updates for NLP processing
 *
 * Events:
 * - nlp_analysis:progress - Progress update (emitted during NLP processing)
 * - nlp_analysis:completed - NLP analysis completed
 * - nlp_analysis:failed - NLP analysis failed
 *
 * Usage:
 *   useNlpWebSocket(extractionIds);
 *
 *   // Automatically subscribes/unsubscribes based on extractionIds array
 *   // Updates are handled by featuresStore.updateExtractionById()
 */

import { useEffect, useCallback, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useFeaturesStore } from '../stores/featuresStore';
import type { NLPAnalysisProgressEvent, NlpStatus } from '../types/features';

export const useNlpWebSocket = (extractionIds: string[]) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const updateExtractionById = useFeaturesStore((state) => state.updateExtractionById);
  const fetchAllExtractions = useFeaturesStore((state) => state.fetchAllExtractions);

  // Use refs to store handler references for cleanup
  const handlersRef = useRef<{
    progress: ((data: NLPAnalysisProgressEvent) => void) | null;
    completed: ((data: NLPAnalysisProgressEvent) => void) | null;
    failed: ((data: NLPAnalysisProgressEvent) => void) | null;
  }>({
    progress: null,
    completed: null,
    failed: null,
  });

  // Create stable handler using useCallback
  const handleProgress = useCallback((data: NLPAnalysisProgressEvent) => {
    console.log('[NLP WS] 📊 Progress event received:', data);

    // Map NLP status to store nlp_status field
    const nlpStatus: NlpStatus = data.status === 'analyzing' ? 'processing' : data.status;

    updateExtractionById(data.extraction_job_id, {
      nlp_status: nlpStatus,
      nlp_progress: data.progress,
      nlp_processed_count: data.features_analyzed,
    });
  }, [updateExtractionById]);

  const handleCompleted = useCallback((data: NLPAnalysisProgressEvent) => {
    console.log('[NLP WS] ✅ NLP analysis completed:', data);
    updateExtractionById(data.extraction_job_id, {
      nlp_status: 'completed',
      nlp_progress: 1.0,
      nlp_processed_count: data.features_analyzed,
    });
    // Refresh to get final state
    fetchAllExtractions();
  }, [updateExtractionById, fetchAllExtractions]);

  const handleFailed = useCallback((data: NLPAnalysisProgressEvent) => {
    console.log('[NLP WS] ❌ NLP analysis failed:', data);
    updateExtractionById(data.extraction_job_id, {
      nlp_status: 'failed',
      nlp_error_message: data.error || data.message,
    });
  }, [updateExtractionById]);

  // Register event handlers
  useEffect(() => {
    console.log('[NLP WS] Setting up event handlers');

    // Store refs for cleanup
    handlersRef.current = {
      progress: handleProgress,
      completed: handleCompleted,
      failed: handleFailed,
    };

    // Register event handlers - NLP events use different prefix
    on('nlp_analysis:progress', handleProgress);
    on('nlp_analysis:completed', handleCompleted);
    on('nlp_analysis:failed', handleFailed);

    console.log('[NLP WS] ✓ Event handlers registered');

    // Cleanup
    return () => {
      console.log('[NLP WS] Cleaning up event handlers');
      const handlers = handlersRef.current;
      if (handlers.progress) off('nlp_analysis:progress', handlers.progress);
      if (handlers.completed) off('nlp_analysis:completed', handlers.completed);
      if (handlers.failed) off('nlp_analysis:failed', handlers.failed);

      // Clear refs
      handlersRef.current = {
        progress: null,
        completed: null,
        failed: null,
      };
    };
  }, [on, off, handleProgress, handleCompleted, handleFailed]);

  // Subscribe to channels for extractions that need NLP processing
  useEffect(() => {
    if (!isConnected) {
      console.log('[NLP WS] ⚠️ Not connected, skipping channel subscriptions');
      return;
    }

    if (extractionIds.length === 0) {
      console.log('[NLP WS] No extractions to subscribe to');
      return;
    }

    console.log('[NLP WS] 📡 Subscribing to', extractionIds.length, 'NLP analysis channel(s)');

    // Subscribe to NLP analysis channel for each extraction
    const subscribedChannels: string[] = [];
    extractionIds.forEach((extractionId) => {
      const channel = `nlp_analysis/${extractionId}`;

      console.log(`[NLP WS] → Subscribing to ${channel}`);
      subscribe(channel);
      subscribedChannels.push(channel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[NLP WS] 📡 Unsubscribing from NLP analysis channels');
      subscribedChannels.forEach((channel) => {
        console.log(`[NLP WS] ← Unsubscribing from ${channel}`);
        unsubscribe(channel);
      });
    };
  }, [extractionIds.join(','), isConnected, subscribe, unsubscribe]);
};

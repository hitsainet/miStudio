/**
 * Labeling WebSocket Hook
 *
 * React hook for subscribing to labeling progress events via WebSocket.
 * Uses WebSocketContext for proper connection management and event queuing.
 *
 * WebSocket Channels:
 * - labeling/{labeling_job_id}/progress - Progress updates
 *
 * Events:
 * - labeling:started - Labeling started
 * - labeling:progress - Progress update
 * - labeling:completed - Labeling completed
 * - labeling:failed - Labeling failed
 *
 * Usage:
 *   useLabelingWebSocket(labelingJobIds);
 *
 *   // Automatically subscribes/unsubscribes based on labelingJobIds array
 *   // Updates are handled by labelingStore.updateLabelingStatus()
 */

import { useEffect, useRef, useMemo } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useLabelingStore } from '../stores/labelingStore';
import { LabelingStatus } from '../types/labeling';
import type {
  LabelingProgressEvent,
  LabelingCompletedEvent,
  LabelingFailedEvent,
} from '../types/labeling';

export const useLabelingWebSocket = (labelingJobIds: string[]) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const { updateLabelingStatus } = useLabelingStore();
  const handlersRegisteredRef = useRef(false);

  // Set up global event handlers (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[Labeling WS] Setting up labeling event handlers');

    // Handler for 'labeling:started' events
    const handleStarted = (data: any) => {
      console.log('[Labeling WS] Labeling started:', data);
      updateLabelingStatus(data.labeling_job_id, {
        status: LabelingStatus.LABELING,
        progress: 0,
        features_labeled: 0,
      });
    };

    // Handler for 'labeling:progress' events
    const handleProgress = (data: LabelingProgressEvent) => {
      console.log('[Labeling WS] Progress event:', data);
      updateLabelingStatus(data.labeling_job_id, {
        status: data.status,
        progress: data.progress,
        features_labeled: data.features_labeled,
        total_features: data.total_features,
      });
    };

    // Handler for 'labeling:completed' events
    const handleCompleted = (data: LabelingCompletedEvent) => {
      console.log('[Labeling WS] Labeling completed:', data);
      updateLabelingStatus(data.labeling_job_id, {
        status: LabelingStatus.COMPLETED,
        progress: 1.0,
        statistics: data.statistics,
        completed_at: new Date().toISOString(),
      });
    };

    // Handler for 'labeling:failed' events
    const handleFailed = (data: LabelingFailedEvent) => {
      console.log('[Labeling WS] Labeling failed:', data);
      updateLabelingStatus(data.labeling_job_id, {
        status: LabelingStatus.FAILED,
        error_message: data.error_message,
      });
    };

    // Register event handlers
    on('labeling:started', handleStarted);
    on('labeling:progress', handleProgress);
    on('labeling:completed', handleCompleted);
    on('labeling:failed', handleFailed);

    handlersRegisteredRef.current = true;
    console.log('[Labeling WS] Event handlers registered');

    // Cleanup
    return () => {
      console.log('[Labeling WS] Cleaning up event handlers');
      off('labeling:started', handleStarted);
      off('labeling:progress', handleProgress);
      off('labeling:completed', handleCompleted);
      off('labeling:failed', handleFailed);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, updateLabelingStatus]);

  // Create a stable key from labelingJobIds to prevent unnecessary re-subscriptions
  const labelingJobIdsKey = useMemo(
    () => labelingJobIds.sort().join(','),
    [labelingJobIds.join(',')]
  );

  // Subscribe to channels for active labeling jobs
  useEffect(() => {
    if (!isConnected) {
      console.log('[Labeling WS] Not connected, skipping channel subscriptions');
      return;
    }

    if (labelingJobIds.length === 0) {
      console.log('[Labeling WS] No labeling jobs to subscribe to');
      return;
    }

    console.log(
      '[Labeling WS] Subscribing to',
      labelingJobIds.length,
      'labeling channels'
    );

    // Subscribe to progress channel for each labeling job
    labelingJobIds.forEach((labelingJobId) => {
      const progressChannel = `labeling/${labelingJobId}/progress`;

      console.log(`[Labeling WS] Subscribing to ${progressChannel}`);
      subscribe(progressChannel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[Labeling WS] Unsubscribing from labeling channels');
      labelingJobIds.forEach((labelingJobId) => {
        unsubscribe(`labeling/${labelingJobId}/progress`);
      });
    };
  }, [labelingJobIdsKey, isConnected, subscribe, unsubscribe]);
};

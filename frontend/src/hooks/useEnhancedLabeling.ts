/**
 * useEnhancedLabeling hook
 *
 * Manages the lifecycle of a single-feature two-pass enhanced labeling job:
 *   - On mount: GET latest job to restore any in-progress / completed state
 *   - start():  POST to queue a new job, subscribe to WebSocket channel
 *   - WebSocket events drive phase/progress/completion state
 *   - completedLabel is populated when the job finishes — caller uses it to
 *     auto-populate the edit form fields
 *
 * WebSocket channel:  enhanced_labeling/{job_id}
 * Events:
 *   enhanced_labeling:progress   → update phase + examples_completed
 *   enhanced_labeling:completed  → populate completedLabel, unsubscribe
 *   enhanced_labeling:failed     → set error, unsubscribe
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { startEnhancedLabeling, getLatestEnhancedLabelingJob } from '../api/enhancedLabeling';
import { useFeaturesStore } from '../stores/featuresStore';
import type {
  EnhancedLabelingJob,
  EnhancedLabelingCompletedEvent,
  EnhancedLabelingFailedEvent,
  EnhancedLabelingProgressEvent,
} from '../types/enhancedLabeling';

export interface CompletedLabel {
  name: string;
  category: string;
  description: string;
  notes: string;
}

export interface UseEnhancedLabelingResult {
  job: EnhancedLabelingJob | null;
  /** Human-readable progress phrase for display in the modal */
  progressPhrase: string | null;
  /** Populated when job completes — caller populates edit fields from this */
  completedLabel: CompletedLabel | null;
  /** Set when the POST /label/enhanced API call itself fails (e.g. settings not configured) */
  startError: string | null;
  start: () => Promise<void>;
  reset: () => void;
}

export function useEnhancedLabeling(featureId: string): UseEnhancedLabelingResult {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const setStarColor = useFeaturesStore((s) => s.setStarColor);
  const patchFeatureLocally = useFeaturesStore((s) => s.patchFeatureLocally);

  const [job, setJob] = useState<EnhancedLabelingJob | null>(null);
  const [completedLabel, setCompletedLabel] = useState<CompletedLabel | null>(null);
  const [startError, setStartError] = useState<string | null>(null);
  const subscribedChannelRef = useRef<string | null>(null);

  // ── helpers ────────────────────────────────────────────────────────────────

  const _subscribeToJob = useCallback(
    (jobId: string) => {
      const channel = `enhanced_labeling/${jobId}`;
      if (subscribedChannelRef.current === channel) return;
      if (subscribedChannelRef.current) {
        unsubscribe(subscribedChannelRef.current);
      }
      subscribe(channel);
      subscribedChannelRef.current = channel;
    },
    [subscribe, unsubscribe]
  );

  const _unsubscribeFromJob = useCallback(() => {
    if (subscribedChannelRef.current) {
      unsubscribe(subscribedChannelRef.current);
      subscribedChannelRef.current = null;
    }
  }, [unsubscribe]);

  // ── restore state on mount ─────────────────────────────────────────────────

  useEffect(() => {
    let cancelled = false;
    getLatestEnhancedLabelingJob(featureId).then((latestJob) => {
      if (cancelled || !latestJob) return;
      setJob(latestJob);
      if (latestJob.status === 'queued' || latestJob.status === 'running') {
        _subscribeToJob(latestJob.id);
        // Sync store star color — DB is authoritative, but store may be stale if modal was closed
        setStarColor(featureId, 'purple').catch(() => {});
      }
    });
    return () => {
      cancelled = true;
    };
  }, [featureId]);

  // ── (re-)subscribe when isConnected changes for an active job ──────────────

  useEffect(() => {
    if (!isConnected || !job) return;
    if (job.status === 'queued' || job.status === 'running') {
      _subscribeToJob(job.id);
    }
  }, [isConnected, job?.id, job?.status]);

  // ── WebSocket event handlers ───────────────────────────────────────────────

  useEffect(() => {
    const handleProgress = (data: EnhancedLabelingProgressEvent) => {
      setJob((prev) =>
        prev && prev.id === data.job_id
          ? {
              ...prev,
              status: 'running',
              phase: data.phase,
              examples_completed: data.examples_completed,
            }
          : prev
      );
    };

    const handleCompleted = (data: EnhancedLabelingCompletedEvent) => {
      setJob((prev) =>
        prev && prev.id === data.job_id
          ? { ...prev, status: 'completed', phase: null }
          : prev
      );
      const label = {
        name: data.name,
        category: data.category,
        description: data.description,
        notes: data.notes,
      };
      setCompletedLabel(label);
      // Immediately reflect new label in feature list rows and open modal view
      patchFeatureLocally(featureId, { ...label, label_source: 'enhanced_llm' });
      // Mark feature aqua — permanent signal that enhanced labeling completed
      setStarColor(featureId, 'aqua').catch(() => {});
      _unsubscribeFromJob();
    };

    const handleFailed = (data: EnhancedLabelingFailedEvent) => {
      setJob((prev) =>
        prev && prev.id === data.job_id
          ? { ...prev, status: 'failed', phase: null, error_message: data.error_message }
          : prev
      );
      _unsubscribeFromJob();
    };

    on('enhanced_labeling:progress', handleProgress);
    on('enhanced_labeling:completed', handleCompleted);
    on('enhanced_labeling:failed', handleFailed);

    return () => {
      off('enhanced_labeling:progress', handleProgress);
      off('enhanced_labeling:completed', handleCompleted);
      off('enhanced_labeling:failed', handleFailed);
    };
  }, [on, off, _unsubscribeFromJob]);

  // ── cleanup on unmount ─────────────────────────────────────────────────────

  useEffect(() => {
    return () => {
      _unsubscribeFromJob();
    };
  }, []);

  // ── actions ────────────────────────────────────────────────────────────────

  const start = useCallback(async () => {
    setCompletedLabel(null);
    setStartError(null);
    try {
      const newJob = await startEnhancedLabeling(featureId);
      setJob(newJob);
      _subscribeToJob(newJob.id);
      // Mark feature purple so it's visible in starred filter while job runs
      setStarColor(featureId, 'purple').catch(() => {});
    } catch (err: any) {
      setStartError(err?.message ?? 'Failed to start enhanced labeling');
    }
  }, [featureId, _subscribeToJob, setStarColor]);

  const reset = useCallback(() => {
    _unsubscribeFromJob();
    setJob(null);
    setCompletedLabel(null);
    setStartError(null);
  }, [_unsubscribeFromJob]);

  // ── progress phrase ────────────────────────────────────────────────────────

  let progressPhrase: string | null = null;
  if (job?.status === 'queued') {
    progressPhrase = 'Queued...';
  } else if (job?.status === 'running') {
    if (job.phase === 'pass1') {
      progressPhrase = `Summarizing example ${job.examples_completed} / ${job.examples_total}...`;
    } else if (job.phase === 'pass2') {
      progressPhrase = 'Synthesizing label...';
    } else {
      progressPhrase = 'Running...';
    }
  }

  return { job, progressPhrase, completedLabel, startError, start, reset };
}

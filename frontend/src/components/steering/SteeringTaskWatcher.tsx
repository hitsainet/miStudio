/**
 * Delivers steering task completions to the store, whichever panel is on screen.
 *
 * INCIDENT 2026-10-28. A batch, a sweep and a blended run are loops in the
 * steering store that wait on a per-task resolver; a single compare waits for
 * its result to land. The only things that resolve them are the
 * `steering/{task_id}` WebSocket subscription and the `/async/result` fallback
 * poll. Both used to live inside SteeringPanel, and App renders one panel at a
 * time: an operator who opened another panel while a batch ran unmounted them,
 * and every remaining prompt waited out its 5-minute timeout and was cancelled
 * although the server had finished it in ~50 s.
 *
 * This component is rendered by App OUTSIDE the panel switch, so it lives as
 * long as the page. It owns:
 *   - the WebSocket subscription for the store's current task id;
 *   - the fallback poll (the socket can be down or miss an event);
 *   - the one recovery of a task that was in flight when the page was reloaded.
 *
 * Every completion is checked against the store's task id AT DELIVERY, not the
 * id this component last rendered with: a late event or poll answer for the
 * previous task of a batch must never be handed to the next task's resolver.
 */

import { useCallback, useEffect, useRef } from 'react';
import { getTaskResult } from '../../api/steering';
import { useSteeringStore } from '../../stores/steeringStore';
import { useSteeringWebSocket } from '../../hooks/useSteeringWebSocket';
import type {
  SteeringCompletedEvent,
  SteeringFailedEvent,
  SteeringProgressEvent,
} from '../../hooks/useSteeringWebSocket';

/** Poll the result endpoint this often while a task is in flight… */
export const STEERING_POLL_INTERVAL_MS = 5_000;
/** …but only once the socket has been silent this long. */
export const STEERING_WS_SILENCE_MS = 15_000;

const isCurrentTask = (taskId: string | null | undefined): boolean =>
  !!taskId && taskId === useSteeringStore.getState().taskId;

export function SteeringTaskWatcher() {
  const taskId = useSteeringStore((s) => s.taskId);
  const hasHydrated = useSteeringStore((s) => s._hasHydrated);
  // Any of the store's waits: a single compare, a batch, a sweep, a blended run.
  const inFlight = useSteeringStore(
    (s) => s.isGenerating || s.isSweeping || s.isCombinedGenerating || !!s.batchState?.isRunning,
  );
  const lastEventAt = useRef<number>(0); // set when a task starts (the poll effect)

  // A task in flight when the page was reloaded: recovered once per page, not
  // per visit to the Steering panel (a re-run mid-batch would race the loop).
  const recovered = useRef(false);
  useEffect(() => {
    if (!hasHydrated || recovered.current) return;
    recovered.current = true;
    void useSteeringStore.getState().recoverActiveTask();
  }, [hasHydrated]);

  const onProgress = useCallback((data: SteeringProgressEvent) => {
    if (!isCurrentTask(data.task_id)) return;
    lastEventAt.current = Date.now();
    useSteeringStore
      .getState()
      .handleAsyncProgress(data.percent, data.message, data.current_feature, data.current_strength);
  }, []);

  const onCompleted = useCallback((data: SteeringCompletedEvent) => {
    if (!isCurrentTask(data.task_id)) return;
    lastEventAt.current = Date.now();
    useSteeringStore.getState().handleAsyncCompleted(data.result);
  }, []);

  const onFailed = useCallback((data: SteeringFailedEvent) => {
    if (!isCurrentTask(data.task_id)) return;
    lastEventAt.current = Date.now();
    useSteeringStore.getState().handleAsyncFailed(data.error);
  }, []);

  useSteeringWebSocket(taskId, { onProgress, onCompleted, onFailed });

  // Fallback poll: the socket can be down, or the event can be emitted before
  // the subscription reaches the server.
  useEffect(() => {
    if (!taskId || !inFlight) return;
    lastEventAt.current = Date.now();

    const poll = async () => {
      if (Date.now() - lastEventAt.current < STEERING_WS_SILENCE_MS) return;
      try {
        const response = await getTaskResult(taskId);
        // The store may have moved on (next batch prompt) while the request was out.
        if (!isCurrentTask(taskId)) return;
        const status = response.status.status;
        if (status === 'success' && response.result) {
          useSteeringStore.getState().handleAsyncCompleted(response.result);
        } else if (status === 'failure') {
          useSteeringStore.getState().handleAsyncFailed(response.status.error || 'Task failed');
        } else if (status === 'revoked') {
          useSteeringStore.getState().handleAsyncFailed('Task was cancelled');
        }
      } catch (error) {
        console.warn('[SteeringTaskWatcher] Fallback poll error:', error);
      }
    };

    const intervalId = setInterval(poll, STEERING_POLL_INTERVAL_MS);
    return () => clearInterval(intervalId);
  }, [taskId, inFlight]);

  return null;
}

/**
 * Shared polling utilities for resource status monitoring.
 *
 * This module provides reusable polling functionality for Zustand stores
 * to monitor async operations like dataset downloads, model loading, etc.
 */

/**
 * Configuration for resource polling.
 */
export interface PollingConfig<T> {
  /**
   * Function to fetch the current resource status.
   * Should return the resource object.
   */
  fetchStatus: () => Promise<T>;

  /**
   * Callback invoked when a status update is received.
   * @param resource - The updated resource state
   */
  onUpdate: (resource: T) => void;

  /**
   * Callback invoked when polling completes (terminal state reached).
   * @param resource - The final resource state
   */
  onComplete?: (resource: T) => void;

  /**
   * Callback invoked if polling times out or encounters an error.
   * @param error - Error message describing what went wrong
   */
  onError?: (error: string) => void;

  /**
   * Function to check if the resource has reached a terminal state.
   * @param resource - The resource to check
   * @returns true if polling should stop
   */
  isTerminal: (resource: T) => boolean;

  /**
   * Polling interval in milliseconds.
   * @default 500 (0.5 seconds)
   */
  interval?: number;

  /**
   * Maximum number of polls before giving up.
   * @default 100
   */
  maxPolls?: number;

  /**
   * Resource identifier for logging.
   */
  resourceId: string;

  /**
   * Resource type name for logging.
   */
  resourceType: string;
}

/**
 * Starts polling a resource's status until it reaches a terminal state.
 *
 * @template T - The type of the resource being polled
 * @param config - Polling configuration
 * @returns Function to stop polling
 *
 * @example
 * ```typescript
 * const stopPolling = startPolling<Dataset>({
 *   fetchStatus: () => getDataset(datasetId),
 *   onUpdate: (dataset) => {
 *     updateStoreWithDataset(dataset);
 *   },
 *   onComplete: (dataset) => {
 *     console.log('Dataset ready:', dataset);
 *   },
 *   onError: (error) => {
 *     console.error('Polling failed:', error);
 *   },
 *   isTerminal: (dataset) =>
 *     dataset.status !== 'downloading' && dataset.status !== 'processing',
 *   interval: 500,
 *   maxPolls: 50,
 *   resourceId: datasetId,
 *   resourceType: 'dataset',
 * });
 *
 * // Later, stop polling if needed
 * stopPolling();
 * ```
 */
/** How many consecutive fetch failures end a poll. One transient 502
 *  must not, which is MIS-E2E-125(1); a genuinely dead endpoint must. */
const MAX_CONSECUTIVE_ERRORS = 5;

export function startPolling<T>(config: PollingConfig<T>): () => void {
  const {
    fetchStatus,
    onUpdate,
    onComplete,
    onError,
    isTerminal,
    interval = 500,
    maxPolls = 100,
    resourceId,
    resourceType,
  } = config;

  let pollCount = 0;
  let intervalId: number | null = null;
  // MIS-E2E-125(2): a slow response could resolve AFTER polling stopped and
  // call `onUpdate` with non-terminal state, re-showing a finished job as
  // in-progress. `stopped` is checked on every path that reports.
  let stopped = false;
  // ...and a slow response could still be in flight when the next tick fires,
  // so two overlapping polls could report out of order.
  let inFlight = false;
  // MIS-E2E-125(1): tolerate transient failures. A single 502 during a
  // ten-minute model download used to end polling permanently, and the only
  // caller discards the returned handle, so nothing could restart it — the
  // download completed while the UI showed it in progress forever.
  let consecutiveErrors = 0;

  console.log(`[Polling] Starting polling for ${resourceType} ${resourceId}`, {
    interval,
    maxPolls,
  });

  const poll = async () => {
    if (stopped || inFlight) return;
    inFlight = true;
    pollCount++;

    console.log(`[Polling] Poll ${pollCount}/${maxPolls} for ${resourceType} ${resourceId}`);

    try {
      // Fetch current status
      const resource = await fetchStatus();

      consecutiveErrors = 0;

      // Check if resource doesn't exist (was deleted or never existed)
      if (!resource) {
        console.warn(`[Polling] Resource not found for ${resourceType} ${resourceId}, stopping polling`);
        if (intervalId !== null) {
          window.clearInterval(intervalId);
          intervalId = null;
        }
        if (onError) {
          onError(`Resource ${resourceType} ${resourceId} not found`);
        }
        return;
      }

      // Notify update callback — unless polling has been stopped since this
      // request went out (MIS-E2E-125(2)).
      if (stopped) return;
      onUpdate(resource);

      // Check if we've reached a terminal state
      if (isTerminal(resource)) {
        console.log(`[Polling] Terminal state reached for ${resourceType} ${resourceId}`);

        // Stop polling
        if (intervalId !== null) {
          window.clearInterval(intervalId);
          intervalId = null;
        }

        // Notify completion
        if (onComplete) {
          onComplete(resource);
        }

        return;
      }

      // Check if we've exceeded max polls
      if (pollCount >= maxPolls) {
        console.warn(`[Polling] Max polls reached for ${resourceType} ${resourceId}`, {
          count: pollCount,
          max: maxPolls,
        });

        // Stop polling
        if (intervalId !== null) {
          window.clearInterval(intervalId);
          intervalId = null;
        }

        // Notify error
        if (onError) {
          onError(
            `Polling timeout: ${resourceType} did not reach terminal state after ${maxPolls} attempts`
          );
        }

        return;
      }
    } catch (error) {
      consecutiveErrors++;
      console.error(
        `[Polling] Fetch error ${consecutiveErrors}/${MAX_CONSECUTIVE_ERRORS} ` +
          `for ${resourceType} ${resourceId}:`,
        error
      );

      // MIS-E2E-125(1): only give up after a RUN of failures. One 502 in the
      // middle of a long download is not a reason to stop watching it.
      if (consecutiveErrors < MAX_CONSECUTIVE_ERRORS) {
        return;
      }

      if (intervalId !== null) {
        window.clearInterval(intervalId);
        intervalId = null;
      }
      stopped = true;

      if (onError) {
        const errorMessage =
          error instanceof Error ? error.message : String(error);
        onError(
          `Polling failed after ${MAX_CONSECUTIVE_ERRORS} consecutive errors: ${errorMessage}`
        );
      }
    } finally {
      inFlight = false;
    }
  };

  // Start polling interval
  intervalId = window.setInterval(poll, interval);

  // Return stop function
  return () => {
    if (intervalId !== null) {
      console.log(`[Polling] Stopping polling for ${resourceType} ${resourceId}`);
      window.clearInterval(intervalId);
      intervalId = null;
      pollCount = 0;
    }
    // Set even if the interval was already cleared: an in-flight request may
    // still resolve, and it must not report after the caller stopped us.
    stopped = true;
  };
}

/**
 * Shared polling hook for resource status monitoring.
 *
 * This hook provides a reusable polling mechanism for monitoring async operations
 * like dataset downloads, model loading, tokenization, etc.
 */

import { useRef, useCallback } from 'react';

/**
 * Configuration for the polling hook.
 */
export interface PollingConfig<T> {
  /**
   * Function to fetch the current status of the resource.
   * Should return the resource object with a status field.
   */
  fetchStatus: () => Promise<T>;

  /**
   * Callback invoked when the resource reaches a terminal state.
   * @param resource - The final resource state
   */
  onComplete?: (resource: T) => void;

  /**
   * Callback invoked if polling times out or encounters an error.
   * @param error - Error message describing what went wrong
   */
  onError?: (error: string) => void;

  /**
   * List of status values that indicate the operation is complete.
   * Polling will stop when the resource status matches one of these.
   *
   * @example ['ready', 'error', 'cancelled']
   */
  terminalStates: string[];

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
   * Function to extract the status value from the resource.
   * Useful if the status field has a different name or structure.
   *
   * @default (resource) => resource.status
   */
  getStatus?: (resource: T) => string;
}

/**
 * Hook for polling a resource's status until it reaches a terminal state.
 *
 * @template T - The type of the resource being polled (must have a status field)
 * @param config - Polling configuration
 * @returns Object with start and stop functions
 *
 * @example
 * ```typescript
 * const { startPolling, stopPolling } = usePolling<Model>({
 *   fetchStatus: () => getModel(modelId),
 *   onComplete: (model) => {
 *     console.log('Model ready:', model);
 *   },
 *   onError: (error) => {
 *     console.error('Polling failed:', error);
 *   },
 *   terminalStates: ['ready', 'error'],
 *   interval: 500,
 *   maxPolls: 100,
 * });
 *
 * // Start polling after initiating download
 * await downloadModel({ repo_id: 'gpt2' });
 * startPolling();
 *
 * // Stop polling if user navigates away
 * useEffect(() => {
 *   return () => stopPolling();
 * }, []);
 * ```
 */
export function usePolling<T extends Record<string, any>>(
  config: PollingConfig<T>
) {
  const {
    fetchStatus,
    onComplete,
    onError,
    terminalStates,
    interval = 500,
    maxPolls = 100,
    getStatus = (resource: T) => resource.status,
  } = config;

  // Use ref to store interval ID so it persists across renders
  const intervalRef = useRef<number | null>(null);
  const pollCountRef = useRef<number>(0);

  /**
   * Start polling for status updates.
   *
   * This function will repeatedly call fetchStatus at the specified interval
   * until one of the following occurs:
   * - The resource reaches a terminal state
   * - Max polls is reached
   * - An error occurs
   * - stopPolling is called
   */
  const startPolling = useCallback(() => {
    // Clear any existing interval
    if (intervalRef.current !== null) {
      window.clearInterval(intervalRef.current);
    }

    // Reset poll count
    pollCountRef.current = 0;

    console.log('[usePolling] Starting polling', {
      interval,
      maxPolls,
      terminalStates,
    });

    // Create polling interval
    intervalRef.current = window.setInterval(async () => {
      pollCountRef.current += 1;

      console.log('[usePolling] Poll attempt', {
        count: pollCountRef.current,
        max: maxPolls,
      });

      try {
        // Fetch current status
        const resource = await fetchStatus();
        const status = getStatus(resource);

        console.log('[usePolling] Fetched status:', status);

        // Check if we've reached a terminal state
        if (terminalStates.includes(status)) {
          console.log('[usePolling] Terminal state reached:', status);

          // Stop polling
          if (intervalRef.current !== null) {
            window.clearInterval(intervalRef.current);
            intervalRef.current = null;
          }

          // Notify completion
          if (onComplete) {
            onComplete(resource);
          }

          return;
        }

        // Check if we've exceeded max polls
        if (pollCountRef.current >= maxPolls) {
          console.warn('[usePolling] Max polls reached', {
            count: pollCountRef.current,
            max: maxPolls,
          });

          // Stop polling
          if (intervalRef.current !== null) {
            window.clearInterval(intervalRef.current);
            intervalRef.current = null;
          }

          // Notify error
          if (onError) {
            onError(
              `Polling timeout: resource did not reach terminal state after ${maxPolls} attempts`
            );
          }

          return;
        }
      } catch (error) {
        console.error('[usePolling] Fetch error:', error);

        // Stop polling on error
        if (intervalRef.current !== null) {
          window.clearInterval(intervalRef.current);
          intervalRef.current = null;
        }

        // Notify error
        if (onError) {
          const errorMessage =
            error instanceof Error ? error.message : String(error);
          onError(`Polling failed: ${errorMessage}`);
        }
      }
    }, interval);
  }, [
    fetchStatus,
    onComplete,
    onError,
    terminalStates,
    interval,
    maxPolls,
    getStatus,
  ]);

  /**
   * Stop polling.
   *
   * Call this function to manually stop polling before a terminal state is reached.
   * Useful when a component unmounts or when the user cancels an operation.
   */
  const stopPolling = useCallback(() => {
    if (intervalRef.current !== null) {
      console.log('[usePolling] Stopping polling');
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
      pollCountRef.current = 0;
    }
  }, []);

  return {
    startPolling,
    stopPolling,
  };
}

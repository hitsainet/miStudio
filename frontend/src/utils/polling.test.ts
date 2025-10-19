/**
 * Unit tests for polling utility.
 *
 * This module tests the shared polling utility used by Zustand stores
 * to monitor resource status, including configuration, callbacks, and cleanup.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { startPolling, PollingConfig } from './polling';

describe('polling utility', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.clearAllTimers();
    vi.useRealTimers();
    vi.restoreAllMocks();
  });

  describe('startPolling', () => {
    it('should start polling and call fetchStatus at interval', async () => {
      const mockResource = { id: '1', status: 'processing', progress: 50 };
      const fetchStatus = vi.fn().mockResolvedValue(mockResource);
      const onUpdate = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      // First poll happens immediately via setInterval
      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(1);
      expect(onUpdate).toHaveBeenCalledWith(mockResource);
      expect(isTerminal).toHaveBeenCalledWith(mockResource);

      // Second poll
      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(2);
      expect(onUpdate).toHaveBeenCalledTimes(2);
    });

    it('should stop polling when terminal state reached', async () => {
      const mockResource = { id: '1', status: 'ready', progress: 100 };
      const fetchStatus = vi.fn().mockResolvedValue(mockResource);
      const onUpdate = vi.fn();
      const onComplete = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(true);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        onComplete,
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(1);
      expect(onUpdate).toHaveBeenCalledWith(mockResource);
      expect(isTerminal).toHaveBeenCalledWith(mockResource);
      expect(onComplete).toHaveBeenCalledWith(mockResource);

      // No more polls after terminal state
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchStatus).toHaveBeenCalledTimes(1);
    });

    it('should stop polling after max polls reached', async () => {
      const mockResource = { id: '1', status: 'processing', progress: 50 };
      const fetchStatus = vi.fn().mockResolvedValue(mockResource);
      const onUpdate = vi.fn();
      const onError = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        onError,
        isTerminal,
        interval: 100,
        maxPolls: 3,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      // Poll 1
      await vi.advanceTimersByTimeAsync(100);
      expect(fetchStatus).toHaveBeenCalledTimes(1);

      // Poll 2
      await vi.advanceTimersByTimeAsync(100);
      expect(fetchStatus).toHaveBeenCalledTimes(2);

      // Poll 3 (max reached)
      await vi.advanceTimersByTimeAsync(100);
      expect(fetchStatus).toHaveBeenCalledTimes(3);
      expect(onError).toHaveBeenCalledWith(
        'Polling timeout: test did not reach terminal state after 3 attempts'
      );

      // No more polls after max
      await vi.advanceTimersByTimeAsync(100);
      expect(fetchStatus).toHaveBeenCalledTimes(3);
    });

    it('should handle fetch errors and stop polling', async () => {
      const fetchError = new Error('Fetch failed');
      const fetchStatus = vi.fn().mockRejectedValue(fetchError);
      const onUpdate = vi.fn();
      const onError = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<any> = {
        fetchStatus,
        onUpdate,
        onError,
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(1);
      expect(onUpdate).not.toHaveBeenCalled();
      expect(onError).toHaveBeenCalledWith('Polling failed: Fetch failed');

      // No more polls after error
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchStatus).toHaveBeenCalledTimes(1);
    });

    it('should return a stop function that stops polling', async () => {
      const mockResource = { id: '1', status: 'processing', progress: 50 };
      const fetchStatus = vi.fn().mockResolvedValue(mockResource);
      const onUpdate = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      const stopPolling = startPolling(config);

      // First poll
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchStatus).toHaveBeenCalledTimes(1);

      // Stop polling
      stopPolling();

      // No more polls after stopping
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchStatus).toHaveBeenCalledTimes(1);
    });

    it('should use default interval and maxPolls', async () => {
      const mockResource = { id: '1', status: 'processing', progress: 50 };
      const fetchStatus = vi.fn().mockResolvedValue(mockResource);
      const onUpdate = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        isTerminal,
        // No interval or maxPolls specified - should use defaults (500ms, 100 polls)
        resourceId: '1',
        resourceType: 'test',
      };

      const stopPolling = startPolling(config);

      // Default interval is 500ms
      await vi.advanceTimersByTimeAsync(500);
      expect(fetchStatus).toHaveBeenCalledTimes(1);

      // Stop before hitting max polls
      stopPolling();
    });

    it('should not call onComplete if not provided', async () => {
      const mockResource = { id: '1', status: 'ready', progress: 100 };
      const fetchStatus = vi.fn().mockResolvedValue(mockResource);
      const onUpdate = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(true);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        // No onComplete provided
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(1);
      expect(onUpdate).toHaveBeenCalledWith(mockResource);
      expect(isTerminal).toHaveBeenCalledWith(mockResource);
      // No error should be thrown
    });

    it('should not call onError if not provided', async () => {
      const fetchError = new Error('Fetch failed');
      const fetchStatus = vi.fn().mockRejectedValue(fetchError);
      const onUpdate = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<any> = {
        fetchStatus,
        onUpdate,
        // No onError provided
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(1);
      expect(onUpdate).not.toHaveBeenCalled();
      // No error should be thrown
    });

    it('should handle non-Error exceptions', async () => {
      const fetchStatus = vi.fn().mockRejectedValue('String error');
      const onUpdate = vi.fn();
      const onError = vi.fn();
      const isTerminal = vi.fn().mockReturnValue(false);

      const config: PollingConfig<any> = {
        fetchStatus,
        onUpdate,
        onError,
        isTerminal,
        interval: 500,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      await vi.advanceTimersByTimeAsync(500);

      expect(fetchStatus).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenCalledWith('Polling failed: String error');
    });

    it('should track poll count correctly across multiple polls', async () => {
      let pollCount = 0;
      const mockResource = { id: '1', status: 'processing', progress: 50 };
      const fetchStatus = vi.fn().mockImplementation(async () => {
        pollCount++;
        return { ...mockResource, progress: pollCount * 10 };
      });
      const onUpdate = vi.fn();
      const isTerminal = vi.fn().mockImplementation((resource) => resource.progress >= 50);

      const config: PollingConfig<typeof mockResource> = {
        fetchStatus,
        onUpdate,
        isTerminal,
        interval: 100,
        maxPolls: 10,
        resourceId: '1',
        resourceType: 'test',
      };

      startPolling(config);

      // Poll 1 (progress 10)
      await vi.advanceTimersByTimeAsync(100);
      expect(onUpdate).toHaveBeenCalledWith({ ...mockResource, progress: 10 });

      // Poll 2 (progress 20)
      await vi.advanceTimersByTimeAsync(100);
      expect(onUpdate).toHaveBeenCalledWith({ ...mockResource, progress: 20 });

      // Poll 3 (progress 30)
      await vi.advanceTimersByTimeAsync(100);
      expect(onUpdate).toHaveBeenCalledWith({ ...mockResource, progress: 30 });

      // Poll 4 (progress 40)
      await vi.advanceTimersByTimeAsync(100);
      expect(onUpdate).toHaveBeenCalledWith({ ...mockResource, progress: 40 });

      // Poll 5 (progress 50 - terminal)
      await vi.advanceTimersByTimeAsync(100);
      expect(onUpdate).toHaveBeenCalledWith({ ...mockResource, progress: 50 });

      // Should stop at terminal state
      expect(fetchStatus).toHaveBeenCalledTimes(5);
    });
  });
});

/**
 * The header stats must keep updating on every page.
 *
 * REPORTED 2026-07-27: "the small monitoring elements that are on the top of
 * every page do not update on their own now."
 *
 * The cause is NOT the extraction-refresh change it was noticed alongside. It
 * is that connection state was used as a proxy for "metrics are arriving":
 *
 *   - `useSystemMonitorWebSocket` subscribes the system/* channels, and it is
 *     mounted only by the Monitor page
 *   - on socket connect, the store STOPS polling
 *   - leaving the Monitor page unsubscribes those channels, but the socket
 *     stays connected app-wide, so isWebSocketConnected remains true
 *   - nothing restarts polling, and the header's own start is a []-deps effect
 *
 * So after one visit to the Monitor page, the header froze everywhere else.
 * A watchdog with this exact logic already existed — inside the Monitor page,
 * which is the one page that was still fine.
 *
 * MUTATION CONTROLS:
 *   * gate the restart on isWebSocketConnected -> the connected-but-silent test fails
 *   * drop the staleness check                 -> the fresh-metrics test fails
 *   * drop the visibility guard                -> the hidden-tab test fails
 */

import * as nodeFs from 'node:fs';
import * as nodePath from 'node:path';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook } from '@testing-library/react';
import {
  useSystemMetricsWatchdog,
  METRICS_STALE_MS,
  WATCHDOG_INTERVAL_MS,
} from './useSystemMetricsWatchdog';
import { useSystemMonitorStore } from '../stores/systemMonitorStore';

const startPolling = vi.fn();

function setState(over: Record<string, unknown>) {
  useSystemMonitorStore.setState({
    isPolling: false,
    isWebSocketConnected: true,
    lastSuccessfulFetch: Date.now() - (METRICS_STALE_MS + 5_000),
    updateInterval: 2000,
    startPolling,
    ...over,
  } as never);
}

describe('useSystemMetricsWatchdog', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    startPolling.mockClear();
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'visible',
    });
  });
  afterEach(() => vi.useRealTimers());

  it('resumes polling when the socket is connected but metrics are stale', () => {
    // The exact state after navigating away from the Monitor page.
    setState({ isWebSocketConnected: true });
    renderHook(() => useSystemMetricsWatchdog());

    vi.advanceTimersByTime(WATCHDOG_INTERVAL_MS + 100);

    expect(startPolling).toHaveBeenCalled();
  });

  it('does nothing while metrics keep arriving', () => {
    // Metrics must be refreshed as time advances — with fake timers Date.now()
    // moves too, so a single "fresh" stamp goes stale mid-test and would make
    // this pass for the wrong reason.
    setState({ lastSuccessfulFetch: Date.now() });
    renderHook(() => useSystemMetricsWatchdog());

    for (let i = 0; i < 6; i++) {
      vi.advanceTimersByTime(WATCHDOG_INTERVAL_MS / 2);
      useSystemMonitorStore.setState({ lastSuccessfulFetch: Date.now() } as never);
    }

    expect(startPolling).not.toHaveBeenCalled();
  });

  it('does nothing when polling is already running', () => {
    setState({ isPolling: true });
    renderHook(() => useSystemMetricsWatchdog());

    vi.advanceTimersByTime(WATCHDOG_INTERVAL_MS * 3);

    expect(startPolling).not.toHaveBeenCalled();
  });

  it('stays quiet in a hidden tab, which is throttled and always looks stale', () => {
    Object.defineProperty(document, 'visibilityState', {
      configurable: true,
      get: () => 'hidden',
    });
    setState({});
    renderHook(() => useSystemMetricsWatchdog());

    vi.advanceTimersByTime(WATCHDOG_INTERVAL_MS * 3);

    expect(startPolling).not.toHaveBeenCalled();
  });

  it('treats never-fetched as stale', () => {
    setState({ lastSuccessfulFetch: null });
    renderHook(() => useSystemMetricsWatchdog());

    vi.advanceTimersByTime(WATCHDOG_INTERVAL_MS + 100);

    expect(startPolling).toHaveBeenCalled();
  });

  it('stops watching on unmount', () => {
    setState({});
    const { unmount } = renderHook(() => useSystemMetricsWatchdog());
    unmount();

    vi.advanceTimersByTime(WATCHDOG_INTERVAL_MS * 3);

    expect(startPolling).not.toHaveBeenCalled();
  });
});

describe('the watchdog is actually wired to the global header', () => {
  /**
   * Reachability, per the house rule: a capability is not shipped until a test
   * FAILS when its wiring is removed. Relying on an unused-import type error is
   * not enough — deleting the call and the import together would be silent, and
   * the header would quietly freeze again on every page but the Monitor.
   */
  it('CompactGPUStatus calls it', async () => {
    const fs = await import('node:fs');
    const path = await import('node:path');
    const src = fs.readFileSync(
      path.resolve(__dirname, '../components/SystemMonitor/CompactGPUStatus.tsx'),
      'utf8',
    );
    expect(src).toContain('useSystemMetricsWatchdog(');
  });

  it('the Monitor page uses the SAME hook rather than its own copy', () => {
    // ESM imports, not require(): this file is ESM and `require` is not
    // defined at runtime under vitest's node environment either.
    const fs = nodeFs;
    const path = nodePath;
    const src = fs.readFileSync(
      path.resolve(__dirname, '../components/SystemMonitor/SystemMonitor.tsx'),
      'utf8',
    );
    expect(src).toContain('useSystemMetricsWatchdog(');
    // The duplicated inline watchdog must be gone, or the two drift.
    expect(src).not.toContain('resuming polling fallback');
  });
});

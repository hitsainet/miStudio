/**
 * The export panel must learn that a build finished.
 *
 * ⚠ It said "Build queued (<task id>)" forever. The message was set on the 202 and nothing
 * could clear it: `build_probe_definition` emitted no socket event, and this hook listened only
 * for `probe_monitor:{progress,completed,failed}`, which only the RUN task sends. A build that
 * completed in seconds still read as queued until the page was reloaded.
 *
 * Reported 2026-09-28 against a build that had finished twenty minutes earlier.
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { renderHook } from '@testing-library/react';

const handlers: Record<string, (d: unknown) => void> = {};
const loadProbes = vi.fn();

vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({
    on: (event: string, fn: (d: unknown) => void) => {
      handlers[event] = fn;
    },
    off: (event: string) => {
      delete handlers[event];
    },
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
    isConnected: true,
  }),
}));

vi.mock('../../stores/probeMonitorsStore', () => ({
  useProbeMonitorsStore: (selector: (s: unknown) => unknown) =>
    selector({
      applyRunProgress: vi.fn(),
      loadRuns: vi.fn(),
      loadProbes,
    }),
}));

const { useProbeMonitorWebSocket } = await import('../useProbeMonitorWebSocket');

describe('probe definition events', () => {
  beforeEach(() => {
    loadProbes.mockClear();
    for (const k of Object.keys(handlers)) delete handlers[k];
  });

  it('⚠ subscribes to the three definition events', () => {
    renderHook(() => useProbeMonitorWebSocket('pmr_1'));
    expect(handlers['probe_monitor:definition_built']).toBeTypeOf('function');
    expect(handlers['probe_monitor:definition_failed']).toBeTypeOf('function');
    expect(handlers['probe_monitor:definition_published']).toBeTypeOf('function');
  });

  it.each([
    'probe_monitor:definition_built',
    'probe_monitor:definition_failed',
    'probe_monitor:definition_published',
  ])('%s refetches the probes for its run', (event) => {
    renderHook(() => useProbeMonitorWebSocket('pmr_1'));
    handlers[event]({ run_id: 'pmr_1', probe_id: 'pm_1' });
    expect(loadProbes).toHaveBeenCalledWith('pmr_1');
  });

  it('still handles the run events it always did', () => {
    renderHook(() => useProbeMonitorWebSocket('pmr_1'));
    expect(handlers['probe_monitor:progress']).toBeTypeOf('function');
    expect(handlers['probe_monitor:completed']).toBeTypeOf('function');
    expect(handlers['probe_monitor:failed']).toBeTypeOf('function');
  });

  it('unsubscribes every definition handler on unmount', () => {
    const { unmount } = renderHook(() => useProbeMonitorWebSocket('pmr_1'));
    unmount();
    expect(handlers['probe_monitor:definition_built']).toBeUndefined();
    expect(handlers['probe_monitor:definition_failed']).toBeUndefined();
    expect(handlers['probe_monitor:definition_published']).toBeUndefined();
  });
});

/**
 * MIS-E2E-125: the shared polling helper gave up too easily and reported too late.
 *
 * 1. A single transient fetch error terminated polling permanently, and the
 *    only caller discards the returned stop handle — so a ten-minute model
 *    download that hit one 502 stopped updating and could never resume. The
 *    download finished; the UI showed it in progress indefinitely.
 * 2. There was no in-flight guard, so a slow response could resolve after
 *    polling had stopped and push stale non-terminal state through onUpdate.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { startPolling } from './polling';

beforeEach(() => vi.useFakeTimers());
afterEach(() => vi.useRealTimers());

function harness(fetchStatus: () => Promise<unknown>, over: Record<string, unknown> = {}) {
  const onUpdate = vi.fn();
  const onComplete = vi.fn();
  const onError = vi.fn();
  const stop = startPolling({
    fetchStatus: fetchStatus as never,
    onUpdate,
    onComplete,
    onError,
    isTerminal: (r: never) => (r as { done?: boolean })?.done === true,
    interval: 10,
    maxPolls: 100,
    resourceId: 'r1',
    resourceType: 'test',
    ...over,
  } as never);
  return { stop, onUpdate, onComplete, onError };
}

describe('startPolling — transient errors', () => {
  it('survives a single failed fetch and keeps polling', async () => {
    let call = 0;
    const fetchStatus = vi.fn(async () => {
      call++;
      if (call === 1) throw new Error('502 Bad Gateway');
      return { done: false };
    });

    const { onUpdate, onError } = harness(fetchStatus);

    await vi.advanceTimersByTimeAsync(10);   // the failure
    await vi.advanceTimersByTimeAsync(10);   // must still be polling
    await vi.advanceTimersByTimeAsync(10);

    expect(onError).not.toHaveBeenCalled();
    expect(onUpdate).toHaveBeenCalled();
    expect(fetchStatus.mock.calls.length).toBeGreaterThan(1);
  });

  it('still gives up on a sustained run of failures', async () => {
    const fetchStatus = vi.fn(async () => {
      throw new Error('connection refused');
    });
    const { onError } = harness(fetchStatus);

    for (let i = 0; i < 8; i++) await vi.advanceTimersByTimeAsync(10);

    expect(onError).toHaveBeenCalled();
    const calls = fetchStatus.mock.calls.length;
    await vi.advanceTimersByTimeAsync(50);
    expect(fetchStatus.mock.calls.length).toBe(calls); // stopped for good
  });

  it('a success resets the failure run', async () => {
    // The pattern has to DISCRIMINATE: four failures, a success, four more.
    // With the reset the longest run is 4 and polling continues. Without it
    // the count reaches 8 and polling stops — so this fails if the reset is
    // removed. An earlier version used 2/1/2, which never reached the
    // threshold either way and so proved nothing (control C203 survived it).
    let call = 0;
    const fetchStatus = vi.fn(async () => {
      call++;
      if (call === 5) return { done: false };
      throw new Error('flaky');
    });
    const { onError } = harness(fetchStatus);

    for (let i = 0; i < 9; i++) await vi.advanceTimersByTimeAsync(10);
    expect(onError).not.toHaveBeenCalled();
  });
});

describe('startPolling — reporting after stop', () => {
  it('does not call onUpdate for a response that lands after stop()', async () => {
    let release: (v: unknown) => void = () => {};
    const fetchStatus = vi.fn(
      () => new Promise((res) => { release = res; })
    );

    const { stop, onUpdate } = harness(fetchStatus);
    await vi.advanceTimersByTimeAsync(10);   // poll goes out, hangs

    stop();                                   // caller stops
    release({ done: false });                 // slow response lands
    await vi.advanceTimersByTimeAsync(1);

    expect(onUpdate).not.toHaveBeenCalled();
  });

  it('does not overlap two polls when a response is slower than the interval', async () => {
    const resolvers: Array<(v: unknown) => void> = [];
    const fetchStatus = vi.fn(
      () => new Promise((res) => { resolvers.push(res); })
    );

    harness(fetchStatus);
    await vi.advanceTimersByTimeAsync(10);
    await vi.advanceTimersByTimeAsync(10);
    await vi.advanceTimersByTimeAsync(10);

    // Three ticks, one still in flight: only one request should be out.
    expect(fetchStatus.mock.calls.length).toBe(1);
    resolvers.forEach((r) => r({ done: true }));
  });
});

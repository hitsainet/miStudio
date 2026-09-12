/**
 * MIS-E2E-120 / -126 — reconnects duplicated every handler.
 *
 * The `connect` handler re-attached every entry in `eventHandlersRef`, under a
 * comment reading "Re-attach existing handlers FIRST (for reconnections)".
 * socket.io does NOT detach listeners on disconnect: the same Socket instance
 * keeps them across the whole reconnect cycle. So each reconnect added a SECOND
 * registration of every handler already attached, and after N reconnects one
 * server message ran every handler N+1 times.
 *
 * Reconnects are routine — a pod restart, a laptop waking. For a progress event
 * the duplication is noise; for `extraction:completed`, or any store action
 * that appends, it is N+1 duplicate effects from a single event.
 *
 * And `unsubscribe` never cleared `pendingSubscriptionsRef`, so a channel the
 * user abandoned while disconnected was subscribed on the next connect and
 * every reconnect after that — compounding the above.
 *
 * This file did not exist. Neither behaviour was pinned by anything.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import { useEffect } from 'react';

/** A socket.io double that behaves the way the real one does on reconnect. */
class FakeSocket {
  id = 'fake-1';
  connected = false;
  /** event -> handlers, exactly as socket.io keeps them: NOT cleared on disconnect. */
  listeners = new Map<string, Array<(...a: unknown[]) => void>>();
  emitted: Array<{ event: string; payload: unknown }> = [];

  on(event: string, handler: (...a: unknown[]) => void) {
    if (!this.listeners.has(event)) this.listeners.set(event, []);
    this.listeners.get(event)!.push(handler);
  }

  off(event: string, handler?: (...a: unknown[]) => void) {
    if (!handler) this.listeners.delete(event);
    else {
      const list = this.listeners.get(event) ?? [];
      this.listeners.set(event, list.filter((h) => h !== handler));
    }
  }

  emit(event: string, payload?: unknown) {
    this.emitted.push({ event, payload });
  }

  disconnect() {
    this.connected = false;
  }

  /** Drive a (re)connect the way socket.io does: fire 'connect' again. */
  fireConnect() {
    this.connected = true;
    (this.listeners.get('connect') ?? []).slice().forEach((h) => h());
  }

  /** Deliver a server message. */
  deliver(event: string, payload: unknown) {
    (this.listeners.get(event) ?? []).slice().forEach((h) => h(payload));
  }

  countFor(event: string) {
    return (this.listeners.get(event) ?? []).length;
  }
}

let fake: FakeSocket;

vi.mock('socket.io-client', () => ({
  io: () => fake,
}));

// Imported after the mock so the provider picks up the double.
const { WebSocketProvider, useWebSocketContext } = await import('./WebSocketContext');

function Consumer({ onEvent }: { onEvent: (p: unknown) => void }) {
  const { on, subscribe } = useWebSocketContext();
  useEffect(() => {
    subscribe('extraction/e1');
    on('extraction:completed', onEvent);
    // Intentionally no cleanup: the provider owns handler lifetime, and this
    // mirrors how the panels actually use it.
  }, [on, subscribe, onEvent]);
  return null;
}

beforeEach(() => {
  fake = new FakeSocket();
  vi.restoreAllMocks();
  vi.spyOn(console, 'log').mockImplementation(() => {});
});

describe('WebSocketContext across reconnects', () => {
  it('registers a handler exactly once, however many reconnects happen', () => {
    const seen = vi.fn();
    render(
      <WebSocketProvider>
        <Consumer onEvent={seen} />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    expect(fake.countFor('extraction:completed')).toBe(1);

    // Three reconnects, the way a flaky network or a pod restart produces them.
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());

    expect(fake.countFor('extraction:completed')).toBe(1);
  });

  it('runs a handler once per server message after reconnects', () => {
    // The consequence, stated as behaviour rather than as a listener count.
    const seen = vi.fn();
    render(
      <WebSocketProvider>
        <Consumer onEvent={seen} />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());

    act(() => fake.deliver('extraction:completed', { id: 'e1' }));

    expect(seen).toHaveBeenCalledTimes(1);
  });

  it('resubscribes each active channel once per connect, not cumulatively', () => {
    render(
      <WebSocketProvider>
        <Consumer onEvent={() => {}} />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    const afterFirst = fake.emitted.filter((e) => e.event === 'subscribe').length;

    act(() => fake.fireConnect());
    const afterSecond = fake.emitted.filter((e) => e.event === 'subscribe').length;

    // One more subscribe per connect for the one active channel — not two, and
    // not growing.
    expect(afterSecond - afterFirst).toBe(1);
  });
});

describe('unsubscribe clears the pending queue (MIS-E2E-126)', () => {
  function Abandoner() {
    const { subscribe, unsubscribe } = useWebSocketContext();
    useEffect(() => {
      // Subscribed while DISCONNECTED, so it lands in the pending queue...
      subscribe('steering/task-abandoned');
      // ...and abandoned before the socket ever connects.
      unsubscribe('steering/task-abandoned');
    }, [subscribe, unsubscribe]);
    return null;
  }

  it('does not subscribe a channel the user abandoned while disconnected', () => {
    render(
      <WebSocketProvider>
        <Abandoner />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());

    const subscribed = fake.emitted
      .filter((e) => e.event === 'subscribe')
      .map((e) => (e.payload as { channel: string }).channel);
    expect(subscribed).not.toContain('steering/task-abandoned');
  });

  it('and does not resubscribe it on every reconnect thereafter', () => {
    render(
      <WebSocketProvider>
        <Abandoner />
      </WebSocketProvider>,
    );

    act(() => fake.fireConnect());
    act(() => fake.fireConnect());
    act(() => fake.fireConnect());

    const count = fake.emitted
      .filter(
        (e) =>
          e.event === 'subscribe' &&
          (e.payload as { channel: string }).channel === 'steering/task-abandoned',
      )
      .length;
    expect(count).toBe(0);
  });
});


/**
 * A subscription belongs to the SOCKET, not to the component that asked for it.
 *
 * Reported 2026-08-26: extraction progress on the Models list froze at
 * "Extraction job queued, waiting for worker..." and only a browser refresh
 * recovered it. The card and the extraction modal both subscribe to
 * `models/{id}/extraction`; closing the modal ran its cleanup, which emitted
 * `unsubscribe` and evicted the socket from the room while the card was still
 * listening. The card's handler stayed registered and simply received nothing
 * — which is why the one event that arrived while BOTH were mounted was the
 * only one ever displayed.
 *
 * Proven against the live system first: a socket.io client subscribed to that
 * channel received 13 `extraction:progress` events in 25 seconds, so the
 * server was never the problem.
 */
describe('channel subscriptions are reference counted', () => {
  function Subscriber({ channel }: { channel: string }) {
    const { subscribe, unsubscribe } = useWebSocketContext();
    useEffect(() => {
      subscribe(channel);
      return () => unsubscribe(channel);
    }, [channel, subscribe, unsubscribe]);
    return null;
  }

  const CH = 'models/m_b55c6926/extraction';

  const subscribes = () => fake.emitted.filter(
    (e) => e.event === 'subscribe' && (e.payload as { channel: string }).channel === CH
  ).length;
  const unsubscribes = () => fake.emitted.filter(
    (e) => e.event === 'unsubscribe' && (e.payload as { channel: string }).channel === CH
  ).length;

  it('joins the room once for two subscribers', () => {
    // Both must subscribe while CONNECTED. Subscribing before connect parks
    // them in the pending Set, which dedupes on its own and would hide a
    // missing guard.
    fake = new FakeSocket();
    const { rerender } = render(<WebSocketProvider>{null}</WebSocketProvider>);
    act(() => fake.fireConnect());

    rerender(
      <WebSocketProvider>
        <Subscriber channel={CH} />
        <Subscriber channel={CH} />
      </WebSocketProvider>
    );

    expect(subscribes()).toBe(1);
  });

  it('does NOT leave the room while another subscriber remains', () => {
    fake = new FakeSocket();
    const { rerender } = render(
      <WebSocketProvider>
        <Subscriber channel={CH} />
        <Subscriber channel={CH} />
      </WebSocketProvider>
    );
    act(() => fake.fireConnect());

    // one of the two goes away — the modal being closed
    rerender(
      <WebSocketProvider>
        <Subscriber channel={CH} />
      </WebSocketProvider>
    );

    expect(unsubscribes()).toBe(0);
  });

  it('leaves the room once the last subscriber goes', () => {
    fake = new FakeSocket();
    const { rerender } = render(
      <WebSocketProvider>
        <Subscriber channel={CH} />
        <Subscriber channel={CH} />
      </WebSocketProvider>
    );
    act(() => fake.fireConnect());

    rerender(<WebSocketProvider><Subscriber channel={CH} /></WebSocketProvider>);
    rerender(<WebSocketProvider>{null}</WebSocketProvider>);

    expect(unsubscribes()).toBe(1);
  });

  it('still delivers events to the survivor after the other unmounts', () => {
    fake = new FakeSocket();
    const seen: unknown[] = [];

    function Listener() {
      const { subscribe, unsubscribe, on, off } = useWebSocketContext();
      useEffect(() => {
        const h = (p: unknown) => seen.push(p);
        subscribe(CH);
        on('extraction:progress', h);
        return () => {
          unsubscribe(CH);
          off('extraction:progress', h);
        };
      }, [subscribe, unsubscribe, on, off]);
      return null;
    }

    const { rerender } = render(
      <WebSocketProvider>
        <Listener />
        <Subscriber channel={CH} />
      </WebSocketProvider>
    );
    act(() => fake.fireConnect());
    rerender(
      <WebSocketProvider>
        <Listener />
      </WebSocketProvider>
    );

    act(() => fake.deliver('extraction:progress', { progress: 42 }));

    expect(unsubscribes()).toBe(0);
    expect(seen).toContainEqual({ progress: 42 });
  });
});

/**
 * WebSocket faults must be visible in a PRODUCTION build.
 *
 * vite marks console.log/debug/info/trace pure, so the production bundle drops
 * them. Every WebSocket lifecycle diagnostic used console.log, which meant that
 * when a user reported extraction progress freezing mid-run (2026-08-27) their
 * console held nothing — "the socket dropped" and "the subscription was
 * refused" were indistinguishable from the browser.
 *
 * Worse, `subscribe_error` had no listener at all. The server emits it when
 * validate_channel rejects a channel; the component went on believing it was
 * subscribed and received nothing forever.
 *
 * These assert the CHANNEL of the log, not its text, because the channel is
 * what survives minification.
 */
describe('WebSocket faults are diagnosable in production', () => {
  let warn: ReturnType<typeof vi.spyOn>;
  let error: ReturnType<typeof vi.spyOn>;
  let log: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    fake = new FakeSocket();
    warn = vi.spyOn(console, 'warn').mockImplementation(() => undefined);
    error = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    log = vi.spyOn(console, 'log').mockImplementation(() => undefined);
  });

  afterEach(() => {
    warn.mockRestore();
    error.mockRestore();
    log.mockRestore();
  });

  const joined = (spy: ReturnType<typeof vi.spyOn>) =>
    spy.mock.calls.map((c) => c.map(String).join(' ')).join('\n');

  it('reports a refused subscription instead of failing silently', () => {
    render(<WebSocketProvider>{null}</WebSocketProvider>);
    act(() => fake.fireConnect());

    act(() =>
      fake.deliver('subscribe_error', { error: "unknown channel topic 'model'" })
    );

    expect(joined(error)).toMatch(/REFUSED/i);
    expect(joined(error)).toMatch(/unknown channel topic/);
  });

  it('warns on disconnect, on a channel the production build keeps', () => {
    render(<WebSocketProvider>{null}</WebSocketProvider>);
    act(() => fake.fireConnect());

    act(() => fake.deliver('disconnect', 'transport close'));

    expect(joined(warn)).toMatch(/Disconnected/);
    expect(joined(warn)).toMatch(/transport close/);
    // console.log is stripped in production, so it must NOT be the only record.
    expect(joined(log)).not.toMatch(/Disconnected/);
  });

  it('warns when a reconnect resubscribes, so recovery is observable', () => {
    function Sub() {
      const { subscribe } = useWebSocketContext();
      useEffect(() => {
        subscribe('models/m_1/extraction');
      }, [subscribe]);
      return null;
    }

    render(
      <WebSocketProvider>
        <Sub />
      </WebSocketProvider>
    );
    act(() => fake.fireConnect());

    warn.mockClear();
    act(() => fake.deliver('disconnect', 'ping timeout'));
    act(() => fake.fireConnect());

    expect(joined(warn)).toMatch(/Resubscribing/i);
    expect(joined(warn)).toMatch(/models\/m_1\/extraction/);
  });
});

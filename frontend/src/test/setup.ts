import '@testing-library/jest-dom';
import { cleanup } from '@testing-library/react';
import { afterEach, vi } from 'vitest';

// Globally stub socket.io-client so no test opens a real network connection.
// WebSocketProvider calls io() inside a useEffect; without this, any component
// tree wrapped in <WebSocketProvider> would attempt a live socket.io handshake.
// The stub returns a harmless fake Socket with no-op event/emit methods.
vi.mock('socket.io-client', () => {
  const createFakeSocket = () => ({
    id: 'test-socket',
    connected: false,
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    removeListener: vi.fn(),
    removeAllListeners: vi.fn(),
  });
  const io = vi.fn(() => createFakeSocket());
  return { io, default: io, Socket: class {} };
});

// jsdom has no ResizeObserver, which recharts' ResponsiveContainer constructs
// unconditionally. Without this, any test that renders a chart dies with
// "ResizeObserver is not defined" — an environment gap, not a component fault.
if (typeof globalThis.ResizeObserver === 'undefined') {
  globalThis.ResizeObserver = class {
    observe() {}
    unobserve() {}
    disconnect() {}
  } as unknown as typeof ResizeObserver;
}

// Cleanup after each test
afterEach(() => {
  cleanup();
});

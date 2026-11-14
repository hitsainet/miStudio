/**
 * Shared test render helper.
 *
 * Wraps the component under test in the app-level providers that many
 * components (transitively) require. Currently this is the WebSocketProvider,
 * which supplies the context consumed by useWebSocketContext() — without it,
 * hooks like useModelExtractionProgress throw
 * "useWebSocketContext must be used within WebSocketProvider".
 *
 * socket.io-client is globally stubbed in src/test/setup.ts, so the provider
 * does NOT open a real connection during tests.
 *
 * Zustand stores are module-level singletons (not React context), so they do
 * not need a provider here.
 */
import type { ReactElement, ReactNode } from 'react';
import { render, renderHook } from '@testing-library/react';
import type { RenderOptions, RenderHookOptions } from '@testing-library/react';
import { WebSocketProvider } from '../contexts/WebSocketContext';

function AllProviders({ children }: { children: ReactNode }) {
  return <WebSocketProvider>{children}</WebSocketProvider>;
}

export function renderWithProviders(
  ui: ReactElement,
  options?: Omit<RenderOptions, 'wrapper'>,
) {
  return render(ui, { wrapper: AllProviders, ...options });
}

export function renderHookWithProviders<Result, Props>(
  callback: (props: Props) => Result,
  options?: Omit<RenderHookOptions<Props>, 'wrapper'>,
) {
  return renderHook(callback, { wrapper: AllProviders, ...options });
}

// Re-export everything from RTL so tests can import screen, waitFor, etc.
// from a single module.
export * from '@testing-library/react';

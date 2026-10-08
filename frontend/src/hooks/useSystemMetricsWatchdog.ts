/**
 * Keep system metrics flowing on EVERY page, not just the Monitor page.
 *
 * THE FAILURE THIS FIXES (reported 2026-07-27: "the small monitoring elements
 * at the top of every page do not update on their own")
 *
 *   1. `useSystemMonitorWebSocket` — the only subscriber to the `system/*`
 *      channels — is mounted by SystemMonitor.tsx, i.e. the Monitor page ONLY.
 *   2. When the socket connects, systemMonitorStore.setIsWebSocketConnected
 *      STOPS polling, on the assumption that WebSocket updates have taken over.
 *   3. Navigating away from the Monitor page unsubscribes those channels — but
 *      the socket itself stays connected app-wide, so `isWebSocketConnected`
 *      remains true and polling is never restarted.
 *   4. CompactGPUStatus (the header, on every page) starts polling once from a
 *      `[]`-dependency effect, so it never re-runs to recover.
 *
 * Net effect: after visiting the Monitor page once, the header freezes on every
 * other page until a full reload.
 *
 * The bug is that CONNECTION STATE is used as a proxy for "metrics are
 * arriving". They are different things: the socket is connected application
 * wide, while the system channels are subscribed on one page.
 *
 * A watchdog with exactly this logic already existed — but only inside
 * SystemMonitor.tsx, so it healed the one page that was not broken. This hook
 * is that logic, extracted so both callers share one implementation and cannot
 * drift.
 */

import { useEffect } from 'react';
import { useSystemMonitorStore } from '../stores/systemMonitorStore';

/** No metric update for this long while "connected" means we are not receiving. */
export const METRICS_STALE_MS = 10_000;
/** How often to test for staleness. */
export const WATCHDOG_INTERVAL_MS = 5_000;

export function useSystemMetricsWatchdog(updateInterval?: number): void {
  const startPolling = useSystemMonitorStore((s) => s.startPolling);
  const storeInterval = useSystemMonitorStore((s) => s.updateInterval);
  const interval = updateInterval ?? storeInterval;

  useEffect(() => {
    const watchdog = window.setInterval(() => {
      // A hidden tab gets throttled and would look stale forever.
      if (document.visibilityState !== 'visible') return;

      // isWebSocketConnected is deliberately NOT consulted — see below.
      const { isPolling, lastSuccessfulFetch } =
        useSystemMonitorStore.getState();

      const stale =
        lastSuccessfulFetch === null ||
        Date.now() - lastSuccessfulFetch > METRICS_STALE_MS;

      // Deliberately keyed on STALENESS, not on connectivity. A connected
      // socket with no subscription to system/* delivers nothing, and that is
      // the common case away from the Monitor page.
      if (!isPolling && stale) {
        console.warn(
          '[SystemMetricsWatchdog] metrics are stale, resuming polling fallback',
        );
        startPolling(interval);
      }
    }, WATCHDOG_INTERVAL_MS);

    return () => window.clearInterval(watchdog);
  }, [startPolling, interval]);
}

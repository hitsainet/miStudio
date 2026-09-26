/**
 * Probe monitor WebSocket hook (Feature 032 FR-14).
 *
 * Channel: `probe_monitors/{run_id}`
 * Events:  probe_monitor:progress | probe_monitor:completed | probe_monitor:failed
 *
 * ⚠ POLLING IS THE FALLBACK, AND IT STOPS WHEN THE SOCKET COMES BACK. A probe run is
 * minutes to hours, so a panel that only listens goes silent whenever the socket drops —
 * and one that only polls hammers the API for the whole run. The J-Lens shape: subscribe,
 * poll while disconnected, stop polling on reconnect.
 *
 * ⚠ THE SUBSCRIPTION IS PER RUN AND UNSUBSCRIBED ON UNMOUNT. Subscriptions here are
 * per-socket, so leaving one open after the component goes away means a later component's
 * unsubscribe can silence a channel this one still needs — the recorded refcount hazard.
 */

import { useEffect, useRef } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useProbeMonitorsStore } from '../stores/probeMonitorsStore';

interface ProbeProgressEvent {
  run_id: string;
  progress: number;
  stage: string | null;
  status: string;
  message: string | null;
}

/** How often to poll while the socket is down. Matches the J-Lens panel's 2.5s. */
const POLL_INTERVAL_MS = 2500;

export const useProbeMonitorWebSocket = (runId: string | null) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const applyRunProgress = useProbeMonitorsStore((state) => state.applyRunProgress);
  const loadRuns = useProbeMonitorsStore((state) => state.loadRuns);
  const registered = useRef(false);

  useEffect(() => {
    if (registered.current) return;

    const handleProgress = (data: ProbeProgressEvent) => {
      applyRunProgress(data.run_id, {
        progress: data.progress,
        stage: data.stage,
        status: data.status,
      });
    };
    const handleCompleted = (data: { run_id: string }) => {
      applyRunProgress(data.run_id, { status: 'completed', progress: 100 });
    };
    const handleFailed = (data: { run_id: string; error: string }) => {
      applyRunProgress(data.run_id, { status: 'failed', error_message: data.error });
    };

    on('probe_monitor:progress', handleProgress);
    on('probe_monitor:completed', handleCompleted);
    on('probe_monitor:failed', handleFailed);
    registered.current = true;

    return () => {
      off('probe_monitor:progress', handleProgress);
      off('probe_monitor:completed', handleCompleted);
      off('probe_monitor:failed', handleFailed);
      registered.current = false;
    };
  }, [on, off, applyRunProgress]);

  useEffect(() => {
    if (!runId || !isConnected) return;
    const channel = `probe_monitors/${runId}`;
    subscribe(channel);
    return () => unsubscribe(channel);
  }, [runId, isConnected, subscribe, unsubscribe]);

  // THE FALLBACK. Only while disconnected, and cleared the moment the socket returns.
  useEffect(() => {
    if (isConnected) return;
    const timer = setInterval(() => {
      void loadRuns();
    }, POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [isConnected, loadRuns]);
};

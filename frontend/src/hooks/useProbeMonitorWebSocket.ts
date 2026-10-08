/**
 * Probe monitor WebSocket hook (Feature 032 FR-14).
 *
 * Channel: `probe_monitors/{run_id}`
 * Events:  probe_monitor:progress | probe_monitor:completed | probe_monitor:failed
 *           probe_monitor:definition_built | :definition_failed | :definition_published
 *
 * ⚠ THE THREE `definition_*` EVENTS EXIST BECAUSE THE BUILD JOB EMITTED NOTHING. The export
 * panel sets "Build queued (<task id>)" on the 202 and had no other way to learn the job ended,
 * so a build that finished in seconds still read as queued until the page was reloaded. Reported
 * against a build that had completed twenty minutes earlier. They ride the RUN's channel, which
 * this hook already subscribes to — the gap was a missing message, not missing plumbing.
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
  const loadProbes = useProbeMonitorsStore((state) => state.loadProbes);
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

    // A definition build or publish changes the PROBE row, not the run's progress, so these
    // refetch the probes rather than touching run state.
    const handleDefinitionSettled = (data: { run_id: string }) => {
      void loadProbes(data.run_id);
    };

    on('probe_monitor:progress', handleProgress);
    on('probe_monitor:completed', handleCompleted);
    on('probe_monitor:failed', handleFailed);
    on('probe_monitor:definition_built', handleDefinitionSettled);
    on('probe_monitor:definition_failed', handleDefinitionSettled);
    on('probe_monitor:definition_published', handleDefinitionSettled);
    registered.current = true;

    return () => {
      off('probe_monitor:progress', handleProgress);
      off('probe_monitor:completed', handleCompleted);
      off('probe_monitor:failed', handleFailed);
      off('probe_monitor:definition_built', handleDefinitionSettled);
      off('probe_monitor:definition_failed', handleDefinitionSettled);
      off('probe_monitor:definition_published', handleDefinitionSettled);
      registered.current = false;
    };
  }, [on, off, applyRunProgress, loadProbes]);

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

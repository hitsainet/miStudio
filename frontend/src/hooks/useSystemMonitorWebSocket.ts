/**
 * System Monitor WebSocket Hook
 *
 * React hook for subscribing to system resource metrics via WebSocket.
 * Replaces the old HTTP polling approach with real-time WebSocket push.
 *
 * WebSocket Channels:
 * - system/gpu/{gpu_id} - Per-GPU metrics (utilization, memory, temperature)
 * - system/cpu - CPU utilization metrics
 * - system/memory - RAM and Swap metrics
 * - system/disk - Disk I/O metrics
 * - system/network - Network I/O metrics
 *
 * Events:
 * - metrics - Metrics update from Celery beat task (every 2 seconds)
 *
 * Usage:
 *   useSystemMonitorWebSocket(gpuIds);
 *
 *   // Automatically subscribes to all system channels + specific GPU channels
 *   // Updates are handled by systemMonitorStore update methods
 */

import { useEffect, useRef, useMemo } from 'react';
import { useWebSocketContext } from '../contexts/WebSocketContext';
import { useSystemMonitorStore } from '../stores/systemMonitorStore';

export const useSystemMonitorWebSocket = (gpuIds: number[] = []) => {
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();
  const {
    setGPUMetrics,
    updateSystemMetrics,
    setIsWebSocketConnected,
  } = useSystemMonitorStore();
  const handlersRegisteredRef = useRef(false);

  // Set up global event handlers (once)
  useEffect(() => {
    if (handlersRegisteredRef.current) return;

    console.log('[System Monitor WS] Setting up system metrics event handlers');

    // Handler for 'metrics' events on all system channels.
    // The backend tags each payload with metric_type; the field-sniffing
    // branches below are a fallback for payloads from older backends.
    const handleMetrics = (data: any) => {
      const metricType: string | undefined = data.metric_type;

      if (metricType === 'gpu' || data.gpu_id !== undefined) {
        setGPUMetrics(data.gpu_id, data);
      } else if (
        metricType === 'cpu' ||
        (data.percent !== undefined && data.count !== undefined)
      ) {
        updateSystemMetrics({
          cpu: { percent: data.percent, count: data.count },
        });
      } else if (
        metricType === 'memory' ||
        (data.ram !== undefined && data.swap !== undefined)
      ) {
        updateSystemMetrics({
          ram: data.ram,
          swap: data.swap,
        });
      } else if (
        metricType === 'disk' ||
        (data.read_bytes !== undefined && data.write_bytes !== undefined)
      ) {
        updateSystemMetrics({
          disk_io: {
            read_bytes: data.read_bytes,
            write_bytes: data.write_bytes,
            read_mb: data.read_bytes / (1024 * 1024),
            write_mb: data.write_bytes / (1024 * 1024),
          },
        });
      } else if (
        metricType === 'network' ||
        (data.sent_bytes !== undefined && data.recv_bytes !== undefined)
      ) {
        updateSystemMetrics({
          network_io: {
            sent_bytes: data.sent_bytes,
            recv_bytes: data.recv_bytes,
            sent_mb: data.sent_bytes / (1024 * 1024),
            recv_mb: data.recv_bytes / (1024 * 1024),
          },
        });
      } else {
        console.warn('[System Monitor WS] Unrecognized metrics payload:', data);
      }
    };

    // Register event handlers with namespace prefix for proper WebSocket routing
    on('system:metrics', handleMetrics);

    handlersRegisteredRef.current = true;
    console.log('[System Monitor WS] Event handlers registered');

    // Cleanup
    return () => {
      console.log('[System Monitor WS] Cleaning up event handlers');
      off('system:metrics', handleMetrics);
      handlersRegisteredRef.current = false;
    };
  }, [on, off, setGPUMetrics, updateSystemMetrics]);

  // Create a stable content-based key from gpuIds.
  // - Spread into a new array before sorting to avoid mutating the prop.
  // - Use numeric sort (a - b) to ensure [0, 1, 2] not ['0', '1', '2'] order.
  // - The dep [gpuIds.join(',')] uses value-based string comparison (Object.is),
  //   so the memo only re-runs when GPU count/IDs actually change.
  const gpuIdsKey = useMemo(
    () => [...gpuIds].sort((a, b) => a - b).join(','),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [gpuIds.join(',')]
  );

  // Subscribe to system monitoring channels
  useEffect(() => {
    if (!isConnected) {
      console.log('[System Monitor WS] Not connected, skipping channel subscriptions');
      setIsWebSocketConnected(false);
      return;
    }

    console.log('[System Monitor WS] Subscribing to system monitoring channels');
    setIsWebSocketConnected(true);

    // Subscribe to global system channels (always subscribed)
    const globalChannels = [
      'system/cpu',
      'system/memory',
      'system/disk',
      'system/network',
    ];

    globalChannels.forEach((channel) => {
      console.log(`[System Monitor WS] Subscribing to ${channel}`);
      subscribe(channel);
    });

    // Subscribe to GPU-specific channels
    gpuIds.forEach((gpuId) => {
      const gpuChannel = `system/gpu/${gpuId}`;
      console.log(`[System Monitor WS] Subscribing to ${gpuChannel}`);
      subscribe(gpuChannel);
    });

    // Cleanup subscriptions
    return () => {
      console.log('[System Monitor WS] Unsubscribing from system monitoring channels');

      globalChannels.forEach((channel) => {
        unsubscribe(channel);
      });

      gpuIds.forEach((gpuId) => {
        unsubscribe(`system/gpu/${gpuId}`);
      });

      setIsWebSocketConnected(false);
    };
  }, [gpuIdsKey, isConnected, subscribe, unsubscribe, setIsWebSocketConnected]);
};

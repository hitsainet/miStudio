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

    // Handler for 'metrics' events on all system channels
    const handleMetrics = (data: any) => {
      console.log('[System Monitor WS] Metrics event:', data);

      // Determine which type of metrics based on the channel or data structure
      // GPU metrics will have gpu_id field
      if (data.gpu_id !== undefined) {
        console.log('[System Monitor WS] GPU metrics update:', data.gpu_id);
        setGPUMetrics(data.gpu_id, data);
      }
      // CPU metrics
      else if (data.percent !== undefined && data.count !== undefined) {
        console.log('[System Monitor WS] CPU metrics update');
        updateSystemMetrics({
          cpu_percent: data.percent,
          cpu_count: data.count,
        });
      }
      // Memory metrics (has both ram and swap)
      else if (data.ram !== undefined && data.swap !== undefined) {
        console.log('[System Monitor WS] Memory metrics update');
        updateSystemMetrics({
          ram_used: data.ram.used,
          ram_total: data.ram.total,
          ram_available: data.ram.available,
          swap_used: data.swap.used,
          swap_total: data.swap.total,
        });
      }
      // Disk I/O metrics
      else if (data.read_bytes !== undefined && data.write_bytes !== undefined) {
        console.log('[System Monitor WS] Disk I/O metrics update');
        updateSystemMetrics({
          disk_read_bytes: data.read_bytes,
          disk_write_bytes: data.write_bytes,
        });
      }
      // Network I/O metrics
      else if (data.sent_bytes !== undefined && data.recv_bytes !== undefined) {
        console.log('[System Monitor WS] Network I/O metrics update');
        updateSystemMetrics({
          network_sent_bytes: data.sent_bytes,
          network_recv_bytes: data.recv_bytes,
        });
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

  // Create a stable key from gpuIds to prevent unnecessary re-subscriptions
  const gpuIdsKey = useMemo(() => gpuIds.sort().join(','), [gpuIds.join(',')]);

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

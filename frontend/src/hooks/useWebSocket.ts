/**
 * WebSocket hook for real-time updates.
 *
 * @deprecated This hook is DEPRECATED. Use `useWebSocketContext()` from
 * `../contexts/WebSocketContext` instead.
 *
 * PROBLEM WITH THIS HOOK:
 * This hook conflates two distinct Socket.IO concepts:
 * 1. Joining a room (to receive events emitted to that room)
 * 2. Listening for events (to handle events with a specific name)
 *
 * The old `subscribe(channel, handler)` method:
 * - Joins the Socket.IO room named `channel`
 * - ALSO listens for events named `channel` (WRONG!)
 *
 * This is incorrect because the backend emits events with their own names
 * (e.g., 'extraction:progress', 'model:completed') to specific rooms
 * (e.g., 'models/123/extraction'). Listening for 'models/123/extraction'
 * as an event name will never receive the 'extraction:progress' events.
 *
 * CORRECT PATTERN (use WebSocketContext instead):
 * ```typescript
 * const { on, off, subscribe, unsubscribe } = useWebSocketContext();
 *
 * // Join the room to receive events for this resource
 * subscribe('models/123/extraction');
 *
 * // Listen for the actual event name
 * on('extraction:progress', handleProgress);
 *
 * // Cleanup
 * unsubscribe('models/123/extraction');
 * off('extraction:progress', handleProgress);
 * ```
 *
 * DO NOT USE THIS HOOK FOR NEW CODE.
 */

import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { WS_URL, WS_PATH } from '../config/api';

interface UseWebSocketOptions {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

/**
 * @deprecated Use `useWebSocketContext()` from `../contexts/WebSocketContext` instead.
 *
 * This hook has a broken API that conflates room joining with event listening.
 * See the file header for details on the correct pattern.
 */
export function useWebSocket(options: UseWebSocketOptions = {}) {
  const socketRef = useRef<Socket | null>(null);
  const { onConnect, onDisconnect, onError } = options;

  useEffect(() => {
    // Log deprecation warning
    console.warn(
      '[useWebSocket] DEPRECATED: This hook is deprecated and has a broken API. ' +
      'Use useWebSocketContext() from ../contexts/WebSocketContext instead. ' +
      'See useWebSocket.ts for migration instructions.'
    );

    // Create Socket.IO connection
    const socket = io(WS_URL, {
      path: WS_PATH,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: 5,
    });

    socketRef.current = socket;

    // Connection event handlers
    socket.on('connect', () => {
      console.log('WebSocket connected');
      onConnect?.();
    });

    socket.on('disconnect', () => {
      console.log('WebSocket disconnected');
      onDisconnect?.();
    });

    socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      onError?.(error as Error);
    });

    // Cleanup on unmount
    return () => {
      socket.disconnect();
      socketRef.current = null;
    };
  }, [onConnect, onDisconnect, onError]);

  /**
   * Subscribe to a channel
   *
   * @deprecated This method is BROKEN. It joins a room AND listens for events
   * with the channel name, but the backend emits events with different names.
   *
   * Use WebSocketContext instead:
   * ```typescript
   * const { on, subscribe } = useWebSocketContext();
   * subscribe(channel);  // Join room only
   * on(eventName, handler);  // Listen for actual event name
   * ```
   */
  const subscribe = useCallback(
    (channel: string, handler: (data: any) => void) => {
      const socket = socketRef.current;
      if (!socket) {
        console.warn('Socket not initialized');
        return;
      }

      // Log warning for the broken pattern
      console.warn(
        `[useWebSocket] BROKEN PATTERN: subscribe('${channel}', handler) ` +
        'both joins the room AND listens for events named "' + channel + '". ' +
        'This is incorrect - use WebSocketContext.on(eventName, handler) for event listening.'
      );

      // Listen for events on this channel (BROKEN - wrong event name!)
      socket.on(channel, handler);

      // Notify server to add us to the channel's subscriber list
      socket.emit('subscribe', { channel });

      console.log(`Subscribed to channel: ${channel}`);
    },
    []
  );

  /**
   * Unsubscribe from a channel
   *
   * @deprecated Use WebSocketContext instead.
   */
  const unsubscribe = useCallback(
    (channel: string, handler?: (data: any) => void) => {
      const socket = socketRef.current;
      if (!socket) return;

      if (handler) {
        socket.off(channel, handler);
      } else {
        socket.off(channel);
      }

      // Notify server to remove us from the channel's subscriber list
      socket.emit('unsubscribe', { channel });

      console.log(`Unsubscribed from channel: ${channel}`);
    },
    []
  );

  /**
   * Emit an event to the server
   */
  const emit = useCallback((event: string, data: any) => {
    const socket = socketRef.current;
    if (!socket) {
      console.warn('Socket not initialized');
      return;
    }

    socket.emit(event, data);
  }, []);

  return {
    subscribe,
    unsubscribe,
    emit,
    isConnected: socketRef.current?.connected ?? false,
  };
}

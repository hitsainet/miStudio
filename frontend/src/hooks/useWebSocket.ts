/**
 * WebSocket hook for real-time updates.
 *
 * This hook manages WebSocket connections and channel subscriptions.
 */

import { useEffect, useRef, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { WS_URL, WS_PATH } from '../config/api';

interface UseWebSocketOptions {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
}

export function useWebSocket(options: UseWebSocketOptions = {}) {
  const socketRef = useRef<Socket | null>(null);
  const { onConnect, onDisconnect, onError } = options;

  useEffect(() => {
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
   */
  const subscribe = useCallback(
    (channel: string, handler: (data: any) => void) => {
      const socket = socketRef.current;
      if (!socket) {
        console.warn('Socket not initialized');
        return;
      }

      // Listen for events on this channel
      socket.on(channel, handler);

      // Notify server to add us to the channel's subscriber list
      socket.emit('subscribe', { channel });

      console.log(`Subscribed to channel: ${channel}`);
    },
    []
  );

  /**
   * Unsubscribe from a channel
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

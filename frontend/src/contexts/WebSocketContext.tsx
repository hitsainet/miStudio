/**
 * WebSocket Context Provider
 *
 * Provides a global WebSocket connection with robust subscription management.
 * Features:
 * - Automatic connection management
 * - Persistent event handlers
 * - Automatic resubscription on reconnect
 * - Subscription tracking and cleanup
 */

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from 'react';
import { io, Socket } from 'socket.io-client';
import { WS_URL, WS_PATH } from '../config/api';

interface WebSocketContextValue {
  socket: Socket | null;
  isConnected: boolean;
  subscribe: (channel: string) => void;
  unsubscribe: (channel: string) => void;
  on: (event: string, handler: (...args: any[]) => void) => void;
  off: (event: string, handler?: (...args: any[]) => void) => void;
}

const WebSocketContext = createContext<WebSocketContextValue | null>(null);

export function WebSocketProvider({ children }: { children: React.ReactNode }) {
  const socketRef = useRef<Socket | null>(null);
  const [isConnected, setIsConnected] = useState(false);

  // Track active subscriptions for auto-resubscribe
  const subscriptionsRef = useRef<Set<string>>(new Set());

  // Track event handlers for persistence
  const eventHandlersRef = useRef<Map<string, Set<(...args: any[]) => void>>>(new Map());

  // Queue for operations requested before socket is ready
  const pendingSubscriptionsRef = useRef<Set<string>>(new Set());
  const pendingHandlersRef = useRef<Array<{ event: string; handler: (...args: any[]) => void }>>(new Array());

  useEffect(() => {
    console.log('[WebSocket] Initializing connection to', WS_URL);

    // Create Socket.IO connection
    const socket = io(WS_URL, {
      path: WS_PATH,
      reconnection: true,
      reconnectionDelay: 1000,
      reconnectionDelayMax: 5000,
      reconnectionAttempts: Infinity, // Keep trying to reconnect
      transports: ['polling', 'websocket'], // Start with polling, upgrade to websocket
    });

    socketRef.current = socket;

    // Connection handlers
    socket.on('connect', () => {
      console.log('[WebSocket] Connected with ID:', socket.id);
      setIsConnected(true);

      // Process pending event handlers first
      if (pendingHandlersRef.current.length > 0) {
        console.log('[WebSocket] Processing', pendingHandlersRef.current.length, 'pending event handlers');
        pendingHandlersRef.current.forEach(({ event, handler }) => {
          if (!eventHandlersRef.current.has(event)) {
            eventHandlersRef.current.set(event, new Set());
          }
          eventHandlersRef.current.get(event)!.add(handler);
          socket.on(event, handler);
          console.log('[WebSocket] Added pending listener for event:', event);
        });
        pendingHandlersRef.current = [];
      }

      // Process pending subscriptions
      if (pendingSubscriptionsRef.current.size > 0) {
        console.log('[WebSocket] Processing', pendingSubscriptionsRef.current.size, 'pending subscriptions');
        pendingSubscriptionsRef.current.forEach(channel => {
          subscriptionsRef.current.add(channel);
          socket.emit('subscribe', { channel });
          console.log('[WebSocket] Subscribing to pending channel:', channel);
        });
        pendingSubscriptionsRef.current.clear();
      }

      // Resubscribe to all active channels (for reconnections)
      const subscriptions = Array.from(subscriptionsRef.current);
      if (subscriptions.length > 0) {
        console.log('[WebSocket] Resubscribing to', subscriptions.length, 'channels');
        subscriptions.forEach(channel => {
          console.log('[WebSocket] Resubscribing to channel:', channel);
          socket.emit('subscribe', { channel });
        });
      }

      // Re-attach all event handlers (for reconnections)
      eventHandlersRef.current.forEach((handlers, event) => {
        handlers.forEach(handler => {
          socket.on(event, handler);
        });
      });
    });

    socket.on('disconnect', (reason) => {
      console.log('[WebSocket] Disconnected:', reason);
      setIsConnected(false);
    });

    socket.on('connect_error', (error) => {
      console.error('[WebSocket] Connection error:', error);
    });

    socket.on('reconnect_attempt', (attemptNumber) => {
      console.log('[WebSocket] Reconnection attempt', attemptNumber);
    });

    socket.on('reconnect', (attemptNumber) => {
      console.log('[WebSocket] Reconnected after', attemptNumber, 'attempts');
    });

    // Listen for subscription confirmations
    socket.on('subscribed', (data: { channel: string }) => {
      console.log('[WebSocket] Subscription confirmed:', data.channel);
    });

    socket.on('unsubscribed', (data: { channel: string }) => {
      console.log('[WebSocket] Unsubscription confirmed:', data.channel);
    });

    // Cleanup on unmount
    return () => {
      console.log('[WebSocket] Cleaning up connection');
      socket.disconnect();
      socketRef.current = null;
    };
  }, []);

  // Subscribe to a channel
  const subscribe = useCallback((channel: string) => {
    const socket = socketRef.current;
    if (!socket || !socket.connected) {
      console.log('[WebSocket] Socket not ready, queuing subscription to:', channel);
      pendingSubscriptionsRef.current.add(channel);
      return;
    }

    // Track subscription for auto-resubscribe
    subscriptionsRef.current.add(channel);

    console.log('[WebSocket] Subscribing to channel:', channel);
    socket.emit('subscribe', { channel });
  }, []);

  // Unsubscribe from a channel
  const unsubscribe = useCallback((channel: string) => {
    const socket = socketRef.current;
    if (!socket) return;

    // Remove from tracked subscriptions
    subscriptionsRef.current.delete(channel);

    console.log('[WebSocket] Unsubscribing from channel:', channel);
    socket.emit('unsubscribe', { channel });
  }, []);

  // Add event listener with tracking
  const on = useCallback((event: string, handler: (...args: any[]) => void) => {
    const socket = socketRef.current;
    if (!socket || !socket.connected) {
      console.log('[WebSocket] Socket not ready, queuing event listener for:', event);
      pendingHandlersRef.current.push({ event, handler });
      return;
    }

    // Track handler for persistence across reconnects
    if (!eventHandlersRef.current.has(event)) {
      eventHandlersRef.current.set(event, new Set());
    }
    eventHandlersRef.current.get(event)!.add(handler);

    console.log('[WebSocket] Adding listener for event:', event);
    socket.on(event, handler);
  }, []);

  // Remove event listener
  const off = useCallback((event: string, handler?: (...args: any[]) => void) => {
    const socket = socketRef.current;
    if (!socket) return;

    if (handler) {
      // Remove specific handler
      const handlers = eventHandlersRef.current.get(event);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          eventHandlersRef.current.delete(event);
        }
      }
      socket.off(event, handler);
    } else {
      // Remove all handlers for this event
      eventHandlersRef.current.delete(event);
      socket.off(event);
    }

    console.log('[WebSocket] Removing listener for event:', event);
  }, []);

  const value: WebSocketContextValue = {
    socket: socketRef.current,
    isConnected,
    subscribe,
    unsubscribe,
    on,
    off,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}

export function useWebSocketContext() {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocketContext must be used within WebSocketProvider');
  }
  return context;
}

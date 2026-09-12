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

  // HOW MANY COMPONENTS WANT EACH CHANNEL.
  //
  // A subscription is a property of the SOCKET, not of the component that
  // asked for it: two components subscribing to one channel share a single
  // room membership on the server. Without a count, the first unmount emitted
  // `unsubscribe` and evicted the socket from the room while other components
  // were still listening — their handlers stayed registered and simply stopped
  // receiving anything.
  //
  // Observed 2026-08-26: the Models list card and the extraction modal both
  // subscribe to `models/{id}/extraction`. Closing the modal killed the card's
  // updates, so progress froze at the one event that arrived while both were
  // mounted ("queued, waiting for worker") and only a refresh recovered it.
  const channelRefCountsRef = useRef<Map<string, number>>(new Map());

  // Track event handlers for persistence
  const eventHandlersRef = useRef<Map<string, Set<(...args: any[]) => void>>>(new Map());

  // Queue for operations requested before socket is ready
  const pendingSubscriptionsRef = useRef<Set<string>>(new Set());
  const pendingHandlersRef = useRef<Array<{ event: string; handler: (...args: any[]) => void }>>([]);

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

      // NO RE-ATTACH HERE (MIS-E2E-120).
      //
      // This used to walk `eventHandlersRef` and `socket.on(...)` every handler
      // again, commented "for reconnections". socket.io does NOT detach
      // listeners on disconnect — the same Socket instance keeps them across
      // the whole reconnect cycle — so each reconnect ADDED a second
      // registration of every handler already attached.
      //
      // After N reconnects every event fired N+1 times. For a progress event
      // that is noise; for `extraction:completed` or a store action that
      // appends, it is N+1 duplicate effects from one server message. And
      // reconnects are routine, not exceptional: a pod restart or a laptop
      // waking does it.
      //
      // Handlers registered while disconnected are a different case, and they
      // are handled below via `pendingHandlersRef` — those genuinely are not
      // on the socket yet.

      // Process pending event handlers (queued while disconnected)
      // These are NOT in eventHandlersRef yet, so no double-registration
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

      // Resubscribe to all active channels (for reconnections)
      // Do this before processing pending subscriptions
      const existingSubscriptions = Array.from(subscriptionsRef.current);
      if (existingSubscriptions.length > 0) {
        console.warn(
          '[WebSocket] Resubscribing to', existingSubscriptions.length,
          'channels after reconnect:', existingSubscriptions.join(', '),
        );
        existingSubscriptions.forEach(channel => {
          socket.emit('subscribe', { channel });
          console.log('[WebSocket] Resubscribed to channel:', channel);
        });
      }

      // Process pending subscriptions (queued while disconnected)
      if (pendingSubscriptionsRef.current.size > 0) {
        console.log('[WebSocket] Processing', pendingSubscriptionsRef.current.size, 'pending subscriptions');
        pendingSubscriptionsRef.current.forEach(channel => {
          subscriptionsRef.current.add(channel);
          socket.emit('subscribe', { channel });
          console.log('[WebSocket] Subscribed to pending channel:', channel);
        });
        pendingSubscriptionsRef.current.clear();
      }
    });

    // The lifecycle events below use console.warn/error DELIBERATELY.
    //
    // vite marks console.log/debug/info/trace pure and the production build
    // drops them, so every diagnostic here was invisible in production. When a
    // user reported extraction progress freezing mid-run (2026-08-27) there
    // was nothing in their console to look at, and the difference between "the
    // socket dropped" and "the subscription was refused" could not be
    // established from the browser at all. These are the events someone is
    // actually asked to read back, which is the bar vite.config.ts sets for
    // keeping a log.
    socket.on('disconnect', (reason) => {
      console.warn('[WebSocket] Disconnected:', reason, '— live updates stop until reconnect');
      setIsConnected(false);
    });

    socket.on('connect_error', (error) => {
      console.error('[WebSocket] Connection error:', error);
    });

    socket.on('reconnect_attempt', (attemptNumber) => {
      console.warn('[WebSocket] Reconnection attempt', attemptNumber);
    });

    socket.on('reconnect', (attemptNumber) => {
      console.warn(
        '[WebSocket] Reconnected after', attemptNumber, 'attempts — resubscribing',
        Array.from(subscriptionsRef.current),
      );
    });

    // A REFUSED SUBSCRIPTION WAS COMPLETELY SILENT.
    //
    // The server emits `subscribe_error` when validate_channel rejects a
    // channel, and nothing here listened for it. The component believed it was
    // subscribed, received nothing forever, and looked identical to a healthy
    // subscription on a quiet channel.
    socket.on('subscribe_error', (data: { error?: string }) => {
      console.error(
        '[WebSocket] Subscription REFUSED by server:', data?.error ?? data,
        '— no events will arrive on that channel',
      );
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
    const claims = (channelRefCountsRef.current.get(channel) ?? 0) + 1;
    channelRefCountsRef.current.set(channel, claims);
    if (claims > 1) {
      // Already joined for this socket; another component wants it too.
      return;
    }

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
    const remaining = (channelRefCountsRef.current.get(channel) ?? 0) - 1;
    if (remaining > 0) {
      // Someone else is still listening — leaving the room would silence them.
      channelRefCountsRef.current.set(channel, remaining);
      return;
    }
    channelRefCountsRef.current.delete(channel);

    const socket = socketRef.current;

    // Remove from tracked subscriptions.
    subscriptionsRef.current.delete(channel);

    // AND FROM THE PENDING QUEUE (MIS-E2E-126).
    //
    // A channel subscribed while disconnected sits in `pendingSubscriptionsRef`
    // until the next connect. Unsubscribing did not clear it, so a channel the
    // user had abandoned was subscribed anyway on reconnect — and, because it
    // was never removed, on every reconnect after that. Compounds with
    // MIS-E2E-120: the abandoned channel's events then fired N+1 times too.
    pendingSubscriptionsRef.current.delete(channel);

    // `socket` may be null when the caller unsubscribes during teardown; the
    // refs above still had to be cleaned, which is why this check moved down.
    if (!socket) return;

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

    if (handler) {
      // Remove specific handler from tracking (always, even if socket is null)
      const handlers = eventHandlersRef.current.get(event);
      if (handlers) {
        handlers.delete(handler);
        if (handlers.size === 0) {
          eventHandlersRef.current.delete(event);
        }
      }
      // Also remove from pending handlers if it was queued
      pendingHandlersRef.current = pendingHandlersRef.current.filter(
        (h) => !(h.event === event && h.handler === handler)
      );
      // Remove from socket if connected
      if (socket) {
        socket.off(event, handler);
      }
    } else {
      // Remove all handlers for this event
      eventHandlersRef.current.delete(event);
      pendingHandlersRef.current = pendingHandlersRef.current.filter(
        (h) => h.event !== event
      );
      if (socket) {
        socket.off(event);
      }
    }

    console.log('[WebSocket] Removed listener for event:', event);
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

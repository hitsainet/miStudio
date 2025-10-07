import { io, Socket } from 'socket.io-client';

const WEBSOCKET_URL = 'ws://localhost:8001';
const RECONNECTION_DELAY_MS = 1000;
const RECONNECTION_DELAY_MAX_MS = 30000;
const RECONNECTION_ATTEMPTS = Infinity;

export class WebSocketClient {
  private socket: Socket | null = null;
  private reconnectAttempts = 0;

  connect(): Socket {
    if (this.socket?.connected) {
      return this.socket;
    }

    this.socket = io(WEBSOCKET_URL, {
      transports: ['websocket'],
      reconnection: true,
      reconnectionDelay: RECONNECTION_DELAY_MS,
      reconnectionDelayMax: RECONNECTION_DELAY_MAX_MS,
      reconnectionAttempts: RECONNECTION_ATTEMPTS,
    });

    this.socket.on('connect', () => {
      console.log('WebSocket connected');
      this.reconnectAttempts = 0;
    });

    this.socket.on('disconnect', (reason) => {
      console.log('WebSocket disconnected:', reason);
    });

    this.socket.on('connect_error', (error) => {
      console.error('WebSocket connection error:', error);
      this.reconnectAttempts++;
    });

    this.socket.on('reconnect', (attemptNumber) => {
      console.log(`WebSocket reconnected after ${attemptNumber} attempts`);
      this.reconnectAttempts = 0;
    });

    this.socket.on('reconnect_attempt', (attemptNumber) => {
      console.log(`WebSocket reconnection attempt ${attemptNumber}`);
    });

    this.socket.on('reconnect_error', (error) => {
      console.error('WebSocket reconnection error:', error);
    });

    this.socket.on('reconnect_failed', () => {
      console.error('WebSocket reconnection failed');
    });

    return this.socket;
  }

  disconnect(): void {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  subscribeToDatasetProgress(
    datasetId: string,
    callback: (data: any) => void
  ): void {
    if (!this.socket) {
      throw new Error('WebSocket not connected');
    }

    const channel = `datasets/${datasetId}/progress`;
    this.socket.on(channel, callback);
  }

  unsubscribeFromDatasetProgress(datasetId: string): void {
    if (!this.socket) {
      return;
    }

    const channel = `datasets/${datasetId}/progress`;
    this.socket.off(channel);
  }

  getSocket(): Socket | null {
    return this.socket;
  }

  isConnected(): boolean {
    return this.socket?.connected ?? false;
  }
}

// Singleton instance
export const websocketClient = new WebSocketClient();

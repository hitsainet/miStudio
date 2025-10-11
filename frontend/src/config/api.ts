/**
 * API Configuration
 *
 * Centralized configuration for API and WebSocket connections.
 * Uses environment variables with sensible defaults.
 */

// API base URL - should NOT include /api/v1 path as that's added by the store
// Can be overridden with VITE_API_URL environment variable
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://192.168.224.222:8000';

// WebSocket URL - can be overridden with VITE_WS_URL environment variable
// Must point to the same port as the API server (8000) where Socket.IO is mounted
export const WS_URL = import.meta.env.VITE_WS_URL || 'http://192.168.224.222:8000';

// WebSocket path
export const WS_PATH = '/ws/socket.io';

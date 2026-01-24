/**
 * API Configuration
 *
 * Centralized configuration for API and WebSocket connections.
 * Uses environment variables with sensible defaults.
 */

// API base URL - should NOT include /api/v1 path as that's added by the store
// Can be overridden with VITE_API_URL environment variable
// Empty string means use same-origin (through nginx proxy at dev-mistudio.mcslab.io)
export const API_BASE_URL = import.meta.env.VITE_API_URL || '';

// WebSocket URL - can be overridden with VITE_WS_URL environment variable
// Empty string means use same-origin (through nginx proxy)
export const WS_URL = import.meta.env.VITE_WS_URL || '';

// WebSocket path
export const WS_PATH = '/ws/socket.io';

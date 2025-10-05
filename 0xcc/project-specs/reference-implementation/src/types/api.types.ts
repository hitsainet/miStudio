/**
 * Common API Types
 *
 * Standardized types for API requests/responses, errors, and pagination
 */

/**
 * Standardized error response format
 *
 * Backend must return this format for all errors
 * Error codes should be consistent across all endpoints
 */
export interface ErrorResponse {
  error: {
    code: string; // Machine-readable error code
    message: string; // Human-readable error message
    details?: Record<string, any>; // Additional context (e.g., validation errors)
    retryable: boolean; // Can client retry this request?
    retry_after?: number; // Seconds to wait before retry (optional)
  };
}

/**
 * Error codes used throughout the API
 */
export enum ErrorCode {
  // Client errors (4xx)
  VALIDATION_ERROR = 'VALIDATION_ERROR',
  AUTHENTICATION_ERROR = 'AUTHENTICATION_ERROR',
  AUTHORIZATION_ERROR = 'AUTHORIZATION_ERROR',
  NOT_FOUND = 'NOT_FOUND',
  RATE_LIMIT_EXCEEDED = 'RATE_LIMIT_EXCEEDED',

  // Server errors (5xx)
  INTERNAL_ERROR = 'INTERNAL_ERROR',
  GPU_UNAVAILABLE = 'GPU_UNAVAILABLE',
  STORAGE_FULL = 'STORAGE_FULL',
  SERVICE_UNAVAILABLE = 'SERVICE_UNAVAILABLE',
}

/**
 * API Error class for throwing structured errors
 *
 * Usage:
 * throw new APIError('VALIDATION_ERROR', 'Invalid learning rate', false, { field: 'learningRate' });
 */
export class APIError extends Error {
  constructor(
    public code: string,
    message: string,
    public retryable: boolean,
    public details?: Record<string, any>,
    public retry_after?: number
  ) {
    super(message);
    this.name = 'APIError';
  }

  toJSON(): ErrorResponse {
    return {
      error: {
        code: this.code,
        message: this.message,
        details: this.details,
        retryable: this.retryable,
        retry_after: this.retry_after,
      },
    };
  }
}

/**
 * Pagination metadata
 *
 * Backend should include in all paginated responses
 */
export interface Pagination {
  page: number;
  limit: number;
  total: number;
  has_next: boolean;
  has_prev: boolean;
}

/**
 * Paginated response wrapper
 */
export interface PaginatedResponse<T> {
  data: T[];
  pagination: Pagination;
}

/**
 * Query parameters for pagination
 */
export interface PaginationParams {
  page?: number;
  limit?: number;
}

/**
 * Job status for long-running operations
 *
 * Backend pattern for async jobs:
 * 1. Client POSTs to start job
 * 2. Backend returns job_id immediately
 * 3. Client polls GET /api/jobs/:jobId for status
 * 4. Backend updates status: queued -> processing -> completed/error
 */
export interface JobStatus {
  job_id: string;
  status: 'queued' | 'processing' | 'completed' | 'error';
  progress?: number; // 0-100
  estimated_duration_seconds?: number;
  result?: any; // Result data when completed
  error?: string; // Error message when failed
}

/**
 * System metrics response
 *
 * Backend API: GET /api/system/metrics
 */
export interface SystemMetrics {
  gpu: {
    device: string;
    utilization: number; // 0-100 percentage
    memory_used_mb: number;
    memory_total_mb: number;
    temperature_c: number;
  };
  system: {
    cpu_percent: number;
    ram_used_gb: number;
    ram_total_gb: number;
    disk_used_gb: number;
    disk_total_gb: number;
  };
}

/**
 * Health check response
 *
 * Backend API: GET /api/health
 * Used to detect backend availability and feature flags
 */
export interface HealthCheckResponse {
  status: 'healthy' | 'degraded';
  version: string;
  features: {
    datasets: boolean;
    training: boolean;
    steering: boolean;
    // Add more feature flags as needed
  };
}

/**
 * API request configuration
 */
export interface APIRequestConfig {
  method: 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
  endpoint: string;
  body?: any;
  params?: Record<string, string | number | boolean>;
  headers?: Record<string, string>;
  signal?: AbortSignal; // For request cancellation
  timeout?: number; // Milliseconds
}

/**
 * WebSocket message base type
 */
export interface WebSocketMessage<T = any> {
  type: string;
  data: T;
  timestamp?: string;
}

/**
 * WebSocket connection status
 */
export type WebSocketStatus = 'connecting' | 'connected' | 'disconnecting' | 'disconnected' | 'error';

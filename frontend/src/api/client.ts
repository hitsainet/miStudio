/**
 * Shared API client utilities.
 *
 * This module provides common functionality for making HTTP requests to the backend API.
 * All API modules (datasets, models, etc.) should use these shared utilities.
 */

import { API_BASE_URL } from '../config/api';

// API endpoints are prefixed with /api/v1
export const API_V1_BASE = `${API_BASE_URL}/api/v1`;

/**
 * Custom error class for API errors.
 * Includes HTTP status code and structured error details from the backend.
 */
export class APIError extends Error {
  status: number;
  detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}

/**
 * Fetch helper with authentication and error handling.
 *
 * This function handles:
 * - Authentication header injection (Bearer token from localStorage)
 * - Content-Type headers
 * - HTTP error status codes
 * - JSON response parsing
 * - 204 No Content responses
 *
 * @template T - The expected response type
 * @param endpoint - API endpoint path (e.g., "/models" or "/datasets/123")
 * @param options - Standard fetch options (method, body, headers, etc.)
 * @returns Promise resolving to the typed response data
 * @throws {APIError} If the request fails or returns an error status
 *
 * @example
 * ```typescript
 * // GET request
 * const models = await fetchAPI<ModelListResponse>('/models?page=1');
 *
 * // POST request
 * const newModel = await fetchAPI<Model>('/models/download', {
 *   method: 'POST',
 *   body: JSON.stringify({ repo_id: 'gpt2' }),
 * });
 *
 * // DELETE request
 * await fetchAPI<void>('/models/m_abc123', {
 *   method: 'DELETE',
 * });
 * ```
 */
export async function fetchAPI<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const token = localStorage.getItem('auth_token');

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    ...(options.headers as Record<string, string>),
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_V1_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({
      detail: `HTTP error! status: ${response.status}`,
    }));
    throw new APIError(
      response.status,
      error.detail || error.message || 'API request failed'
    );
  }

  // Handle 204 No Content responses (e.g., DELETE)
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

/**
 * Build query string from parameters object.
 *
 * This function filters out undefined values and properly encodes parameters
 * for use in URLs.
 *
 * @param params - Object containing query parameters
 * @returns Query string (e.g., "page=1&limit=50") or empty string if no params
 *
 * @example
 * ```typescript
 * const params = { page: 1, limit: 50, search: 'gpt2', status: undefined };
 * const query = buildQueryString(params);
 * // Result: "page=1&limit=50&search=gpt2"
 *
 * const endpoint = `/models${query ? `?${query}` : ''}`;
 * // Result: "/models?page=1&limit=50&search=gpt2"
 * ```
 */
export function buildQueryString(params: Record<string, any>): string {
  const queryParams = new URLSearchParams();

  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== null) {
      queryParams.append(key, String(value));
    }
  });

  return queryParams.toString();
}

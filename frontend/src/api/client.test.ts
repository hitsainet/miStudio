/**
 * Unit tests for shared API client utilities.
 *
 * This module tests the fetchAPI function and buildQueryString helper
 * to ensure correct API communication and error handling.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { fetchAPI, buildQueryString, APIError } from './client';

// Mock API config
vi.mock('../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
}));

// Mock fetch globally
global.fetch = vi.fn();

describe('buildQueryString', () => {
  it('should build query string from object', () => {
    const params = {
      skip: 0,
      limit: 50,
      search: 'test',
      status: 'ready',
    };
    const result = buildQueryString(params);
    expect(result).toBe('skip=0&limit=50&search=test&status=ready');
  });

  it('should handle empty object', () => {
    const result = buildQueryString({});
    expect(result).toBe('');
  });

  it('should skip undefined values', () => {
    const params = {
      skip: 0,
      limit: undefined,
      search: 'test',
    };
    const result = buildQueryString(params);
    expect(result).toBe('skip=0&search=test');
  });

  it('should skip null values', () => {
    const params = {
      skip: 0,
      limit: null,
      search: 'test',
    };
    const result = buildQueryString(params);
    expect(result).toBe('skip=0&search=test');
  });

  it('should handle special characters with URL encoding', () => {
    const params = {
      search: 'test query with spaces',
      filter: 'name=value&other=thing',
    };
    const result = buildQueryString(params);
    // URLSearchParams uses + for spaces (RFC 3986)
    expect(result).toBe(
      'search=test+query+with+spaces&filter=name%3Dvalue%26other%3Dthing'
    );
  });

  it('should handle boolean values', () => {
    const params = {
      active: true,
      archived: false,
    };
    const result = buildQueryString(params);
    expect(result).toBe('active=true&archived=false');
  });

  it('should handle number values including zero', () => {
    const params = {
      page: 0,
      count: 100,
      rating: 4.5,
    };
    const result = buildQueryString(params);
    expect(result).toBe('page=0&count=100&rating=4.5');
  });
});

describe('APIError', () => {
  it('should create error with status and detail', () => {
    const error = new APIError(404, 'Not found');
    expect(error).toBeInstanceOf(Error);
    expect(error.status).toBe(404);
    expect(error.detail).toBe('Not found');
    expect(error.message).toBe('Not found');
  });

  it('should have correct name property', () => {
    const error = new APIError(500, 'Server error');
    expect(error.name).toBe('APIError');
  });
});

describe('fetchAPI', () => {
  beforeEach(() => {
    // Reset fetch mock
    vi.clearAllMocks();
  });

  afterEach(() => {
    // Clean up
    vi.clearAllTimers();
  });

  it('should make GET request successfully', async () => {
    const mockData = { id: '123', name: 'Test Dataset' };
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({ data: mockData }),
    });

    const result = await fetchAPI<{ data: any }>('/datasets/123');

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/datasets/123',
      expect.objectContaining({
        headers: { 'Content-Type': 'application/json' },
      })
    );
    expect(result).toEqual({ data: mockData });
  });

  it('should make POST request with body', async () => {
    const mockRequest = { repo_id: 'test/dataset' };
    const mockResponse = { id: '456', status: 'downloading' };

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    });

    const result = await fetchAPI<any>('/datasets/download', {
      method: 'POST',
      body: JSON.stringify(mockRequest),
    });

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/datasets/download',
      expect.objectContaining({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(mockRequest),
      })
    );
    expect(result).toEqual(mockResponse);
  });

  it('should handle DELETE request', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 204,
    });

    await fetchAPI<void>('/datasets/123', { method: 'DELETE' });

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/datasets/123',
      expect.objectContaining({
        method: 'DELETE',
      })
    );
  });

  it('should handle 204 No Content response', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 204,
    });

    const result = await fetchAPI<void>('/datasets/123', { method: 'DELETE' });

    expect(result).toBeUndefined();
  });

  it('should throw APIError on 404', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 404,
      json: async () => ({ detail: 'Dataset not found' }),
    });

    try {
      await fetchAPI('/datasets/nonexistent');
      expect.fail('Should have thrown error');
    } catch (error) {
      expect(error).toBeInstanceOf(APIError);
      if (error instanceof APIError) {
        expect(error.status).toBe(404);
        expect(error.detail).toBe('Dataset not found');
        expect(error.message).toBe('Dataset not found');
      }
    }
  });

  it('should throw APIError on 500', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'Internal server error' }),
    });

    await expect(fetchAPI('/datasets')).rejects.toThrow(APIError);

    try {
      await fetchAPI('/datasets');
    } catch (error) {
      if (error instanceof APIError) {
        expect(error.status).toBe(500);
        expect(error.detail).toBe('Internal server error');
      }
    }
  });

  it('should handle network error', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

    await expect(fetchAPI('/datasets')).rejects.toThrow('Network error');
  });

  it('should handle error response without detail field', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 400,
      statusText: 'Bad Request',
      json: async () => ({ message: 'Invalid input' }),
    });

    try {
      await fetchAPI('/datasets');
      expect.fail('Should have thrown error');
    } catch (error) {
      expect(error).toBeInstanceOf(APIError);
      if (error instanceof APIError) {
        expect(error.status).toBe(400);
        // Should use message field as fallback
        expect(error.detail).toBe('Invalid input');
      }
    }
  });

  it('should handle JSON parse error in error response', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: 'Internal Server Error',
      json: async () => {
        throw new Error('Invalid JSON');
      },
    });

    try {
      await fetchAPI('/datasets');
      expect.fail('Should have thrown error');
    } catch (error) {
      expect(error).toBeInstanceOf(APIError);
      if (error instanceof APIError) {
        expect(error.status).toBe(500);
        // Should use fallback message when JSON parsing fails
        expect(error.detail).toBe('HTTP error! status: 500');
      }
    }
  });

  it('should include custom headers', async () => {
    const mockData = { id: '123' };
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockData,
    });

    await fetchAPI<any>('/datasets', {
      headers: { 'X-Custom-Header': 'custom-value' },
    });

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/datasets',
      expect.objectContaining({
        headers: {
          'Content-Type': 'application/json',
          'X-Custom-Header': 'custom-value',
        },
      })
    );
  });

  it('should handle PATCH request', async () => {
    const mockUpdate = { name: 'Updated Name' };
    const mockResponse = { id: '123', name: 'Updated Name' };

    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => mockResponse,
    });

    const result = await fetchAPI<any>('/datasets/123', {
      method: 'PATCH',
      body: JSON.stringify(mockUpdate),
    });

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/datasets/123',
      expect.objectContaining({
        method: 'PATCH',
        body: JSON.stringify(mockUpdate),
      })
    );
    expect(result).toEqual(mockResponse);
  });

  it('should construct full URL with API_V1_BASE', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({}),
    });

    await fetchAPI('/datasets');

    expect(global.fetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/v1/datasets',
      expect.any(Object)
    );
  });
});

/**
 * Feature 040 FR-40: a structured refusal reaches the caller as data. `APIError.data` carries the server's
 * `{code, message, …, next}` so a screen can act on the code (a stale base version offers a reload) instead
 * of matching the message text.
 */
import { afterEach, describe, expect, it, vi } from 'vitest';

import { APIError, fetchAPI } from '../client';

afterEach(() => vi.unstubAllGlobals());

describe('a structured refusal keeps its fields', () => {
  it('exposes code and details on APIError.data', async () => {
    const detail = { code: 'stale_base_version', message: 'changed', current_version: 4, base_version: 3, next: [] };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail }), { status: 409 })));
    const error = await fetchAPI('/x').catch((e: unknown) => e);
    expect(error).toBeInstanceOf(APIError);
    expect((error as APIError).status).toBe(409);
    expect((error as APIError).detail).toBe('changed');
    expect((error as APIError).data).toEqual(detail);
  });

  it('a plain or validation error has no data', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: 'nope' }), { status: 400 })));
    const error = (await fetchAPI('/x').catch((e: unknown) => e)) as APIError;
    expect(error.detail).toBe('nope');
    expect(error.data).toBeUndefined();
  });
});

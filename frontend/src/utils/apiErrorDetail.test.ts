/**
 * apiErrorDetail returns the backend's reason in each shape FastAPI sends (review R2, B6).
 */

import { describe, it, expect } from 'vitest';
import { apiErrorDetail } from './apiErrorDetail';

const axiosError = (data: unknown) =>
  Object.assign(new Error('Request failed with status code 422'), { response: { status: 422, data } });

describe('apiErrorDetail', () => {
  it('returns a string detail', () => {
    expect(apiErrorDetail(axiosError({ detail: 'it has no activations for [\'mlp\']' }), 'fallback'))
      .toBe("it has no activations for ['mlp']");
  });

  it("joins FastAPI's list of validation errors with each field's location, without 'body'", () => {
    const detail = [
      { loc: ['body', 'hyperparameters', 'hook_types', 0], msg: "Input should be 'residual', 'mlp' or 'attention'", type: 'enum' },
      { loc: ['query', 'gpu'], msg: 'Field required', type: 'missing' },
    ];
    expect(apiErrorDetail(axiosError({ detail }), 'fallback')).toBe(
      "hyperparameters.hook_types.0: Input should be 'residual', 'mlp' or 'attention'; query.gpu: Field required",
    );
  });

  it('reads an object detail by its message', () => {
    expect(apiErrorDetail(axiosError({ detail: { message: 'busy', code: 'GPU_BUSY' } }), 'fallback')).toBe('busy');
  });

  it('falls back to a message field, then to the fallback, never to the status line', () => {
    expect(apiErrorDetail(axiosError({ message: 'Validation failed' }), 'fallback')).toBe('Validation failed');
    expect(apiErrorDetail(axiosError({}), 'Failed to create training')).toBe('Failed to create training');
    expect(apiErrorDetail(new Error('Network Error'), 'Failed to create training')).toBe('Failed to create training');
    expect(apiErrorDetail(undefined, 'fallback')).toBe('fallback');
  });

  it('prefers detail over message when both are present', () => {
    expect(apiErrorDetail(axiosError({ detail: 'the reason', message: 'other' }), 'fallback')).toBe('the reason');
  });
});

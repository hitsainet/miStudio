/**
 * Feature 035 FR-8: the request body carries `revision` only when one was typed.
 * MUTATION CONTROL C12: delete the `body.revision = …` line → the first test fails.
 */
import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { useDatasetsStore, errorDetailText } from './datasetsStore';
import { DatasetStatus } from '../types/dataset';

const fetchMock = vi.fn();
globalThis.fetch = fetchMock as unknown as typeof fetch;

function lastBody(): Record<string, unknown> {
  const download = fetchMock.mock.calls.find((c) => String(c[0]).endsWith('/api/v1/datasets/download'));
  if (!download) throw new Error('no download request was made');
  return JSON.parse((download[1] as RequestInit).body as string);
}

describe('downloadDataset revision', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      json: async () => ({ id: 'r1', name: 'ds', status: DatasetStatus.DOWNLOADING, revision: 'a'.repeat(40) }),
    });
    useDatasetsStore.setState({ datasets: [], loading: false, error: null });
  });
  afterEach(() => vi.useRealTimers());

  it('sends a typed revision, trimmed', async () => {
    await useDatasetsStore.getState().downloadDataset('org/ds', undefined, 'train', undefined, ' v1.0 ');
    expect(lastBody().revision).toBe('v1.0');
  });

  it.each([undefined, '', '   '])('omits a blank revision (%j)', async (rev) => {
    await useDatasetsStore.getState().downloadDataset('org/ds', undefined, undefined, undefined, rev);
    expect('revision' in lastBody()).toBe(false);
  });

  it('returns the created dataset, carrying the resolved commit', async () => {
    const created = await useDatasetsStore.getState().downloadDataset('org/ds');
    expect(created.revision).toBe('a'.repeat(40));
  });

  it('a 409 string detail becomes the error message verbatim', async () => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValueOnce({
      ok: false,
      json: async () => ({ detail: 'Dataset org/ds at ' + 'a'.repeat(40) + ' already exists (dataset x)' }),
    });
    await expect(useDatasetsStore.getState().downloadDataset('org/ds')).rejects.toThrow('already exists (dataset x)');
  });
});

describe('errorDetailText', () => {
  it('renders a validation list as text, not [object Object]', () => {
    expect(errorDetailText([{ loc: ['body', 'revision'], msg: 'too long' }])).toBe('too long');
  });
  it('passes a string through and reads a message field', () => {
    expect(errorDetailText('x')).toBe('x');
    expect(errorDetailText({ code: 'c', message: 'm' })).toBe('m');
    expect(errorDetailText(undefined)).toBe('');
  });
});

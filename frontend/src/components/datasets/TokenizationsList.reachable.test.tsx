/**
 * Reachability: the form's controls must reach the request.
 *
 * WHY THIS FILE EXISTS SEPARATELY from TokenizationsList.payload.test.ts.
 * That file tests `buildTokenizationPayload` in isolation and is worth having —
 * but a mutation control proved it insufficient. Replacing
 * `buildTokenizationPayload({...})` in `handleCreate` with a bare object
 * literal `({...})` left the builder perfect, its fourteen tests green, and
 * `tsc` silent (the store types `params: any`), while the component shipped
 * camelCase keys the backend ignores. The mutation survived.
 *
 * That is this repo's standing rule verbatim: a capability is not shipped until
 * a test FAILS when its wiring is removed, and "assert the payload, not that it
 * was called". So this drives the real component and asserts what
 * `createTokenization` actually receives.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { TokenizationsList } from './TokenizationsList';

const createTokenization = vi.fn().mockResolvedValue(undefined);

vi.mock('../../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
  WS_URL: '',
  WS_PATH: '/ws/socket.io',
}));

vi.mock('../../stores/datasetsStore', () => ({
  useDatasetsStore: () => ({
    tokenizations: {},
    tokenizationProgress: {},
    fetchTokenizations: vi.fn(),
    deleteTokenization: vi.fn(),
    cancelTokenization: vi.fn(),
    createTokenization,
    error: null,
  }),
}));

vi.mock('../../stores/modelsStore', () => ({
  useModelsStore: () => ({
    models: [
      { id: 'm_88d55564', name: 'LFM2.5-1.2B-Instruct', status: 'ready' },
    ],
    fetchModels: vi.fn(),
  }),
}));

vi.mock('../../hooks/useTokenizationWebSocket', () => ({
  useTokenizationWebSocket: vi.fn(),
}));

/** The samples fetch that populates the column dropdown. */
const sampleColumns = (columns: string[]) =>
  vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      data: [
        { index: 0, data: Object.fromEntries(columns.map((c) => [c, 'x'])) },
      ],
    }),
  });

async function openCreateForm() {
  fireEvent.click(screen.getByRole('button', { name: /add tokenization/i }));
  await screen.findByLabelText(/text column/i);
}

describe('TokenizationsList wiring', () => {
  beforeEach(() => {
    createTokenization.mockClear();
    globalThis.fetch = sampleColumns(['Headline', 'Journalists', 'Article']) as never;
  });

  it('offers the dataset’s real columns, read from a sample', async () => {
    render(<TokenizationsList datasetId="ds_bloomberg" />);
    await openCreateForm();

    const select = screen.getByLabelText(/text column/i) as HTMLSelectElement;
    const options = Array.from(select.options).map((o) => o.value);

    expect(options).toEqual(['', 'Headline', 'Journalists', 'Article']);
  });

  it('sends text_column, chat_format and pack_sequences that the operator set', async () => {
    render(<TokenizationsList datasetId="ds_bloomberg" />);
    await openCreateForm();

    fireEvent.change(screen.getByLabelText(/model/i), {
      target: { value: 'm_88d55564' },
    });
    fireEvent.change(screen.getByLabelText(/max sequence length/i), {
      target: { value: '2048' },
    });
    fireEvent.change(screen.getByLabelText(/text column/i), {
      target: { value: 'Article' },
    });
    fireEvent.change(screen.getByLabelText(/conversation rendering/i), {
      target: { value: 'none' },
    });
    fireEvent.click(screen.getByLabelText(/pack sequences/i));

    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));

    await waitFor(() => expect(createTokenization).toHaveBeenCalledTimes(1));

    const [datasetId, modelId, payload] = createTokenization.mock.calls[0];
    expect(datasetId).toBe('ds_bloomberg');
    expect(modelId).toBe('m_88d55564');

    // The keys the BACKEND reads. camelCase here means the request is ignored.
    expect(payload).toMatchObject({
      max_length: 2048,
      text_column: 'Article',
      chat_format: 'none',
      pack_sequences: true,
    });
  });

  it('NEGATIVE CONTROL — an untouched form sends no text_column and does not pack', async () => {
    render(<TokenizationsList datasetId="ds_bloomberg" />);
    await openCreateForm();

    fireEvent.change(screen.getByLabelText(/model/i), {
      target: { value: 'm_88d55564' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));

    await waitFor(() => expect(createTokenization).toHaveBeenCalledTimes(1));

    const payload = createTokenization.mock.calls[0][2];
    expect('text_column' in payload).toBe(false);
    expect(payload.pack_sequences).toBe(false);
    expect(payload.chat_format).toBe('auto');
    expect(payload.enable_cleaning).toBe(false);
  });

  it('sends enable_cleaning=true only when Clean Text is ticked', async () => {
    render(<TokenizationsList datasetId="ds_bloomberg" />);
    await openCreateForm();

    fireEvent.change(screen.getByLabelText(/model/i), {
      target: { value: 'm_88d55564' },
    });
    fireEvent.click(screen.getByLabelText(/clean text/i));
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));

    await waitFor(() => expect(createTokenization).toHaveBeenCalledTimes(1));
    expect(createTokenization.mock.calls[0][2]).toMatchObject({ enable_cleaning: true });
  });

  it('degrades to a text input when the columns cannot be read', async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error('boom')) as never;
    render(<TokenizationsList datasetId="ds_bloomberg" />);
    await openCreateForm();

    const field = screen.getByLabelText(/text column/i);
    expect(field.tagName).toBe('INPUT');

    fireEvent.change(field, { target: { value: 'Article' } });
    fireEvent.change(screen.getByLabelText(/model/i), {
      target: { value: 'm_88d55564' },
    });
    fireEvent.click(screen.getByRole('button', { name: /^create$/i }));

    await waitFor(() => expect(createTokenization).toHaveBeenCalledTimes(1));
    expect(createTokenization.mock.calls[0][2]).toMatchObject({
      text_column: 'Article',
    });
  });
});

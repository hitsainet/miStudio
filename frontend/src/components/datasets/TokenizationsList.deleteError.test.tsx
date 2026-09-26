/**
 * A failed delete must say so on screen.
 *
 * Reported 2026-08-25: "why the fuck can't I delete the Bloomberg
 * Tokenization?" Every DELETE was returning 500 (a NameError in the handler),
 * and the UI showed nothing at all -- `handleDelete` swallowed the rejection
 * into `console.error`, and the store's `error` field was never rendered by
 * this component. Three attempts, three 500s, no feedback. A silent failure is
 * indistinguishable from a dead button.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { TokenizationsList } from './TokenizationsList';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { TokenizationStatus } from '../../types/dataset';

vi.mock('../../hooks/useTokenizationWebSocket', () => ({
  useTokenizationWebSocket: () => undefined,
}));

const DATASET_ID = 'ds-1';

const tokenization = {
  id: 'tok_x_m_1_512',
  dataset_id: DATASET_ID,
  model_id: 'm_1',
  max_length: 512,
  status: TokenizationStatus.READY,
  progress: 100,
  tokenizer_repo_id: 'google/gemma-4-12B-it',
  num_tokens: 1000,
  error_message: null,
} as never;

function seedStores(deleteTokenization: () => Promise<void>) {
  useDatasetsStore.setState({
    tokenizations: { [DATASET_ID]: [tokenization] },
    tokenizationProgress: {},
    error: null,
    fetchTokenizations: vi.fn().mockResolvedValue(undefined),
    deleteTokenization,
    cancelTokenization: vi.fn().mockResolvedValue(undefined),
    createTokenization: vi.fn().mockResolvedValue(undefined),
  } as never);

  useModelsStore.setState({
    models: [],
    fetchModels: vi.fn().mockResolvedValue(undefined),
  } as never);
}

describe('TokenizationsList delete failures', () => {
  beforeEach(() => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    vi.spyOn(console, 'error').mockImplementation(() => undefined);
  });

  it('shows the reason when the delete is rejected', async () => {
    const deleteTokenization = vi
      .fn()
      .mockRejectedValue(new Error('Failed to delete tokenization'));
    seedStores(deleteTokenization);

    render(<TokenizationsList datasetId={DATASET_ID} />);

    const button = await screen.findByTitle(/delete tokenization/i);
    await userEvent.click(button);

    await waitFor(() => expect(deleteTokenization).toHaveBeenCalled());

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/failed to delete tokenization/i);
  });

  it('shows nothing when the delete succeeds', async () => {
    seedStores(vi.fn().mockResolvedValue(undefined));

    render(<TokenizationsList datasetId={DATASET_ID} />);

    const button = await screen.findByTitle(/delete tokenization/i);
    await userEvent.click(button);

    await waitFor(() =>
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    );
  });
});

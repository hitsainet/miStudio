/**
 * Reachability: the mixture controls must reach the training request.
 *
 * WHY THIS EXISTS SEPARATELY from utils/trainingMixture.test.ts. That file
 * tests the serialiser and is worth having — but the identical split on the
 * tokenization form proved a serialiser test insufficient: replacing
 * `buildTokenizationPayload({...})` with a bare object literal left every
 * builder test green and `tsc` silent, because the store types its params as
 * `any`. The mutation survived, and the capability shipped unreachable.
 *
 * `dataset_weights` and `holdout_fraction` had ZERO non-test references in the
 * frontend when this was written. Both are backend-schema fields the UI never
 * sent — the mixture weighting is the whole point of the corpus work, and the
 * held-out split is the only thing standing between "FVU 0.02" and "FVU 0.02
 * in-sample, which is not a number about generalisation at all".
 *
 * So this drives the real panel and asserts what `createTraining` receives.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { TrainingPanel } from './TrainingPanel';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTrainingWebSocket } from '../../hooks/useTrainingWebSocket';
import { useDeletionProgressWebSocket } from '../../hooks/useDeletionProgressWebSocket';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { SAEArchitectureType } from '../../types/training';

vi.mock('../../stores/trainingsStore');
vi.mock('../../stores/modelsStore');
vi.mock('../../stores/datasetsStore');
vi.mock('../../hooks/useTrainingWebSocket');
vi.mock('../../hooks/useDeletionProgressWebSocket');
vi.mock('../../contexts/WebSocketContext');

const DS_WEB = 'ds_owt';
const DS_CHAT = 'ds_hermes';
const EXT_WEB = 'ext_web_20260911';
const EXT_CHAT = 'ext_chat_20260911';

const EXTRACTIONS = [
  {
    extraction_id: EXT_WEB, dataset_id: DS_WEB, status: 'completed',
    layer_indices: [12], num_samples_processed: 10000,
    created_at: '2026-09-11T10:00:00Z',
    statistics: { layer_12_residual: { shape: [10000, 2048, 2048] } },
  },
  {
    extraction_id: EXT_CHAT, dataset_id: DS_CHAT, status: 'completed',
    layer_indices: [12], num_samples_processed: 10000,
    created_at: '2026-09-11T11:00:00Z',
    statistics: { layer_12_residual: { shape: [10000, 2048, 2048] } },
  },
];

const createTraining = vi.fn().mockResolvedValue(undefined);
const updateConfig = vi.fn();

type Mocked = { mockReturnValue: (value: unknown) => void };

function setup(config: Record<string, unknown>) {
  (useTrainingsStore as never as Mocked).mockReturnValue({
    trainings: [], config, updateConfig,
    fetchTrainings: vi.fn(), fetchTraining: vi.fn(),
    createTraining, deleteTraining: vi.fn(),
    statusFilter: 'all', setStatusFilter: vi.fn(),
    statusCounts: { all: 0, running: 0, completed: 0, failed: 0, pending: 0 },
    isLoading: false, error: null,
  });
  (useModelsStore as never as Mocked).mockReturnValue({
    models: [{
      id: 'm_88d55564', name: 'LFM2.5-1.2B-Instruct', status: 'ready',
      architecture_config: { num_hidden_layers: 16, hidden_size: 2048 },
    }],
    fetchModels: vi.fn(),
  });
  (useDatasetsStore as never as Mocked).mockReturnValue({
    datasets: [
      { id: DS_WEB, name: 'OpenWebText-2M', status: 'ready' },
      { id: DS_CHAT, name: 'OpenHermes-2.5', status: 'ready' },
    ],
    fetchDatasets: vi.fn(),
  });
  (useTrainingWebSocket as never as Mocked).mockReturnValue({});
  (useDeletionProgressWebSocket as never as Mocked).mockReturnValue({});
  (useWebSocketContext as never as Mocked).mockReturnValue({
    on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(),
    isConnected: true,
  });

  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ extractions: EXTRACTIONS }),
  }) as never;
}

const baseConfig = {
  model_id: 'm_88d55564',
  dataset_ids: [DS_WEB, DS_CHAT],
  extraction_ids: [EXT_WEB, EXT_CHAT],
  training_layers: [12],
  hook_types: ['residual'],
  architecture_type: SAEArchitectureType.JUMPRELU,
  hidden_dim: 2048,
  latent_dim: 16384,
  learning_rate: 1e-4,
  batch_size: 2048,
  total_steps: 50000,
};

async function startTraining() {
  const button = await screen.findByRole('button', { name: /Start Training/i });
  await waitFor(() => expect(button).not.toBeDisabled());
  fireEvent.click(button);
  await waitFor(() => expect(createTraining).toHaveBeenCalledTimes(1));
  return createTraining.mock.calls[0][0].hyperparameters;
}

describe('the mixture controls reach the request', () => {
  beforeEach(() => {
    createTraining.mockClear();
    updateConfig.mockClear();
  });

  it('sends dataset_weights in extraction_ids order', async () => {
    setup({
      ...baseConfig,
      dataset_weights_by_extraction: { [EXT_CHAT]: 0.25, [EXT_WEB]: 0.75 },
    });
    render(<TrainingPanel />);

    const hp = await startTraining();
    // Ordered by extraction_ids ([web, chat]) — NOT by the map's key order,
    // which is chat-first above precisely so a naive implementation fails here.
    expect(hp.dataset_weights).toEqual([0.75, 0.25]);
  });

  it('sends holdout_fraction', async () => {
    setup({ ...baseConfig, holdout_fraction: 0.05 });
    render(<TrainingPanel />);

    const hp = await startTraining();
    expect(hp.holdout_fraction).toBe(0.05);
  });

  it('NEGATIVE CONTROL — an untouched config sends neither key', async () => {
    // The historical request must stay reproducible, or every existing
    // training becomes incomparable for a reason nobody chose.
    setup({ ...baseConfig });
    render(<TrainingPanel />);

    const hp = await startTraining();
    expect('dataset_weights' in hp).toBe(false);
    expect('holdout_fraction' in hp).toBe(false);
  });

  it('renders one weight input per selected extraction, labelled by source', async () => {
    setup({ ...baseConfig });
    render(<TrainingPanel />);

    fireEvent.click(screen.getByText(/Advanced Configuration/i));

    const web = await screen.findByLabelText('OpenWebText-2M');
    const chat = await screen.findByLabelText('OpenHermes-2.5');
    expect(web.id).toBe(`mixture-weight-${EXT_WEB}`);
    expect(chat.id).toBe(`mixture-weight-${EXT_CHAT}`);
  });

  it('offers a held-out fraction control', async () => {
    setup({ ...baseConfig });
    render(<TrainingPanel />);

    fireEvent.click(screen.getByText(/Advanced Configuration/i));
    expect(await screen.findByLabelText(/held-out fraction/i)).toBeTruthy();
  });

  it('does not offer mixture weights for a single source', async () => {
    // One weight normalises to 1 whatever it is, so the control would imply a
    // choice that does not exist.
    setup({
      ...baseConfig,
      dataset_ids: [DS_WEB],
      extraction_ids: [EXT_WEB],
    });
    render(<TrainingPanel />);

    fireEvent.click(screen.getByText(/Advanced Configuration/i));
    await screen.findByLabelText(/held-out fraction/i);
    expect(screen.queryByLabelText('OpenWebText-2M')).toBeNull();
  });
});

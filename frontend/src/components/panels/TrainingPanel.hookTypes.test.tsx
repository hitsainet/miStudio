/**
 * A training may only ask for hook types every selected extraction holds, and
 * when the server refuses one anyway the user must see its reason (review R2, B6).
 *
 * The Hook Types buttons offered MLP Output and Attention Output whatever the
 * selected extraction had captured. The server refuses such a training at create
 * with a 422 naming the missing hooks, but the store read
 * `error.response.data.message` (FastAPI sends `detail`) and the panel alerted
 * axios's `err.message`, so the user saw "Request failed with status code 422".
 *
 * MUTATION CONTROLS (one line broken at a time, this file run, bytes restored,
 * sha256 and `git diff` verified clean; the table is in the R2-E record):
 *   F1 `&& extractionHookMismatches.length === 0` removed from isFormValid
 *        -> "blocks Start Training ..." (both the single and the multi case)
 *   F2 the memo's `missing` keeps hooks the extraction HAS -> every alert test
 *   F3 the alert's apiErrorDetail replaced with err.message -> "shows the server's reason"
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

const DS_CODE = 'ds_code';
const DS_OWT = 'ds_owt';
const EXT_CODE_RES = 'ext_code_residual';
const EXT_CODE_RES_MLP = 'ext_code_residual_mlp';
const EXT_OWT_RES = 'ext_owt_residual';

const EXTRACTIONS = [
  {
    extraction_id: EXT_CODE_RES, dataset_id: DS_CODE, status: 'completed',
    layer_indices: [11], hook_types: ['residual'], num_samples_processed: 100,
    created_at: '2026-09-14T10:00:00Z',
  },
  {
    extraction_id: EXT_CODE_RES_MLP, dataset_id: DS_CODE, status: 'completed',
    layer_indices: [11], hook_types: ['residual', 'mlp'], num_samples_processed: 100,
    created_at: '2026-09-14T11:00:00Z',
  },
  {
    extraction_id: EXT_OWT_RES, dataset_id: DS_OWT, status: 'completed',
    layer_indices: [11], hook_types: ['residual'], num_samples_processed: 100,
    created_at: '2026-09-14T12:00:00Z',
  },
];

type Mocked = { mockReturnValue: (value: unknown) => void };

function setup(config: Record<string, unknown>, createTraining = vi.fn()) {
  (useTrainingsStore as never as Mocked).mockReturnValue({
    trainings: [], config, updateConfig: vi.fn(),
    fetchTrainings: vi.fn(), fetchTraining: vi.fn(),
    createTraining, deleteTraining: vi.fn(),
    statusFilter: 'all', setStatusFilter: vi.fn(),
    statusCounts: { all: 0, running: 0, completed: 0, failed: 0, pending: 0 },
    isLoading: false, error: null,
  });
  (useModelsStore as never as Mocked).mockReturnValue({
    models: [{ id: 'm_lfm', name: 'LFM2.5-1.2B-Instruct', status: 'ready',
               architecture_config: { num_hidden_layers: 16, hidden_size: 2048 } }],
    fetchModels: vi.fn(),
  });
  (useDatasetsStore as never as Mocked).mockReturnValue({
    datasets: [
      { id: DS_CODE, name: 'github-code-clean', status: 'ready' },
      { id: DS_OWT, name: 'OpenWebText-2M', status: 'ready' },
    ],
    fetchDatasets: vi.fn(),
  });
  (useTrainingWebSocket as never as Mocked).mockReturnValue({});
  (useDeletionProgressWebSocket as never as Mocked).mockReturnValue({});
  (useWebSocketContext as never as Mocked).mockReturnValue({
    on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: true,
  });

  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ extractions: EXTRACTIONS }),
  }) as never;
  return createTraining;
}

const baseConfig = {
  model_id: 'm_lfm',
  dataset_ids: [DS_CODE],
  architecture_type: SAEArchitectureType.JUMPRELU,
  hidden_dim: 2048,
  latent_dim: 16384,
  training_layers: [11],
  total_steps: 1000,
  warmup_steps: 0,
};

const hookAlert = () => screen.findByTestId('extraction-hook-mismatch');
const startButton = () => screen.getByRole('button', { name: /start training/i });

describe('hook types are validated against the selected extractions', () => {
  beforeEach(() => vi.clearAllMocks());

  it('names the extraction that lacks a requested hook, and what it has', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_CODE_RES], hook_types: ['residual', 'mlp'] });
    render(<TrainingPanel />);

    const alert = await hookAlert();
    expect(alert).toHaveAttribute('role', 'alert');
    expect(alert).toHaveTextContent(/not present in every selected extraction/i);
    expect(alert).toHaveTextContent(/github-code-clean/);
    expect(alert).toHaveTextContent(/has Residual Stream, missing MLP Output/);
  });

  it('blocks Start Training while a selected extraction lacks a requested hook', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_CODE_RES], hook_types: ['residual', 'mlp'] });
    render(<TrainingPanel />);

    await hookAlert();
    expect(startButton()).toBeDisabled();
  });

  it('allows Start Training when the extraction holds every requested hook', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_CODE_RES_MLP], hook_types: ['residual', 'mlp'] });
    render(<TrainingPanel />);

    // The panel has fetched the extractions (their options render), so an absent
    // alert is a verdict and not a race. The dataset has two, hence All.
    await screen.findAllByText(/L11 ·/);
    expect(screen.queryByTestId('extraction-hook-mismatch')).not.toBeInTheDocument();
    expect(startButton()).not.toBeDisabled();
  });

  it('flags only the extraction that lacks the hook when several are selected', async () => {
    setup({
      ...baseConfig,
      dataset_ids: [DS_CODE, DS_OWT],
      extraction_ids: [EXT_CODE_RES_MLP, EXT_OWT_RES],
      hook_types: ['residual', 'mlp'],
    });
    render(<TrainingPanel />);

    const alert = await hookAlert();
    const items = alert.querySelectorAll('li');
    expect(items).toHaveLength(1);
    expect(items[0]).toHaveTextContent(/OpenWebText-2M — selected extraction has Residual Stream, missing MLP Output/);
    expect(alert).not.toHaveTextContent(/github-code-clean/);
    expect(startButton()).toBeDisabled();
  });

  it('does not refuse a residual-only training on a residual extraction', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_CODE_RES] });
    render(<TrainingPanel />);

    await screen.findAllByText(/L11 ·/);
    expect(screen.queryByTestId('extraction-hook-mismatch')).not.toBeInTheDocument();
    expect(startButton()).not.toBeDisabled();
  });

  it("shows the server's reason when it refuses the training, not axios's status line", async () => {
    const detail =
      "Activation extraction ext_code_residual holds hook types ['residual'] and this training " +
      "asks for ['residual', 'attention']: it has no activations for ['attention'].";
    const refusal = Object.assign(new Error('Request failed with status code 422'), {
      response: { status: 422, data: { detail } },
    });
    // On the fly: no extraction is selected, so only the server can refuse.
    const createTraining = setup({ ...baseConfig }, vi.fn().mockRejectedValue(refusal));
    const alertSpy = vi.spyOn(window, 'alert').mockImplementation(() => {});
    render(<TrainingPanel />);

    fireEvent.click(startButton());

    await waitFor(() => expect(alertSpy).toHaveBeenCalledTimes(1));
    expect(createTraining).toHaveBeenCalledTimes(1);
    expect(alertSpy).toHaveBeenCalledWith(`Failed to start training: ${detail}`);
    alertSpy.mockRestore();
  });
});

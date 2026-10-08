/**
 * The training request carries the GPU the user chose.
 *
 * SAE training ran on `torch.device("cuda")` — card 0 — with no way to choose.
 * Since 2026-09-13 card 0 is the 12 GB 3080 Ti, so every run landed on the
 * smaller card. The backend now places a run by the request's `gpu` field
 * ("auto" or a card UUID); this pins that the panel sends it, and sends what
 * was picked rather than a constant.
 *
 * "All GPUs" (multi-GPU Phase 2) is offered only when the run trains ON THE FLY:
 * that path loads the base model, which can be split across cards. A run on
 * cached activations loads no model and the backend refuses `gpu: "all"` for it.
 * So choosing cached activations after "All GPUs" resets the picker to Auto and
 * says why beside it — the choice is not kept and refused at start, and the
 * user is not left guessing why it changed.
 *
 * MUTATION CONTROLS:
 *   * delete `gpu,` from the request in handleStartTraining -> both tests fail
 *   * hard-code `gpu: AUTO_GPU` in the request                -> the UUID test fails
 *
 * "All GPUs" CONTROLS (2026-09-14; each restored byte-identically, sha256):
 *   F18 allowSplit unconditional                     -> "on cached activations: does not offer it",
 *                                                       "choosing cached activations after "All GPUs""
 *   F19 condition flipped                            -> all five tests in the "All GPUs" block
 *   F20 reset effect keeps "all" (no setGpu(AUTO))   -> "choosing cached activations after "All GPUs"
 *                                                       resets the picker to Auto"
 *   F21 reset notice never rendered                  -> "... says why", "picking a card after the reset"
 *   F22 usesCachedActivations([]) === true           -> ""Use Cached Activations" ticked with nothing
 *                                                       chosen still ... offers it" (+ trainingMixture.test)
 *   F23 request drops extraction_ids                 -> "on cached activations: ... names the extraction"
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { TrainingPanel } from './TrainingPanel';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
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

const RTX_3080_TI_UUID = 'GPU-f47ba814-49a2-603f-3595-275284140251';
const RTX_3090_UUID = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';
const TWO_CARDS = {
  gpu_count: 2,
  gpus: [
    { gpu_id: 0, name: 'NVIDIA GeForce RTX 3080 Ti', uuid: RTX_3080_TI_UUID, total_memory_gb: 12 },
    { gpu_id: 1, name: 'NVIDIA GeForce RTX 3090', uuid: RTX_3090_UUID, total_memory_gb: 24 },
  ],
};

const EXT_OWT = 'ext_owt_20260912';
const EXTRACTIONS = [
  {
    extraction_id: EXT_OWT, dataset_id: 'ds_owt', status: 'completed',
    layer_indices: [12], num_samples_processed: 10000,
    created_at: '2026-09-12T10:00:00Z',
    statistics: { layer_12_residual: { shape: [10000, 2048, 2048] } },
  },
];

/** No extraction selected: the run loads the base model and trains on the fly. */
const ON_THE_FLY_CONFIG = {
  model_id: 'm_b5911e07',
  dataset_ids: ['ds_owt'],
  training_layers: [12],
  hook_types: ['residual'],
  architecture_type: SAEArchitectureType.JUMPRELU,
  hidden_dim: 2048,
  latent_dim: 16384,
  learning_rate: 1e-4,
  batch_size: 2048,
  total_steps: 50000,
};

const CACHED_CONFIG = { ...ON_THE_FLY_CONFIG, extraction_ids: [EXT_OWT] };

const createTraining = vi.fn().mockResolvedValue(undefined);

type Mocked = { mockReturnValue: (value: unknown) => void };

function mockTrainings(config: Record<string, unknown>) {
  (useTrainingsStore as never as Mocked).mockReturnValue({
    trainings: [],
    config,
    updateConfig: vi.fn(),
    fetchTrainings: vi.fn(), fetchTraining: vi.fn(),
    createTraining, deleteTraining: vi.fn(),
    statusFilter: 'all', setStatusFilter: vi.fn(),
    statusCounts: { all: 0, running: 0, completed: 0, failed: 0, pending: 0 },
    isLoading: false, error: null,
  });
}

function setup(config: Record<string, unknown> = ON_THE_FLY_CONFIG) {
  mockTrainings(config);
  (useModelsStore as never as Mocked).mockReturnValue({
    models: [{
      id: 'm_b5911e07', name: 'LFM2.5-1.2B-Instruct', status: 'ready',
      architecture_config: { num_hidden_layers: 16, hidden_size: 2048 },
    }],
    fetchModels: vi.fn(),
  });
  (useDatasetsStore as never as Mocked).mockReturnValue({
    datasets: [{ id: 'ds_owt', name: 'OpenWebText-2M', status: 'ready' }],
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
  useSystemMonitorStore.setState({ gpuList: TWO_CARDS } as never);
}

const gpuPicker = () => screen.getByLabelText('GPU') as HTMLSelectElement;
const offeredValues = () => Array.from(gpuPicker().options).map((o) => o.value);

async function startAndCapture() {
  const button = await screen.findByRole('button', { name: /Start Training/i });
  await waitFor(() => expect(button).not.toBeDisabled());
  fireEvent.click(button);
  await waitFor(() => expect(createTraining).toHaveBeenCalledTimes(1));
  return createTraining.mock.calls[0][0];
}

describe('TrainingPanel GPU placement', () => {
  beforeEach(() => {
    createTraining.mockClear();
    setup();
  });

  it('sends gpu: "auto" when no card was chosen', async () => {
    render(<TrainingPanel />);
    const request = await startAndCapture();
    expect(request.gpu).toBe('auto');
    // A top-level field, not a hyperparameter: placement is not part of the recipe.
    expect(request.hyperparameters.gpu).toBeUndefined();
  });

  it('sends the chosen card UUID — the picker is visible without opening Advanced', async () => {
    render(<TrainingPanel />);
    fireEvent.change(gpuPicker(), { target: { value: RTX_3090_UUID } });
    const request = await startAndCapture();
    expect(request.gpu).toBe(RTX_3090_UUID);
  });
});

describe('"All GPUs" follows whether the run loads the base model (multi-GPU Phase 2)', () => {
  beforeEach(() => {
    createTraining.mockClear();
  });

  it('on the fly: offers "All GPUs" and sends gpu "all"', async () => {
    setup(ON_THE_FLY_CONFIG);
    render(<TrainingPanel />);

    expect(offeredValues()).toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID, 'all']);
    fireEvent.change(gpuPicker(), { target: { value: 'all' } });

    const request = await startAndCapture();
    expect(request.gpu).toBe('all');
    expect(request.extraction_ids).toBeUndefined();
  });

  it('"Use Cached Activations" ticked with nothing chosen still trains on the fly, so still offers it', () => {
    // The request sends no extraction_ids until one is chosen, and the worker
    // then loads the model — the picker must agree with the request, not the box.
    setup({ ...ON_THE_FLY_CONFIG, extraction_ids: [] });
    render(<TrainingPanel />);

    expect(offeredValues()).toContain('all');
  });

  it('on cached activations: does not offer it, and the request names the extraction', async () => {
    setup(CACHED_CONFIG);
    render(<TrainingPanel />);

    // The cards are there, so the absence is not an empty list.
    expect(offeredValues()).toEqual(['auto', RTX_3080_TI_UUID, RTX_3090_UUID]);
    const request = await startAndCapture();
    expect(request.extraction_ids).toEqual([EXT_OWT]);
    expect(request.gpu).toBe('auto');
  });

  it('choosing cached activations after "All GPUs" resets the picker to Auto, says why, and sends auto', async () => {
    setup(ON_THE_FLY_CONFIG);
    const { rerender } = render(<TrainingPanel />);
    fireEvent.change(gpuPicker(), { target: { value: 'all' } });
    expect(gpuPicker().value).toBe('all');
    expect(screen.queryByText(/GPU reset to Auto/)).toBeNull();

    // The user selects an extraction: the store's config now carries it.
    mockTrainings(CACHED_CONFIG);
    rerender(<TrainingPanel />);

    await waitFor(() => expect(gpuPicker().value).toBe('auto'));
    expect(screen.getByText(/GPU reset to Auto/)).toBeInTheDocument();
    expect(offeredValues()).not.toContain('all');

    const request = await startAndCapture();
    expect(request.extraction_ids).toEqual([EXT_OWT]);
    expect(request.gpu).toBe('auto');
  });

  it('picking a card after the reset clears the notice', async () => {
    setup(ON_THE_FLY_CONFIG);
    const { rerender } = render(<TrainingPanel />);
    fireEvent.change(gpuPicker(), { target: { value: 'all' } });
    mockTrainings(CACHED_CONFIG);
    rerender(<TrainingPanel />);
    await screen.findByText(/GPU reset to Auto/);

    fireEvent.change(gpuPicker(), { target: { value: RTX_3090_UUID } });

    expect(screen.queryByText(/GPU reset to Auto/)).toBeNull();
    expect(gpuPicker().value).toBe(RTX_3090_UUID);
  });
});

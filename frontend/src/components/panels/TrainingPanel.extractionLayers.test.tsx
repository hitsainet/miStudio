/**
 * You must be able to see which layers an extraction covers, and be stopped
 * before submitting one that cannot serve the training.
 *
 * Reported 2026-08-27, after three identical failures 15 seconds apart:
 *
 *   Extraction ext_m_b55c6926_20260826_140902 is missing layers [44, 46].
 *   Available: [45]. Requested: [44, 46].
 *
 * Two things were wrong. The dropdown rendered "{count}L, {samples} samples",
 * so the OpenWebText-2M extraction at layer 45 and the one at layers 44+46 both
 * read as "…L" and could not be told apart — "I am not being given an option to
 * choose the correct extractions." And `training_layers` is autodiscovered from
 * the FIRST selected extraction while the rest went unchecked, so the mistake
 * was only caught server-side, after a training row had been created.
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

const DS_OWT = 'ds_owt';
const EXT_L45 = 'ext_m_b55c6926_20260826_140902';
const EXT_L44_46 = 'ext_m_b55c6926_20260827_015935';

const EXTRACTIONS = [
  {
    extraction_id: EXT_L45, dataset_id: DS_OWT, status: 'completed',
    layer_indices: [45], num_samples_processed: 10000,
    created_at: '2026-08-26T14:09:02Z',
    statistics: { layer_45_residual: { shape: [10000, 512, 3840] } },
  },
  {
    extraction_id: EXT_L44_46, dataset_id: DS_OWT, status: 'completed',
    layer_indices: [44, 46], num_samples_processed: 10000,
    created_at: '2026-08-27T01:59:35Z',
    statistics: { layer_44_residual: { shape: [10000, 512, 3840] } },
  },
];

function setup(config: Record<string, unknown>) {
  (useTrainingsStore as never as { mockReturnValue: Function }).mockReturnValue({
    trainings: [], config, updateConfig: vi.fn(),
    fetchTrainings: vi.fn(), fetchTraining: vi.fn(),
    createTraining: vi.fn(), deleteTraining: vi.fn(),
    statusFilter: 'all', setStatusFilter: vi.fn(),
    statusCounts: { all: 0, running: 0, completed: 0, failed: 0, pending: 0 },
    isLoading: false, error: null,
  });
  (useModelsStore as never as { mockReturnValue: Function }).mockReturnValue({
    models: [{ id: 'm_b55c6926', name: 'gemma-4-12B-it', status: 'ready',
               architecture_config: { num_hidden_layers: 48, hidden_size: 3840 } }],
    fetchModels: vi.fn(),
  });
  (useDatasetsStore as never as { mockReturnValue: Function }).mockReturnValue({
    datasets: [{ id: DS_OWT, name: 'OpenWebText-2M', status: 'ready' }],
    fetchDatasets: vi.fn(),
  });
  (useTrainingWebSocket as never as { mockReturnValue: Function }).mockReturnValue({});
  (useDeletionProgressWebSocket as never as { mockReturnValue: Function }).mockReturnValue({});
  (useWebSocketContext as never as { mockReturnValue: Function }).mockReturnValue({
    on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(),
    isConnected: true,
  });

  globalThis.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({ extractions: EXTRACTIONS }),
  }) as never;
}

const baseConfig = {
  model_id: 'm_b55c6926',
  dataset_ids: [DS_OWT],
  architecture_type: SAEArchitectureType.JUMPRELU,
  hidden_dim: 3840,
  latent_dim: 30720,
};

describe('extraction layers are visible and validated', () => {
  beforeEach(() => vi.clearAllMocks());

  it('names each extraction by its layers, not just how many', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_L44_46], training_layers: [44, 46] });
    render(<TrainingPanel />);

    // "1L" / "2L" told the user nothing; the layers must be legible.
    expect(await screen.findByText(/L45 ·/)).toBeInTheDocument();
    expect(await screen.findByText(/L44, L46 ·/)).toBeInTheDocument();
  });

  it('flags an extraction that cannot serve the training layers', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_L45], training_layers: [44, 46] });
    render(<TrainingPanel />);

    const alert = await screen.findByRole('alert');
    expect(alert).toHaveTextContent(/not present in every selected extraction/i);
    expect(alert).toHaveTextContent(/OpenWebText-2M/);
    expect(alert).toHaveTextContent(/has L45/);
    expect(alert).toHaveTextContent(/missing L44, L46/);
  });

  it('blocks Start Training while a selected extraction is incompatible', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_L45], training_layers: [44, 46] });
    render(<TrainingPanel />);

    await screen.findByRole('alert');
    const start = screen.getByRole('button', { name: /start training/i });
    expect(start).toBeDisabled();
  });

  it('allows Start Training once a compatible extraction is chosen', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_L44_46], training_layers: [44, 46] });
    render(<TrainingPanel />);

    await waitFor(() =>
      expect(screen.queryByRole('alert')).not.toBeInTheDocument()
    );
    expect(screen.getByRole('button', { name: /start training/i })).not.toBeDisabled();
  });
});

/**
 * The extraction list must not go stale while the panel is open.
 *
 * Reported 2026-08-27: "There were never two extractions for the same model
 * (different layers) in the dropdown. I couldn't select the correct one."
 * The API returned both all along. The panel fetched the list only when
 * `config.model_id` changed, so an extraction that completed while the user was
 * on this panel never appeared — the only cures were navigating away and back,
 * or reloading the browser.
 */
describe('the extraction list stays current', () => {
  beforeEach(() => vi.clearAllMocks());

  it('re-reads when the tab regains focus', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_L44_46], training_layers: [44, 46] });
    render(<TrainingPanel />);

    await screen.findByText(/L44, L46 ·/);
    const before = (globalThis.fetch as never as { mock: { calls: unknown[] } }).mock.calls.length;

    window.dispatchEvent(new Event('focus'));

    await waitFor(() => {
      const after = (globalThis.fetch as never as { mock: { calls: unknown[] } }).mock.calls.length;
      expect(after).toBeGreaterThan(before);
    });
  });

  it('offers a manual refresh when the automatic triggers do not fire', async () => {
    setup({ ...baseConfig, extraction_ids: [EXT_L44_46], training_layers: [44, 46] });
    render(<TrainingPanel />);

    const refresh = await screen.findByRole('button', { name: /refresh/i });
    // The control is disabled while a fetch is in flight, so clicking before the
    // mount fetch settles is a no-op — which made this flake in a full run (2 of
    // 3) while always passing in isolation.
    await waitFor(() => expect(refresh).not.toBeDisabled());
    const before = (globalThis.fetch as never as { mock: { calls: unknown[] } }).mock.calls.length;

    fireEvent.click(refresh);

    await waitFor(() => {
      const after = (globalThis.fetch as never as { mock: { calls: unknown[] } }).mock.calls.length;
      expect(after).toBeGreaterThan(before);
    });
  });

  it('shows an extraction that appears only on the second read', async () => {
    // The stale state the user saw: only the L45 extraction is known yet.
    let secondExtractionExists = false;
    setup({ ...baseConfig, extraction_ids: [], training_layers: [44, 46] });
    globalThis.fetch = vi.fn().mockImplementation(async () => ({
      ok: true,
      json: async () => ({
        extractions: secondExtractionExists ? EXTRACTIONS : [EXTRACTIONS[0]],
      }),
    })) as never;

    render(<TrainingPanel />);
    await screen.findByText(/L45 ·/);
    expect(screen.queryByText(/L44, L46 ·/)).not.toBeInTheDocument();

    // the extraction completes in another panel
    secondExtractionExists = true;
    window.dispatchEvent(new Event('focus'));

    expect(await screen.findByText(/L44, L46 ·/)).toBeInTheDocument();
  });
});

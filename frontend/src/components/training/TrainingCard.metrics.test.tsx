/**
 * The card's FVU history is the run's aggregate series (review R1-C, 2026-09-15).
 *
 * The card fetched the last 20 metric rows and kept the first row of each step.
 * A step's rows include one per SAE and held-out rows, so the history could show one
 * layer's FVU, or a held-out FVU, labelled as the run's. The rows below put a
 * per-layer row FIRST in each step, which is exactly what the old code would keep.
 *
 * NEGATIVE CONTROL (applied alone, this file run, restored, sha256 verified):
 *   UI13 the card deduplicates the raw rows (aggregateMetricRows call removed) -> RED
 *
 * Review R1-A, A5: the rows include a multi-hook run's two SAEs on one layer, and the card
 * asks the endpoint for aggregate rows only, because a raw window is shared by every SAE
 * and held-out row of every hook type.
 *   F1 the card's request without aggregate_only -> RED
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { TrainingCard } from './TrainingCard';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { fetchTrainingMetrics } from '../../api/trainings';
import { SAEArchitectureType, TrainingStatus } from '../../types/training';
import type { Training } from '../../types/training';
import { METRICS_WINDOW_STEPS } from '../../utils/trainingMetricRows';

vi.mock('../../stores/trainingsStore');
vi.mock('../../api/trainings', () => ({ fetchTrainingMetrics: vi.fn() }));
vi.mock('../training/LiveMetrics', () => ({ LiveMetrics: () => <div /> }));
vi.mock('../training/CheckpointManagement', () => ({ CheckpointManagement: () => <div /> }));

const training: Training = {
  id: 'train_series',
  model_id: 'm_model1',
  dataset_id: 'ds_dataset1',
  dataset_ids: ['ds_dataset1'],
  status: TrainingStatus.RUNNING,
  progress: 50,
  current_step: 200,
  total_steps: 400,
  current_loss: 0.1,
  current_l0_sparsity: 0.05,
  current_dead_neurons: 0,
  current_learning_rate: 0.0003,
  hyperparameters: {
    hidden_dim: 2048,
    latent_dim: 8192,
    architecture_type: SAEArchitectureType.JUMPRELU,
    learning_rate: 0.0003,
    batch_size: 4096,
    total_steps: 400,
  } as any,
  created_at: '2026-09-15T10:00:00Z',
  updated_at: '2026-09-15T10:30:00Z',
};

const row = (step: number, layer_idx: number | null, fvu_centred: number, hook_type: string | null = null) => ({
  id: step * 100 + (layer_idx ?? 0),
  training_id: 'train_series',
  step,
  loss: 0.1,
  l0_sparsity: 0.05,
  dead_neurons: 0,
  fvu: fvu_centred - 0.05,
  fvu_centred,
  layer_idx,
  hook_type,
  timestamp: `2026-09-15T10:${String(step / 10).padStart(2, '0')}:00Z`,
});

describe('TrainingCard metrics history', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useTrainingsStore as any).mockReturnValue({
      pauseTraining: vi.fn(), resumeTraining: vi.fn(), stopTraining: vi.fn(), retryTraining: vi.fn(),
      fetchCheckpoints: vi.fn().mockResolvedValue([]), saveCheckpoint: vi.fn(), deleteCheckpoint: vi.fn(),
      stopAndFinalizeTraining: vi.fn(), finalizeTraining: vi.fn(), evaluateTraining: vi.fn(),
      fetchTraining: vi.fn(),
    });
    (fetchTrainingMetrics as any).mockResolvedValue([
      row(100, 11, 0.9, 'residual'), row(100, 11, 0.88, 'mlp'), row(100, -12, 0.8, 'residual'),
      row(100, -12, 0.78, 'mlp'), row(100, null, 0.31),
      row(200, 12, 0.95, 'residual'), row(200, 12, 0.93, 'mlp'), row(200, null, 0.29),
    ]);
  });

  it('draws the aggregate rows, never a layer or held-out row', async () => {
    const { container } = render(
      <TrainingCard training={training} isSelected={false} onToggleSelect={vi.fn()} models={[]} datasets={[]} />
    );
    fireEvent.click(screen.getByRole('button', { name: /Show Live Metrics/ }));

    await waitFor(() => expect(container.textContent).toContain('FVU=0.3100'));
    expect(container.textContent).toContain('FVU=0.2900');
    expect(container.textContent).not.toContain('FVU=0.9000');
    expect(container.textContent).not.toContain('FVU=0.9500');
    expect(container.textContent).not.toContain('FVU=0.8800');
    expect(container.textContent).not.toContain('FVU=0.9300');
    expect(fetchTrainingMetrics).toHaveBeenCalledTimes(1);
    expect(fetchTrainingMetrics).toHaveBeenCalledWith('train_series', { limit: METRICS_WINDOW_STEPS, aggregate_only: true });
  });
});

/**
 * L0 IS A COUNT, and the card must show its TREND.
 *
 * Two defects, one family. `l0_sparsity` is a fraction of d_sae, so 0.05 of
 * 8,192 latents is ~410 active features. Reported as "0.0500" a healthy SAE
 * reads as collapsed — the train_9355afa6 failure (FVU 0.092, 0 dead, ~7 of
 * 30,720 active displayed as "L0: 0.000"), fixed on two surfaces in 2026-08 and
 * still present on two more: the in-card log list and the copied logs.
 *
 * And the window was 20 rows. At log_interval 250 that is 5,000 steps — about
 * fourteen minutes of a 2h45m run — so L0 falling 825 -> 49 over train_b14d263e
 * showed on the card as 115 -> 113. The trend line states first -> latest in
 * counts so the shape is readable without reading the sparkline's pixels.
 *
 * NEGATIVE CONTROLS (each applied alone, this file run, restored, sha256 verified):
 *   UI14 revert the log line to `sparsity.toFixed(4)` -> 'reports L0 as a count' RED
 *   UI15 remove the l0-trend block                    -> 'states the trend' RED
 *   UI16 restore `?? 0` on the fetch path             -> 'a missing L0 is not zero' RED
 */
describe('TrainingCard L0 is reported as a count, with a trend', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    (useTrainingsStore as any).mockReturnValue({
      pauseTraining: vi.fn(), resumeTraining: vi.fn(), stopTraining: vi.fn(), retryTraining: vi.fn(),
      fetchCheckpoints: vi.fn().mockResolvedValue([]), saveCheckpoint: vi.fn(), deleteCheckpoint: vi.fn(),
      stopAndFinalizeTraining: vi.fn(), finalizeTraining: vi.fn(), evaluateTraining: vi.fn(),
      fetchTraining: vi.fn(),
    });
  });

  const open = () => {
    const rendered = render(
      <TrainingCard training={training} isSelected={false} onToggleSelect={vi.fn()} models={[]} datasets={[]} />
    );
    fireEvent.click(screen.getByRole('button', { name: /Show Live Metrics/ }));
    return rendered;
  };

  it('reports L0 as a count of active latents, never the raw fraction', async () => {
    // 0.05 of 8,192 latents = 409.6 -> "~410". "0.0500" is the defect.
    (fetchTrainingMetrics as any).mockResolvedValue([
      row(100, null, 0.31), row(200, null, 0.29),
    ]);
    const { container } = open();

    await waitFor(() => expect(container.textContent).toContain('L0=~410'));
    expect(container.textContent).toContain('(5.0%)');
    expect(container.textContent).not.toContain('L0=0.0500');
  });

  it('states the trend, first to latest, in counts', async () => {
    (fetchTrainingMetrics as any).mockResolvedValue([
      { ...row(100, null, 0.31), l0_sparsity: 0.05 },   // ~410
      { ...row(200, null, 0.29), l0_sparsity: 0.01 },   // ~82
    ]);
    open();

    const trend = await screen.findByTestId('l0-trend');
    expect(trend.textContent).toContain('~410');
    expect(trend.textContent).toContain('~82');
    expect(trend.textContent).toContain('8,192');
  });

  it('a missing L0 is not drawn as zero — that is a collapsed dictionary', async () => {
    (fetchTrainingMetrics as any).mockResolvedValue([
      { ...row(100, null, 0.31), l0_sparsity: null },
      { ...row(200, null, 0.29), l0_sparsity: 0.05 },
    ]);
    const { container } = open();

    await waitFor(() => expect(container.textContent).toContain('L0=~410'));
    // The missing point renders as an em dash, never "NaN" and never a zero
    // count that would drag the chart's minimum to the floor.
    expect(container.textContent).toContain('L0=—');
    expect(container.textContent).not.toContain('NaN');
    expect(container.textContent).not.toContain('L0=0,');
  });

  it('asks for the whole run, not a 20-row keyhole', async () => {
    (fetchTrainingMetrics as any).mockResolvedValue([row(100, null, 0.31)]);
    open();

    await waitFor(() => expect(fetchTrainingMetrics).toHaveBeenCalled());
    expect(METRICS_WINDOW_STEPS).toBeGreaterThanOrEqual(240); // a 60k-step run at log_interval 250
    expect(fetchTrainingMetrics).toHaveBeenCalledWith('train_series', {
      limit: METRICS_WINDOW_STEPS,
      aggregate_only: true,
    });
  });
});

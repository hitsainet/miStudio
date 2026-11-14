/**
 * TrainingEvaluationPanel (SAE training remediation, item 6).
 *
 * The headline is loss recovered against MEAN ablation; zero ablation is shown
 * for reference with a note that it is a uniform-output floor. A failure or skip
 * shows its reason — a panel that went blank on failure would be the "quietly
 * dark" outcome this evaluation exists to end.
 *
 * MUTATION CONTROL (2026-09-15; applied alone, source restored and verified by sha256):
 *   UI8 the headline cell shows loss_recovered_vs_zero   -> RED, "shows loss recovered against mean ablation as the headline"
 * (The panel's wiring into the card is controlled by UI4/UI5 — see utils/fvu.test.ts.)
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { TrainingEvaluationPanel } from './TrainingEvaluationPanel';
import { TrainingStatus } from '../../types/training';
import type { Training, TrainingEvaluation } from '../../types/training';

const base: Training = {
  id: 'train_6247e768',
  model_id: 'm_x',
  dataset_id: 'ds_x',
  status: TrainingStatus.COMPLETED,
  progress: 100,
  current_step: 10,
  total_steps: 10,
  hyperparameters: { hidden_dim: 2048, latent_dim: 8192 } as any,
  created_at: '2026-09-15T00:00:00Z',
  updated_at: '2026-09-15T00:00:00Z',
} as Training;

const completed: TrainingEvaluation = {
  status: 'completed',
  trigger: 'rerun',
  ce_base: 2.1,
  tokens: 131072,
  sources: [{ label: 'ext_a' }, { label: 'ext_b' }],
  layers: [
    {
      layer: 11, ce_spliced: 2.4, ce_mean_ablated: 3.5, ce_zero_ablated: 11.09, ce_delta: 0.3,
      loss_recovered_vs_mean: 0.7857, loss_recovered_vs_zero: 0.9666, kl: 0.28, l0: 65.2, fvu_centred: 0.32,
    },
    {
      layer: 12, ce_spliced: 2.45, ce_mean_ablated: 3.4, ce_zero_ablated: 11.09, ce_delta: 0.35,
      loss_recovered_vs_mean: 0.73, loss_recovered_vs_zero: 0.96, kl: 0.31, l0: 60.0, fvu_centred: 0.3,
    },
  ],
  all_layers_spliced: { layers: [11, 12], ce: 3.55, ce_delta: 1.45, kl: 1.2 },
  zero_ablation_note: 'Zero ablation is a floor, not a baseline',
};

describe('TrainingEvaluationPanel', () => {
  beforeEach(() => vi.useRealTimers());
  afterEach(() => vi.useRealTimers());

  // THE STAGE THE RECORD HAS CARRIED ALL ALONG. `evaluation.progress` was declared
  // in the wire type and written by the backend, and nothing read it — so a running
  // evaluation said "running…" and no more, however long it took. Loading the base
  // model is the slow part (minutes on a large model), which is exactly the moment
  // an operator needs to tell "working" from "stuck".
  //
  // Mutation controls (each applied alone, source restored and verified by sha256):
  //   UI9  the stage span removed          -> RED, "names the stage a running evaluation is in"
  //   UI10 batch counts shown with only batches_done -> RED, "shows batch counts only when both numbers are present"
  it('names the stage a running evaluation is in', () => {
    const evaluation: TrainingEvaluation = {
      status: 'running',
      progress: { stage: 'loading_model' },
      updated_at: new Date().toISOString(),
    };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} />);

    expect(screen.getByTestId('evaluation-stage')).toHaveTextContent('loading the base model');
  });

  it('shows batch counts only when both numbers are present', () => {
    const withBoth: TrainingEvaluation = {
      status: 'running',
      progress: { stage: 'cross_entropy', batches_done: 12, batches: 32 },
      updated_at: new Date().toISOString(),
    };
    const { unmount } = render(<TrainingEvaluationPanel training={{ ...base, evaluation: withBoth }} />);
    expect(screen.getByTestId('evaluation-batches')).toHaveTextContent('12 of 32 batches');
    unmount();

    // A count with no denominator would read as a total, so it is not shown at all.
    const doneOnly: TrainingEvaluation = {
      status: 'running',
      progress: { stage: 'cross_entropy', batches_done: 12 },
      updated_at: new Date().toISOString(),
    };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation: doneOnly }} />);
    expect(screen.queryByTestId('evaluation-batches')).not.toBeInTheDocument();
    expect(screen.getByTestId('evaluation-stage')).toHaveTextContent('measuring cross-entropy');
  });

  it('falls back to the raw name for a stage it does not know', () => {
    // `stage` is typed `(string & {})` so the backend can add one; an unknown stage
    // must still appear rather than vanish because this map is behind.
    const evaluation: TrainingEvaluation = {
      status: 'running',
      progress: { stage: 'some_new_stage' },
      updated_at: new Date().toISOString(),
    };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} />);

    expect(screen.getByTestId('evaluation-stage')).toHaveTextContent('some_new_stage');
  });

  it('offers to evaluate a training that has never been evaluated', () => {
    const onEvaluate = vi.fn().mockResolvedValue(undefined);
    render(<TrainingEvaluationPanel training={base} onEvaluate={onEvaluate} />);

    expect(screen.getByText(/Not evaluated/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Evaluate/ }));
    expect(onEvaluate).toHaveBeenCalledTimes(1);
    expect(onEvaluate).toHaveBeenCalledWith('train_6247e768');
  });

  it('shows loss recovered against mean ablation as the headline, per layer', () => {
    render(<TrainingEvaluationPanel training={{ ...base, evaluation: completed }} />);

    const row = screen.getByTestId('evaluation-layer-11');
    expect(within(row).getByText('78.6%')).toBeInTheDocument(); // vs mean
    expect(within(row).getByText('96.7%')).toBeInTheDocument(); // vs zero, for reference
    expect(within(row).getByText('(+0.300)')).toBeInTheDocument();
    expect(within(row).getByText('0.320')).toBeInTheDocument(); // FVU
    expect(screen.getByText('Recovered vs mean')).toBeInTheDocument();
  });

  it('says zero ablation is a floor', () => {
    render(<TrainingEvaluationPanel training={{ ...base, evaluation: completed }} />);
    expect(screen.getByTestId('evaluation-zero-note')).toHaveTextContent(/uniform-output floor/);
  });

  it('shows the all-layers-spliced cost when more than one layer was evaluated', () => {
    render(<TrainingEvaluationPanel training={{ ...base, evaluation: completed }} />);
    const row = screen.getByTestId('evaluation-all-layers');
    expect(within(row).getByText('(+1.450)')).toBeInTheDocument();
  });

  it('shows why an evaluation failed', () => {
    const evaluation: TrainingEvaluation = { status: 'failed', reason: 'RuntimeError: CUDA out of memory' };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} onEvaluate={vi.fn()} />);
    expect(screen.getByText('RuntimeError: CUDA out of memory')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Re-evaluate/ })).not.toBeDisabled();
  });

  it('shows why an evaluation was skipped', () => {
    const evaluation: TrainingEvaluation = { status: 'skipped', reason: 'no block the training never read' };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} />);
    expect(screen.getByText('no block the training never read')).toBeInTheDocument();
  });

  it('disables re-running while one is pending, and refreshes until it finishes', async () => {
    vi.useFakeTimers();
    const onRefresh = vi.fn().mockResolvedValue(undefined);
    const evaluation: TrainingEvaluation = { status: 'pending', task_id: 'abc' };
    render(
      <TrainingEvaluationPanel
        training={{ ...base, evaluation }}
        onEvaluate={vi.fn()}
        onRefresh={onRefresh}
        pollMs={1000}
      />
    );

    expect(screen.getByRole('button', { name: /Re-evaluate/ })).toBeDisabled();
    await act(async () => {
      vi.advanceTimersByTime(2500);
    });
    expect(onRefresh).toHaveBeenCalledTimes(2);
    expect(onRefresh).toHaveBeenCalledWith('train_6247e768', true);
  });

  it('does not poll once the evaluation has finished', async () => {
    vi.useFakeTimers();
    const onRefresh = vi.fn().mockResolvedValue(undefined);
    render(<TrainingEvaluationPanel training={{ ...base, evaluation: completed }} onRefresh={onRefresh} pollMs={1000} />);
    await act(async () => {
      vi.advanceTimersByTime(5000);
    });
    expect(onRefresh).not.toHaveBeenCalled();
  });

  /*
   * REVIEW R1-C (2026-09-15). A pending or running record whose worker died stayed
   * that way: the button was disabled forever, the panel polled forever, and the
   * endpoint's 409 named `force=true`, which nothing in the UI could send.
   *
   * NEGATIVE CONTROLS (each applied alone, this file run, restored, sha256 verified):
   *   UI9  `stale` forced to false                          -> RED  offers a forced re-run...
   *   UI10 the force button calls handleEvaluate() (no force) -> RED  offers a forced re-run...
   *   UI11 the plain button is `onClick={handleEvaluate}`   -> RED  a plain Evaluate click never forces
   */
  it('offers a forced re-run once a running record has gone quiet, and sends force', async () => {
    const now = Date.parse('2026-09-15T12:00:00Z');
    vi.useFakeTimers();
    vi.setSystemTime(now);
    const onEvaluate = vi.fn().mockResolvedValue(undefined);
    const evaluation: TrainingEvaluation = {
      status: 'running', task_id: 'abc', updated_at: '2026-09-15T11:30:00Z',
    };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} onEvaluate={onEvaluate} />);

    expect(screen.getByRole('button', { name: /Re-evaluate/ })).toBeDisabled();
    expect(screen.getByTestId('evaluation-stale')).toHaveTextContent('30 min');
    await act(async () => {
      fireEvent.click(screen.getByTestId('evaluation-force'));
    });
    expect(onEvaluate).toHaveBeenCalledTimes(1);
    expect(onEvaluate).toHaveBeenCalledWith('train_6247e768', { force: true });
  });

  it('does not offer to force a record that is still being written', () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.parse('2026-09-15T12:00:00Z'));
    const evaluation: TrainingEvaluation = {
      status: 'running', task_id: 'abc', updated_at: '2026-09-15T11:59:00Z',
    };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} onEvaluate={vi.fn()} />);
    expect(screen.queryByTestId('evaluation-force')).toBeNull();
    expect(screen.queryByTestId('evaluation-stale')).toBeNull();
  });

  it('judges a pending request by when it was requested', () => {
    vi.useFakeTimers();
    vi.setSystemTime(Date.parse('2026-09-15T12:00:00Z'));
    const evaluation: TrainingEvaluation = {
      status: 'pending', task_id: 'abc', requested_at: '2026-09-15T09:00:00Z',
    };
    render(<TrainingEvaluationPanel training={{ ...base, evaluation }} onEvaluate={vi.fn()} />);
    expect(screen.getByTestId('evaluation-force')).toBeInTheDocument();
  });

  it('a plain Evaluate click never forces', async () => {
    const onEvaluate = vi.fn().mockResolvedValue(undefined);
    render(<TrainingEvaluationPanel training={{ ...base, evaluation: completed }} onEvaluate={onEvaluate} />);
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Re-evaluate/ }));
    });
    expect(onEvaluate).toHaveBeenCalledTimes(1);
    expect(onEvaluate).toHaveBeenCalledWith('train_6247e768');
  });
});

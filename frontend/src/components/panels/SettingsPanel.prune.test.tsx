/**
 * MIS-E2E-128 — the prune confirmation must describe what will actually happen.
 *
 * `preview` is a snapshot taken when Preview was clicked. The Celery task
 * re-reads `checkpoint_prune_dry_run` from settings at execution time. So:
 *
 *   1. Preview while dry-run is ON   -> snapshot says dry_run: true
 *   2. Untick "dry run" and Save     -> live policy is now destructive
 *   3. Prune now                     -> dialog said "This will report on 12
 *                                       checkpoint file(s)", the toast said
 *                                       "Dry-run prune queued", and the task
 *                                       permanently deleted all twelve.
 *
 * A destructive, irreversible action behind a dialog stating the opposite of
 * what it will do — and the dialog is the only confirmation step there is.
 *
 * MUTATION CONTROLS (each must turn one of these red):
 *   * confirm against `preview` instead of the re-fetched policy
 *   * drop the re-fetch entirely
 *   * default `dry_run` to true when the refresh fails (must fail CLOSED)
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { CheckpointPrunePreviewPanel } from './SettingsPanel';
import { useTrainingsStore } from '../../stores/trainingsStore';
import type { CheckpointPrunePreview } from '../../types/training';

vi.mock('../../stores/trainingsStore');

const previewCheckpointPrune = vi.fn();
const pruneCheckpoints = vi.fn();

function policy(dry_run: boolean): CheckpointPrunePreview {
  // The real shape, from types/training.ts. A fixture that agrees with itself
  // rather than with production is this audit's most common reason a suite
  // stays green over a live defect.
  return {
    training_id: 'train_969e90af',
    policy: { enabled: true, dry_run, keep_last: 3, keep_best: true, min_age_hours: 0 },
    prunable_steps: [1000, 2000, 3000],
    kept_steps: [10300],
    checkpoint_count: 12,
    estimated_bytes: 78_000_000_000,
    skipped_reason: null,
  };
}

beforeEach(() => {
  vi.clearAllMocks();
  (useTrainingsStore as any).mockReturnValue({
    trainings: [{ id: 'train_969e90af', status: 'completed' }],
    fetchTrainings: vi.fn().mockResolvedValue(undefined),
    previewCheckpointPrune,
    pruneCheckpoints,
  });
  pruneCheckpoints.mockResolvedValue(undefined);
});

async function previewThenPrune(user: ReturnType<typeof userEvent.setup>) {
  await user.selectOptions(screen.getByRole('combobox'), 'train_969e90af');
  await user.click(screen.getByRole('button', { name: /preview/i }));
  await screen.findByRole('button', { name: /prune now/i });
  await user.click(screen.getByRole('button', { name: /prune now/i }));
}

describe('checkpoint prune confirmation', () => {
  it('warns about PERMANENT DELETION when the policy changed since the preview', async () => {
    // The exact reported sequence: snapshot dry-run, live policy destructive.
    previewCheckpointPrune
      .mockResolvedValueOnce(policy(true))
      .mockResolvedValueOnce(policy(false));
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();

    render(<CheckpointPrunePreviewPanel />);
    await previewThenPrune(user);

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    const message = confirmSpy.mock.calls[0][0] as string;
    expect(message).toContain('PERMANENTLY DELETE');
    expect(message).not.toContain('report on');
  });

  it('re-fetches the policy before confirming, not after', async () => {
    previewCheckpointPrune
      .mockResolvedValueOnce(policy(true))
      .mockResolvedValueOnce(policy(false));
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();

    render(<CheckpointPrunePreviewPanel />);
    await previewThenPrune(user);

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    // Two preview calls: the explicit one, then the confirmation refresh.
    expect(previewCheckpointPrune).toHaveBeenCalledTimes(2);
  });

  it('still says "report on" when the policy really is dry-run', async () => {
    // Without this the fix could be "always say PERMANENTLY DELETE", which
    // passes the first test and trains the user to ignore the dialog.
    previewCheckpointPrune.mockResolvedValue(policy(true));
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();

    render(<CheckpointPrunePreviewPanel />);
    await previewThenPrune(user);

    await waitFor(() => expect(confirmSpy).toHaveBeenCalled());
    expect(confirmSpy.mock.calls[0][0] as string).toContain('report on');
  });

  it('does not prune when the user declines', async () => {
    previewCheckpointPrune.mockResolvedValue(policy(false));
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    const user = userEvent.setup();

    render(<CheckpointPrunePreviewPanel />);
    await previewThenPrune(user);

    await waitFor(() => expect(previewCheckpointPrune).toHaveBeenCalledTimes(2));
    expect(pruneCheckpoints).not.toHaveBeenCalled();
  });

  it('prunes nothing when the policy refresh fails', async () => {
    // Fail CLOSED. The old code would have confirmed against the stale
    // snapshot; proceeding on an unknown policy is how `dry_run` defaults to
    // its deleting value.
    previewCheckpointPrune
      .mockResolvedValueOnce(policy(true))
      .mockRejectedValueOnce(new Error('network'));
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const user = userEvent.setup();

    render(<CheckpointPrunePreviewPanel />);
    await previewThenPrune(user);

    await waitFor(() => expect(previewCheckpointPrune).toHaveBeenCalledTimes(2));
    expect(confirmSpy).not.toHaveBeenCalled();
    expect(pruneCheckpoints).not.toHaveBeenCalled();
    expect(
      await screen.findByText(/could not confirm the current prune policy/i)
    ).toBeInTheDocument();
  });
});

/**
 * "Run cleanup now" must sweep every training, and must never promise a report
 * and then delete.
 *
 * Before this the sweep was reachable only from the daily scheduler, which
 * no-ops while `checkpoint_prune_enabled` is false — the shipped default. 84 GB
 * of prunable checkpoints accumulated with no UI path to reclaim them except
 * previewing and pruning one training at a time (2026-08-28).
 *
 * The confirmation follows MIS-E2E-128: the Celery task re-reads
 * `checkpoint_prune_dry_run` when it EXECUTES, so the local form state — which
 * may be edited and unsaved — cannot describe what will happen. The dialog is
 * built from the policy re-read from the server, and a failed re-read must stop
 * the sweep rather than assume dry-run.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { StorageTab } from './SettingsPanel';
import { useSettingsStore } from '../../stores/settingsStore';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { fetchAPI } from '../../api/client';

vi.mock('../../stores/settingsStore');
vi.mock('../../stores/trainingsStore');
vi.mock('../../api/client');

const PREVIEW = (dry_run: boolean) => ({
  data: {
    policy: { enabled: true, dry_run, keep_last: 2, keep_best: true, min_age_hours: 24 },
    trainings_scanned: 20,
    trainings_affected: 19,
    total_checkpoints: 394,
    estimated_bytes: 84_415_594_600,
    skipped: {},
    per_training: [],
  },
});

function setup(settingRows: Array<{ key: string; value: string }>) {
  (useSettingsStore as never as { mockReturnValue: Function }).mockReturnValue({
    settings: settingRows,
    upsert: vi.fn().mockResolvedValue(undefined),
  });
  (useTrainingsStore as never as { mockReturnValue: Function }).mockReturnValue({
    trainings: [],
    fetchTrainings: vi.fn().mockResolvedValue(undefined),
    previewCheckpointPrune: vi.fn(),
    pruneCheckpoints: vi.fn(),
  });
}

describe('Run cleanup now (all trainings)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'confirm').mockReturnValue(true);
  });
  afterEach(() => vi.restoreAllMocks());

  it('previews the whole sweep, sized', async () => {
    setup([{ key: 'checkpoint_prune_dry_run', value: 'false' }]);
    vi.mocked(fetchAPI).mockResolvedValue(PREVIEW(false) as never);

    render(<StorageTab />);
    await userEvent.click(screen.getByRole('button', { name: /preview all trainings/i }));

    const status = await screen.findByRole('status');
    expect(status).toHaveTextContent(/394/);
    expect(status).toHaveTextContent(/19/);
    expect(status).toHaveTextContent(/84\.4 GB/);
  });

  it('sweeps every training through one request', async () => {
    setup([{ key: 'checkpoint_prune_dry_run', value: 'false' }]);
    vi.mocked(fetchAPI).mockResolvedValue(PREVIEW(false) as never);

    render(<StorageTab />);
    await userEvent.click(screen.getByRole('button', { name: /run cleanup now/i }));

    await waitFor(() =>
      expect(fetchAPI).toHaveBeenCalledWith(
        '/api/v1/trainings/checkpoints/prune-all',
        { method: 'POST' },
      )
    );
  });

  it('warns of PERMANENT deletion when the LIVE policy is destructive', async () => {
    // Local form says dry-run; the server says otherwise. The dialog must
    // follow the server, or it promises a report and deletes.
    setup([{ key: 'checkpoint_prune_dry_run', value: 'true' }]);
    vi.mocked(fetchAPI).mockResolvedValue(PREVIEW(false) as never);

    render(<StorageTab />);
    await userEvent.click(screen.getByRole('button', { name: /run cleanup now/i }));

    await waitFor(() => expect(window.confirm).toHaveBeenCalled());
    const asked = vi.mocked(window.confirm).mock.calls[0][0] as string;
    expect(asked).toMatch(/PERMANENTLY DELETE/);
    expect(asked).not.toMatch(/without deleting/i);
  });

  it('says report-only when the live policy is dry-run', async () => {
    setup([{ key: 'checkpoint_prune_dry_run', value: 'false' }]);
    vi.mocked(fetchAPI).mockResolvedValue(PREVIEW(true) as never);

    render(<StorageTab />);
    await userEvent.click(screen.getByRole('button', { name: /run cleanup now/i }));

    await waitFor(() => expect(window.confirm).toHaveBeenCalled());
    const asked = vi.mocked(window.confirm).mock.calls[0][0] as string;
    expect(asked).toMatch(/without deleting/i);
    expect(asked).not.toMatch(/PERMANENTLY DELETE/);
  });

  it('does not sweep when the operator declines', async () => {
    setup([{ key: 'checkpoint_prune_dry_run', value: 'false' }]);
    vi.mocked(fetchAPI).mockResolvedValue(PREVIEW(false) as never);
    vi.mocked(window.confirm).mockReturnValue(false);

    render(<StorageTab />);
    await userEvent.click(screen.getByRole('button', { name: /run cleanup now/i }));

    await waitFor(() => expect(window.confirm).toHaveBeenCalled());
    expect(fetchAPI).not.toHaveBeenCalledWith(
      '/api/v1/trainings/checkpoints/prune-all',
      expect.anything(),
    );
  });

  it('fails CLOSED when the live policy cannot be read', async () => {
    setup([{ key: 'checkpoint_prune_dry_run', value: 'true' }]);
    vi.mocked(fetchAPI).mockRejectedValue(new Error('network down'));

    render(<StorageTab />);
    await userEvent.click(screen.getByRole('button', { name: /run cleanup now/i }));

    expect(await screen.findByRole('alert')).toBeInTheDocument();
    expect(fetchAPI).not.toHaveBeenCalledWith(
      '/api/v1/trainings/checkpoints/prune-all',
      expect.anything(),
    );
  });
});

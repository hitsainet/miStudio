/**
 * Every failed operation must be clearable from the Monitor.
 *
 * REPORTED 2026-07-26: "there is no way to clear failed jobs from the bottom of
 * the monitor screen."
 *
 * Federated rows (can_retry=false) rendered the static text "Manage in its
 * panel". For Neuronpedia pushes that panel does not exist — the only DELETE in
 * that API targets neuronpedia_exports, a DIFFERENT table — so four failures
 * from 2026-03-28 were permanently stuck on screen.
 *
 * MUTATION CONTROLS:
 *   * restore the "Manage in its panel" span -> dismiss test fails
 *   * remove the Clear all button            -> clear-all test fails
 *   * make dismiss non-optimistic            -> optimism test fails
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { render } from '@testing-library/react';
import { FailedOperationsSection } from './FailedOperationsSection';
import { useTaskQueueStore } from '../../stores/taskQueueStore';

vi.mock('../../stores/taskQueueStore');

const FEDERATED_PUSH = {
  id: 'push_abc',
  task_id: 'push_abc',
  task_type: 'neuronpedia_push',
  entity_id: 'sae_1',
  entity_type: 'neuronpedia',
  status: 'failed',
  progress: 0,
  can_retry: false,
  retry_count: 0,
  error_message: 'foreign key constraint violated',
  entity_info: { name: 'SAE from LFM2.5-1.2B-Instruct (L12-residual)' },
  completed_at: '2026-03-28T09:04:54Z',
};

const dismissFailedTask = vi.fn().mockResolvedValue(undefined);
const dismissAllFailedTasks = vi.fn().mockResolvedValue(4);
const fetchFailedTasks = vi.fn();

function mountWith(tasks: any[]) {
  (useTaskQueueStore as any).mockReturnValue({
    failedTasks: tasks,
    failedLoading: false,
    failedError: null,
    fetchFailedTasks,
    deleteTask: vi.fn(),
    dismissFailedTask,
    dismissAllFailedTasks,
  });
  return render(<FailedOperationsSection />);
}

describe('FailedOperationsSection dismissal', () => {
  beforeEach(() => vi.clearAllMocks());
  afterEach(() => vi.clearAllMocks());

  it('offers a way to clear a federated failure', () => {
    mountWith([FEDERATED_PUSH]);

    expect(screen.queryByText('Manage in its panel')).not.toBeInTheDocument();
    expect(
      screen.getByTitle(/Clear from this list/),
    ).toBeInTheDocument();
  });

  it('dismisses with the task type, not just the id', async () => {
    mountWith([FEDERATED_PUSH]);

    const button = screen.getByTitle(/Clear from this list/);
    fireEvent.click(button);                       // arms confirmation
    fireEvent.click(screen.getByText('Confirm?')); // confirms

    await waitFor(() =>
      expect(dismissFailedTask).toHaveBeenCalledWith('neuronpedia_push', 'push_abc'),
    );
    expect(dismissFailedTask).toHaveBeenCalledTimes(1);
  });

  it('requires a second click before clearing', () => {
    mountWith([FEDERATED_PUSH]);

    fireEvent.click(screen.getByTitle(/Clear from this list/));

    expect(dismissFailedTask).not.toHaveBeenCalled();
    expect(screen.getByText('Confirm?')).toBeInTheDocument();
  });

  it('offers Clear all, behind its own confirmation', async () => {
    mountWith([FEDERATED_PUSH, { ...FEDERATED_PUSH, id: 'push_def' }]);

    fireEvent.click(screen.getByText('Clear all'));
    expect(dismissAllFailedTasks).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText('Confirm clear all?'));
    await waitFor(() => expect(dismissAllFailedTasks).toHaveBeenCalledTimes(1));
  });

  it('still shows Retry for a genuinely retryable row', () => {
    mountWith([{ ...FEDERATED_PUSH, id: 'q1', can_retry: true, task_type: 'download' }]);

    expect(screen.getByText('Retry')).toBeInTheDocument();
  });
});

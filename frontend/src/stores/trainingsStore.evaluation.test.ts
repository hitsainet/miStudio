/**
 * trainingsStore.evaluateTraining (SAE training remediation, item 6).
 *
 * The Evaluate button reaches the API only through this action, so the URL, the
 * query and the call count are the contract: a store posting to the wrong route
 * would leave every card saying "pending" while nothing was queued.
 *
 * MUTATION CONTROL (2026-09-15; applied alone, source restored and verified by sha256):
 *   UI7 the store posts to /evaluation instead of /evaluate -> RED, "POSTs the evaluate route once..." and
 *       "sends the token budget and force when asked"
 */

import { beforeEach, describe, expect, it, vi } from 'vitest';
import axios from 'axios';
import { useTrainingsStore } from './trainingsStore';
import { TrainingStatus } from '../types/training';
import type { Training } from '../types/training';

vi.mock('axios');
const mockedAxios = axios as any;

const completed = {
  id: 'train_123',
  model_id: 'm_x',
  dataset_id: 'ds_x',
  status: TrainingStatus.COMPLETED,
  progress: 100,
  current_step: 10,
  total_steps: 10,
  hyperparameters: {} as any,
  created_at: '2026-09-15T00:00:00Z',
  updated_at: '2026-09-15T00:00:00Z',
} as Training;

describe('trainingsStore.evaluateTraining', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useTrainingsStore.setState({ trainings: [completed], error: null });
  });

  it('POSTs the evaluate route once, with no query by default', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { data: { task_id: 'task-1', status: 'pending' } } });

    await useTrainingsStore.getState().evaluateTraining('train_123');

    expect(mockedAxios.post).toHaveBeenCalledTimes(1);
    expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings/train_123/evaluate');
  });

  it('sends the token budget and force when asked', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { data: { task_id: 'task-2' } } });

    await useTrainingsStore.getState().evaluateTraining('train_123', { tokenBudget: 65536, force: true });

    expect(mockedAxios.post).toHaveBeenCalledWith(
      '/api/v1/trainings/train_123/evaluate?token_budget=65536&force=true'
    );
  });

  it('marks the training pending with the queued task id', async () => {
    mockedAxios.post.mockResolvedValueOnce({ data: { data: { task_id: 'task-3' } } });

    await useTrainingsStore.getState().evaluateTraining('train_123');

    const training = useTrainingsStore.getState().trainings.find((t) => t.id === 'train_123');
    expect(training?.evaluation).toMatchObject({ status: 'pending', task_id: 'task-3' });
  });

  it('surfaces the API refusal and rethrows', async () => {
    mockedAxios.post.mockRejectedValueOnce({ response: { data: { detail: 'An evaluation is already running' } } });

    await expect(useTrainingsStore.getState().evaluateTraining('train_123')).rejects.toBeTruthy();
    expect(useTrainingsStore.getState().error).toBe('An evaluation is already running');
  });
});

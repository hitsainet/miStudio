/**
 * The compact banner must name the model and show real counts.
 *
 * `entity_info.name` was ALREADY in the /active response — resolved by
 * TaskQueueService.get_entity_info because J-lens rows carry a real model id
 * with entity_type="model" — and this component simply never read it. With two
 * fits queued, "52% · running" cannot say which model is moving.
 *
 * MUTATION CONTROLS:
 *   * stop reading entity_info.name    -> "names the model" fails
 *   * render absent counts as 0/0      -> "absent, never zero" fails
 *   * drop the percentage from counts  -> "keeps the percentage" fails
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { counts, RunningWork } from './RunningWork';

vi.mock('../../api/taskQueue', () => ({ getActiveTasks: vi.fn() }));
import { getActiveTasks } from '../../api/taskQueue';

describe('counts', () => {
  it('renders the percentage AND the raw counts', () => {
    expect(
      counts({
        progress: 52.833,
        entity_info: { prompts_seen: 634, total_prompts: 1200 },
      }),
    ).toBe('52.8% · 634/1,200');
  });

  it('returns null when the counts are absent, never a zeroed placeholder', () => {
    /**
     * A fit that has not reported yet is UNKNOWN, not "0 of 1200". Rendering a
     * zero would claim the fit had done nothing, and the caller falls back to
     * the status instead — which is the honest thing to show.
     *
     * MUTATION CONTROL: default seen/total to 0 and this fails.
     */
    expect(counts({ progress: 52.8, entity_info: null })).toBeNull();
    expect(counts({ progress: 52.8, entity_info: { prompts_seen: 634 } })).toBeNull();
    expect(
      counts({ progress: 52.8, entity_info: { prompts_seen: 0, total_prompts: 0 } }),
    ).toBeNull();
  });

  it('still reports counts when the percentage is unknown', () => {
    expect(
      counts({ progress: null, entity_info: { prompts_seen: 12, total_prompts: 400 } }),
    ).toBe('12/400');
  });
});


describe('RunningWork rendering', () => {
  beforeEach(() => vi.clearAllMocks());

  it('names the model it is fitting', async () => {
    // MUTATION CONTROL: stop reading entity_info.name and this fails.
    (getActiveTasks as any).mockResolvedValue({ data: [
      {
        id: 'tq_1',
        task_id: 'c1',
        task_type: 'jlens_fit',
        entity_id: 'm_98a4b47e',
        entity_type: 'model',
        status: 'running',
        progress: 52.8,
        error_message: null,
        retry_params: null,
        retry_count: 0,
        can_retry: false,
        created_at: null,
        started_at: null,
        completed_at: null,
        updated_at: null,
        entity_info: {
          name: 'gemma-2-2b-it',
          prompts_seen: 634,
          total_prompts: 1200,
        },
      },
    ] });

    render(<RunningWork modelId="m_98a4b47e" />);

    await waitFor(() => {
      expect(screen.getByText('gemma-2-2b-it')).toBeInTheDocument();
    });
    expect(screen.getByText('52.8% · 634/1,200')).toBeInTheDocument();
  });
});

/**
 * Work in flight must be visible on the page that started it.
 *
 * A 45-minute fit burned the GPU with nothing in the J-Lens panel saying so.
 * The fit card only knew about a fit THIS tab had submitted — its polling is
 * component state — so a fit queued from the API, from MCP, from a second tab,
 * or before a refresh was invisible.
 *
 * MUTATION CONTROLS:
 *   * drop the jlens_ filter        -> "only J-space work" fails
 *   * drop the model filter         -> "only this model" fails
 *   * render null progress as 100%  -> "unreported progress" fails
 *   * clear the list on a poll error-> "a failed poll" fails
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { RunningWork, isJSpaceWork, summarise } from './RunningWork';
import { TaskType } from '../../types/taskQueue';

vi.mock('../../api/taskQueue', () => ({ getActiveTasks: vi.fn() }));
import { getActiveTasks } from '../../api/taskQueue';

const row = (over: Partial<any> = {}) => ({
  id: 'tq_1',
  task_id: 't1',
  task_type: TaskType.JLENS_FIT,
  entity_id: 'm_gemma',
  entity_type: 'model',
  status: 'running',
  progress: 42,
  error_message: null,
  retry_params: null,
  retry_count: 0,
  can_retry: false,
  created_at: null,
  started_at: null,
  completed_at: null,
  updated_at: null,
  entity_info: null,
  ...over,
});

beforeEach(() => vi.clearAllMocks());

describe('isJSpaceWork', () => {
  it('matches only jlens_ task types', () => {
    expect(isJSpaceWork(row())).toBe(true);
    expect(isJSpaceWork(row({ task_type: TaskType.TRAINING }))).toBe(false);
    expect(isJSpaceWork(row({ task_type: TaskType.EXTRACTION }))).toBe(false);
  });
});

describe('RunningWork', () => {
  it('shows a fit that this tab did not start', async () => {
    vi.mocked(getActiveTasks).mockResolvedValue({ data: [row()] } as any);
    render(<RunningWork modelId="m_gemma" />);

    await waitFor(() =>
      expect(screen.getByText('Fitting a J-lens')).toBeInTheDocument()
    );
    expect(screen.getByText(/42% · running/)).toBeInTheDocument();
  });

  it('shows only J-space work, not every job on the box', async () => {
    vi.mocked(getActiveTasks).mockResolvedValue({
      data: [row({ task_type: TaskType.TRAINING, id: 'tq_t' })],
    } as any);
    const { container } = render(<RunningWork modelId="m_gemma" />);
    await waitFor(() => expect(getActiveTasks).toHaveBeenCalled());
    expect(container.firstChild).toBeNull();
  });

  it('shows only work for the model on screen', async () => {
    vi.mocked(getActiveTasks).mockResolvedValue({
      data: [row({ entity_id: 'm_other' })],
    } as any);
    const { container } = render(<RunningWork modelId="m_gemma" />);
    await waitFor(() => expect(getActiveTasks).toHaveBeenCalled());
    expect(container.firstChild).toBeNull();
  });

  it('renders unreported progress as an EMPTY bar, never a full one', async () => {
    // A task that has not reported yet is at the START of its work. Showing a
    // full bar would say the opposite.
    vi.mocked(getActiveTasks).mockResolvedValue({
      data: [row({ progress: null, status: 'queued' })],
    } as any);
    const { container } = render(<RunningWork modelId="m_gemma" />);

    await waitFor(() => expect(screen.getByText('queued')).toBeInTheDocument());
    const bar = container.querySelector('span[style*="width"]') as HTMLElement;
    expect(bar.style.width).toBe('0%');
  });

  it('a failed poll leaves the list alone rather than emptying it', async () => {
    // "nothing is running" and "I could not ask" look identical, and the first
    // is the reading that stops someone investigating.
    //
    // THE TIMER ADVANCE IS LOAD-BEARING. An earlier version waited 40ms while
    // the poll interval is 5s, so the failing call never happened and the test
    // passed against a catch block that cleared the list.
    vi.useFakeTimers();
    try {
      vi.mocked(getActiveTasks)
        .mockResolvedValueOnce({ data: [row()] } as any)
        .mockRejectedValue(new Error('network'));
      render(<RunningWork modelId="m_gemma" />);

      await vi.waitFor(() =>
        expect(screen.getByText('Fitting a J-lens')).toBeInTheDocument()
      );
      expect(getActiveTasks).toHaveBeenCalledTimes(1);

      // Reach the failing poll, then let React flush whatever the catch did.
      // Without the second advance the render has not happened yet and a catch
      // block that cleared the list still passes.
      await vi.advanceTimersByTimeAsync(5100);
      expect(getActiveTasks).toHaveBeenCalledTimes(2);
      await vi.advanceTimersByTimeAsync(50);

      // The row is still on screen: the failure changed nothing.
      expect(screen.getByText('Fitting a J-lens')).toBeInTheDocument();
    } finally {
      vi.useRealTimers();
    }
  });
});

describe('the summary counts what is actually running', () => {
  it('says "1 running · 2 queued", never "3 jobs running"', () => {
    // A single-GPU queue runs one at a time, so a bare total implies
    // concurrency the product does not have.
    //
    // MUTATION CONTROL: return `${rows.length} jobs running` and this fails.
    expect(
      summarise([{ status: 'running' }, { status: 'queued' }, { status: 'queued' }])
    ).toBe('1 running · 2 queued');
  });

  it('surfaces a stopped job rather than folding it into a total', () => {
    // A worker that died is the thing a reader most needs to notice.
    expect(
      summarise([{ status: 'running' }, { status: 'orphaned' }])
    ).toBe('1 running · 1 stopped reporting');
  });

  it('says idle rather than an empty string', () => {
    expect(summarise([])).toBe('idle');
  });
});

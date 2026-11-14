/**
 * The fit tile must say WHAT it is fitting and HOW it is going.
 *
 * A J-lens fit is the longest-running job in the product — 2h51m for gemma,
 * with LFM2 queued ~3h behind it — and this tile rendered `jlens_fit` against
 * the raw model id `m_98a4b47e` with a bare percentage. With two fits queued
 * there was no way to tell which one was moving.
 *
 * Three traps this pins:
 *   * elapsed measured from `created_at` counts QUEUE time as fit time;
 *   * an unreported count rendered as 0 claims the fit has done nothing;
 *   * a status the badge does not know falls through to amber "Queued", so a
 *     job whose worker died looked like one waiting its turn.
 *
 * MUTATION CONTROLS:
 *   * elapsed from created_at when started_at exists -> "queue time" fails
 *   * drop the orphaned badge case                   -> "stopped reporting" fails
 *   * drop the jlens_* label entries                 -> "J-Lens fit" fails
 *   * render absent counts as 0                      -> "absent, never zero" fails
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';

const mockStore = {
  activeTasks: [] as any[],
  activeLoading: false,
  activeError: null as string | null,
  fetchActiveTasks: vi.fn(),
};

vi.mock('../../stores/taskQueueStore', () => ({
  useTaskQueueStore: () => mockStore,
}));

import { ActiveOperationsSection, elapsedLabel } from './ActiveOperationsSection';

function fitRow(overrides: Record<string, any> = {}) {
  return {
    id: 'tq_1',
    task_id: 'celery-1',
    task_type: 'jlens_fit',
    entity_id: 'm_98a4b47e',
    entity_type: 'model',
    status: 'running',
    progress: 52.8,
    error_message: null,
    retry_params: null,
    retry_count: 0,
    can_retry: false,
    created_at: new Date(Date.now() - 3 * 3600 * 1000).toISOString(),
    started_at: new Date(Date.now() - 600 * 1000).toISOString(),
    completed_at: null,
    updated_at: null,
    entity_info: {
      name: 'gemma-2-2b-it',
      repo_id: 'google/gemma-2-2b-it',
      prompts_seen: 634,
      total_prompts: 1200,
      details: '634 / 1,200 prompts · delta 1.03e-3 (target 1e-03)',
    },
    ...overrides,
  };
}

beforeEach(() => {
  mockStore.activeTasks = [];
  mockStore.activeError = null;
});

describe('ActiveOperationsSection — J-lens fit tile', () => {
  it('names the MODEL, not the raw entity id', () => {
    mockStore.activeTasks = [fitRow()];
    render(<ActiveOperationsSection />);
    expect(screen.getByText('gemma-2-2b-it')).toBeInTheDocument();
    expect(screen.queryByText('m_98a4b47e')).not.toBeInTheDocument();
  });

  it('renders a human task-type label, not the raw enum string', () => {
    // MUTATION CONTROL: remove the jlens_* entries from TASK_TYPE_LABELS.
    mockStore.activeTasks = [fitRow()];
    render(<ActiveOperationsSection />);
    expect(screen.getByText('J-Lens fit')).toBeInTheDocument();
    expect(screen.queryByText('jlens_fit')).not.toBeInTheDocument();
  });

  it('shows the counts alongside the percentage', () => {
    mockStore.activeTasks = [fitRow()];
    render(<ActiveOperationsSection />);
    expect(screen.getByText('52.8%')).toBeInTheDocument();
    expect(screen.getByText(/634 \/ 1,200 prompts/)).toBeInTheDocument();
  });

  it('renders a stopped worker as STOPPED REPORTING, never as Queued', () => {
    /**
     * The badge was binary running/queued, so `orphaned` fell through to amber
     * "Queued" — a dead job presented as one waiting its turn, which is the
     * confusion the backend janitor exists to remove.
     *
     * MUTATION CONTROL: delete the `status === 'orphaned'` branch.
     */
    mockStore.activeTasks = [fitRow({ status: 'orphaned' })];
    render(<ActiveOperationsSection />);
    expect(screen.getByText('Stopped reporting')).toBeInTheDocument();
    expect(screen.queryByText('Queued')).not.toBeInTheDocument();
  });
});

describe('elapsedLabel — the two clocks are not the same clock', () => {
  it('measures a RUNNING job from started_at, not created_at', () => {
    /**
     * `created_at` is enqueue time. LFM2 waited ~3h behind gemma, so measuring
     * from it would have reported a four-hour fit after one hour of work.
     *
     * MUTATION CONTROL: use created_at when started_at is present.
     */
    const label = elapsedLabel({
      status: 'running',
      created_at: new Date(Date.now() - 3 * 3600 * 1000).toISOString(),
      started_at: new Date(Date.now() - 600 * 1000).toISOString(),
    });
    expect(label).toMatch(/^Elapsed 10m/);
    expect(label).not.toMatch(/3h/);
  });

  it('calls a QUEUED job’s wait "Queued", never "Elapsed"', () => {
    const label = elapsedLabel({
      status: 'queued',
      created_at: new Date(Date.now() - 7200 * 1000).toISOString(),
      started_at: null,
    });
    expect(label).toMatch(/^Queued 2h/);
    expect(label).not.toMatch(/Elapsed/);
  });

  it('appends the heartbeat age when one is known', () => {
    const label = elapsedLabel({
      status: 'running',
      created_at: null,
      started_at: new Date(Date.now() - 600 * 1000).toISOString(),
      entity_info: { seconds_since_heartbeat: 4 },
    });
    expect(label).toMatch(/beat 4s ago/);
  });

  it('says nothing about a heartbeat it has not been told about', () => {
    const label = elapsedLabel({
      status: 'running',
      created_at: null,
      started_at: new Date(Date.now() - 600 * 1000).toISOString(),
    });
    expect(label).not.toMatch(/beat/);
  });
});

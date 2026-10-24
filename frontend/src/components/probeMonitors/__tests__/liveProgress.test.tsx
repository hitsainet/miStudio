/**
 * A live probe job says how far it has got (operator, 2026-10-21: "The probe run does not show
 * progress").
 *
 * The panel is mounted with only the network mocked, so removing the live block from the Judge tab,
 * its polling, or the run tile's live line turns a test here red. Controls are recorded in
 * 0xcc/reviews/live_progress_2026-10-21.md:
 *   U1  <JudgeRunLiveProgress> removed from the Judge tab        → the panel test
 *   U2  the Judge tab's 5 s refresh removed                      → the polling test
 *   U3  <RunLiveProgress> removed from RunTile                   → the run tile test
 *   U4  an unreachable rate rendered as a recall                 → the not-reachable test
 */
import { act, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProbeJudgeRun, ProbeRun } from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({
  listDatasets: vi.fn(),
  listRuns: vi.fn(),
  listProbes: vi.fn(),
  listJudgeRuns: vi.fn(),
  listJudgePrompts: vi.fn(),
}));

vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: api }));
vi.mock('../../../hooks/useProbeMonitorWebSocket', () => ({ useProbeMonitorWebSocket: () => undefined }));

import { ProbeMonitorsPanel } from '../../panels/ProbeMonitorsPanel';
import { JudgeRunResults } from '../JudgeRunResults';
import {
  JudgeRunLiveProgress,
  describeJudgeProgress,
  describeRunProgress,
  formatRate,
  formatTimeLeft,
} from '../LiveProgress';
import { RunTile } from '../RunTile';
import { useProbeMonitorsStore } from '../../../stores/probeMonitorsStore';

const JUDGE_DETAIL = {
  kind: 'judge' as const, phase: 'judging' as const, rows_done: 812, rows_total: 1370,
  set_index: 1, set_count: 2, set_id: 'pmd_us', set_name: 'US test',
  set_rows_done: 127, set_rows_total: 685, rate_per_second: 2.1, eta_seconds: 540,
  parse_failures: 3, updated_at: '2026-10-21T12:00:00Z',
};

function judgeRun(overrides: Partial<ProbeJudgeRun> = {}): ProbeJudgeRun {
  return {
    id: 'pmj_live', endpoint: 'http://j/v1', model: 'qwen', prompt_version: 'pii-presence/v1',
    prompt_id: 'jp_1', prompt_hash: 'a', prompt_title: 'Personal data present (PII)',
    prompt_behaviour: 'pii', counts_toward_rung: null, rung_note: null, dataset_ids: ['pmd_us'],
    probe_id: null, status: 'running', progress: 59.3, progress_detail: JUDGE_DETAIL,
    parse_failures: 0, max_rows_per_set: 2000, metrics: {}, error_message: null,
    ...overrides,
  } as ProbeJudgeRun;
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  window.localStorage.clear();
  api.listDatasets.mockResolvedValue([]);
  api.listRuns.mockResolvedValue([]);
  api.listProbes.mockResolvedValue([]);
  api.listJudgeRuns.mockResolvedValue([]);
  api.listJudgePrompts.mockResolvedValue({ prompts: [], placeholders: {} });
});

afterEach(() => {
  vi.useRealTimers();
});

describe('the words', () => {
  it('reads like the operator asked: set · rows of rows · speed · time left', () => {
    expect(describeJudgeProgress(JUDGE_DETAIL)).toBe('US test · 812 of 1,370 · 2.1 rows/s · about 9 min left');
  });

  it('says it is still measuring rather than guessing a speed', () => {
    const line = describeJudgeProgress({ ...JUDGE_DETAIL, rate_per_second: null, eta_seconds: null });
    expect(line).toBe('US test · 812 of 1,370 · measuring speed');
  });

  it('rounds time the way a person says it', () => {
    expect(formatTimeLeft(30)).toBe('under a minute left');
    expect(formatTimeLeft(266)).toBe('about 4 min left');
    expect(formatTimeLeft(2 * 3600 + 15 * 60)).toBe('about 2 h 15 min left');
    expect(formatTimeLeft(null)).toBeNull();
  });

  it('a slow rate is per minute', () => {
    expect(formatRate(0.1, 'batches')).toBe('6.0 batches/min');
    expect(formatRate(2.13, 'rows')).toBe('2.1 rows/s');
    expect(formatRate(null, 'rows')).toBeNull();
  });

  it('a training run names its step, its units and the item inside it', () => {
    expect(
      describeRunProgress({
        kind: 'training', stage: 'training', stage_index: 4, stage_count: 8, done: 2, total: 8,
        unit: 'probes', current: { name: 'mean at layer 12', done: 14, total: 400, unit: 'epochs' },
        rate_per_second: null, eta_seconds: null, updated_at: '',
      }),
    ).toBe('Training probes (step 5 of 8) · 2 of 8 probes · mean at layer 12: 14 of 400 epochs');
  });
});

describe('a live judge run shows its bar and its line', () => {
  it('renders the bar, the line, the set and the unreadable count', () => {
    render(<JudgeRunLiveProgress run={judgeRun()} />);
    expect(screen.getByTestId('judge-live-line')).toHaveTextContent('US test · 812 of 1,370 · 2.1 rows/s');
    const block = screen.getByTestId('judge-live-progress');
    expect(block).toHaveTextContent('Set 2 of 2: 127 of 685 rows');
    expect(block).toHaveTextContent('3 replies unreadable so far');
  });

  it('a finished run shows none of it', () => {
    render(<JudgeRunLiveProgress run={judgeRun({ status: 'completed' })} />);
    expect(screen.queryByTestId('judge-live-progress')).toBeNull();
  });

  it('the Judge tab mounts it for a live run', async () => {
    api.listJudgeRuns.mockResolvedValue([judgeRun()]);
    useProbeMonitorsStore.setState({ activeTab: 'judge', judgeRuns: [] });
    render(<ProbeMonitorsPanel />);
    const row = await screen.findByTestId('judge-row');
    expect(within(row).getByTestId('judge-live-line')).toHaveTextContent('812 of 1,370');
  });

  it('the Judge tab REFRESHES while a run is live', async () => {
    vi.useFakeTimers();
    api.listJudgeRuns.mockResolvedValue([judgeRun()]);
    useProbeMonitorsStore.setState({ activeTab: 'judge', judgeRuns: [] });
    render(<ProbeMonitorsPanel />);
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    const before = api.listJudgeRuns.mock.calls.length;
    expect(before).toBeGreaterThanOrEqual(1);
    await act(async () => { await vi.advanceTimersByTimeAsync(5000); });
    expect(api.listJudgeRuns.mock.calls.length).toBe(before + 1);
  });

  it('and stops refreshing once nothing is live', async () => {
    vi.useFakeTimers();
    api.listJudgeRuns.mockResolvedValue([judgeRun({ status: 'completed' })]);
    useProbeMonitorsStore.setState({ activeTab: 'judge', judgeRuns: [] });
    render(<ProbeMonitorsPanel />);
    await act(async () => { await vi.advanceTimersByTimeAsync(10); });
    const before = api.listJudgeRuns.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(15000); });
    expect(api.listJudgeRuns.mock.calls.length).toBe(before);
  });
});

describe('a live training run says what its bar counts', () => {
  it('the tile shows the stage reading', () => {
    const run = {
      id: 'pmr_1', model_id: 'm_1', train_dataset_id: 'pmd_t', eval_dataset_ids: [],
      calibration_dataset_id: null, config: {}, stage: 'token_capture', status: 'running',
      progress: 48, error_message: null, celery_task_id: null, gpu_request: null, gpu_uuid: null,
      artifact_dir: null, environment: {}, layer_selection: null,
      created_at: '2026-10-21T10:00:00Z', updated_at: '2026-10-21T10:05:00Z', completed_at: null,
      progress_detail: {
        kind: 'training', stage: 'token_capture', stage_index: 3, stage_count: 8, done: 120,
        total: 400, unit: 'batches', current: null, rate_per_second: 3.2, eta_seconds: 88,
        updated_at: '2026-10-21T10:05:00Z',
      },
    } as unknown as ProbeRun;
    render(<RunTile run={run} onCancel={vi.fn()} onDelete={vi.fn()} />);
    expect(screen.getByTestId('run-live-line')).toHaveTextContent(
      'Capturing token activations (step 4 of 8) · 120 of 400 batches · 3.2 batches/s · about 1 min left in this step',
    );
  });
});

describe('a judge that cannot run at the probe’s rate says so, never "recall 0"', () => {
  it('renders the server’s sentence', () => {
    const run = judgeRun({
      status: 'completed',
      metrics: {
        per_set: {
          pmd_us: {
            set_name: 'US test', scored: true, auroc: 0.9, n_positive: 685, n_negative: 685,
            decision: { rule: 'yes counts', recall: 0.86, false_positive_rate: 0.045, precision: 0.95, judged_positive: 620, judged: 1370 },
            at_probe_fpr: {
              target_fpr: 0.01, reachable: false, recall: null, own_false_positive_rate: 0.045,
              own_recall: 0.86,
              reason: "this judge can't run at the probe's 1% false-positive rate; at its own rate (4.5%) it catches 86%",
            },
          },
        },
      } as never,
    });
    render(<JudgeRunResults run={run} datasetNames={{}} />);
    const cell = screen.getByTestId('judge-at-probe-unreachable');
    expect(cell).toHaveTextContent("this judge can't run at the probe's 1% false-positive rate; at its own rate (4.5%) it catches 86%");
    expect(screen.getByTestId('judge-set-row')).not.toHaveTextContent('0.0% at 1.0%');
  });
});

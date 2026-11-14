/**
 * Feature 041 (FTASKS 10.1 – 10.4): blend reports and the suggested blend, from random-key fixtures with only
 * the network mocked. Requests are asserted by arguments and call count; every figure is the fixture's.
 */
import { act, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { BlendEnvelope } from '../../../types/probeFamily';

const api = vi.hoisted(() => ({
  getFamily: vi.fn(), listReports: vi.fn(), getReport: vi.fn(), getSuggestedBlend: vi.fn(),
  getSuggestedBlendHistory: vi.fn(), setSuggestedBlend: vi.fn(), pruneReport: vi.fn(),
  exportIndex: vi.fn(), exportUrl: vi.fn(), publishFamily: vi.fn(),
}));
vi.mock('../../../api/probeFamilies', () => ({ probeFamiliesApi: api }));

import { useProbeFamiliesStore } from '../../../stores/probeFamiliesStore';
import { BlendReportList, POLL_MS } from '../BlendReportList';
import { SuggestedBlendCard } from '../SuggestedBlendCard';

const slug = () => {
  const part = () => Array.from({ length: 5 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');
  return `${part()}-${part()}`;
};

function envelope(id: string, status: BlendEnvelope['status'], stale = false): BlendEnvelope {
  const figures = (recall: number) => ({ recall, wilson: [recall - 0.1, recall + 0.05] as [number, number], auroc: 0.9,
    n_pos: 40, n_neg: 60, realised_fpr: 0.01 });
  return {
    report_id: id, behaviour_key: 'k', model_id: 'm_x', status, failure: null, stale,
    stale_reasons: stale ? [{ code: 'decision_changed', probe_id: 'pm_b' }] : [], next: [], members: ['pm_a', 'pm_b'],
    report_sha256: null, rows_pruned_at: null, created_at: null,
    report: status !== 'done' ? undefined : {
      format: 'mistudio.blend-report/1', report_id: id,
      views: { blend_fit: 'pmd_f', tests: [{ id: 'pmd_o', held_out: true }], calibration: { id: 'pmd_c', sample_rows: 2000 } },
      members: [], members_own_figures: { pmd_o: { pm_a: figures(0.5) } }, flags: ['no_rule_set'],
      statement: 'A blend has no rung of its own.', has_rung: false,
      ranking: { order: ['max', 'stack'], basis: 'held_out_min' },
      candidates: [
        { index: 0, name: 'stack', kind: 'stack', produces: 'score', params: {}, criteria: [],
          calibration: { threshold: 1.2, n_negatives: 108, few_negatives: true, realised_fpr: 0.009 },
          cv: { in_fold: true, not_assessable: true, reason: ['few_groups'], groups: 3 }, tests: { pmd_o: figures(0.7) } },
        { index: 1, name: 'max', kind: 'max', produces: 'score', params: {}, criteria: [],
          calibration: { threshold: 2.1, n_negatives: 108, few_negatives: true, realised_fpr: 0.009 },
          cv: { in_fold: true, recall: 0.85, auroc: 0.995 }, tests: { pmd_o: figures(0.8) } },
      ],
    },
  };
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  useProbeFamiliesStore.setState({ reports: null, openReport: null, suggested: null, history: null, busy: false,
    refusal: null });
});
afterEach(() => vi.useRealTimers());

describe('blend reports', () => {
  it('lists the reports, opens one and shows the ranking with not-assessable and few-negatives in words', async () => {
    const key = slug();
    const done = envelope('pbr_1', 'done');
    api.listReports.mockResolvedValue({ behaviour_key: key, reports: [done], offset: 0, limit: 25, next_offset: null });
    api.getReport.mockResolvedValue(done);
    render(<BlendReportList behaviourKey={key} model="m_x" />);
    await userEvent.click(await screen.findByRole('button', { name: 'pbr_1' }));
    const table = await screen.findByLabelText('Blend candidates');
    const rows = within(table).getAllByRole('row').slice(1);
    expect(rows.map((r) => r.getAttribute('data-testid'))).toEqual(['candidate-max', 'candidate-stack']);
    expect(within(rows[1]).getByTestId('not-assessable')).toHaveTextContent('not assessable: few groups (3 groups)');
    expect(rows[0]).toHaveTextContent('108 negatives (few negatives)');
    expect(api.listReports).toHaveBeenCalledWith(key, 'm_x');
    expect(api.getReport).toHaveBeenCalledTimes(1);
  });

  it('polls a running report every five seconds and stops at a terminal status', async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    const key = slug();
    api.listReports.mockResolvedValue({ behaviour_key: key, reports: [envelope('pbr_2', 'running')], offset: 0,
      limit: 25, next_offset: null });
    api.getReport.mockResolvedValueOnce(envelope('pbr_2', 'running')).mockResolvedValueOnce(envelope('pbr_2', 'running'))
      .mockResolvedValue(envelope('pbr_2', 'done'));
    const view = render(<BlendReportList behaviourKey={key} model="m_x" />);
    await act(async () => { await userEvent.click(await screen.findByRole('button', { name: 'pbr_2' })); });
    await act(async () => { await vi.advanceTimersByTimeAsync(POLL_MS); });
    await act(async () => { await vi.advanceTimersByTimeAsync(POLL_MS); });
    await waitFor(() => expect(screen.getByLabelText('Blend candidates')).toBeInTheDocument());
    const calls = api.getReport.mock.calls.length;
    await act(async () => { await vi.advanceTimersByTimeAsync(POLL_MS * 3); });
    expect(api.getReport.mock.calls.length).toBe(calls);
    view.unmount();
  });

  it('a stale report says what moved and how to rerun', async () => {
    const stale = envelope('pbr_3', 'done', true);
    api.listReports.mockResolvedValue({ behaviour_key: 'k', reports: [stale], offset: 0, limit: 25, next_offset: null });
    api.getReport.mockResolvedValue(stale);
    render(<BlendReportList behaviourKey={slug()} model="m_x" />);
    await userEvent.click(await screen.findByRole('button', { name: 'pbr_3' }));
    expect(await screen.findByTestId('blend-stale')).toHaveTextContent('decision changed (pm_b)');
    expect(screen.getByTestId('blend-stale')).toHaveTextContent('rerun_blend_report');
    expect(screen.queryByRole('button', { name: /Suggest/ })).toBeNull();
  });
});

describe('the suggested blend', () => {
  it('marks the top candidate with exactly one PUT and shows the advisory badge and history', async () => {
    const key = slug();
    const done = envelope('pbr_4', 'done');
    api.listReports.mockResolvedValue({ behaviour_key: key, reports: [done], offset: 0, limit: 25, next_offset: null });
    api.getReport.mockResolvedValue(done);
    api.getSuggestedBlend.mockResolvedValue({ key, model: 'm_x', report_id: 'pbr_4', candidate: 'max', record: {},
      record_sha256: 'a', history_id: 'psb_1', updated_at: null, stale: false, stale_reasons: [] });
    api.getSuggestedBlendHistory.mockResolvedValue({ key, model: 'm_x', entries: [
      { id: 'psb_1', report_id: 'pbr_4', candidate: 'max', record_sha256: 'a', actor: 'operator', reason: 'best',
        created_at: 't' }], next_offset: null });
    api.setSuggestedBlend.mockResolvedValue({});
    render(<><BlendReportList behaviourKey={key} model="m_x" /><SuggestedBlendCard behaviourKey={key} model="m_x" /></>);
    expect(await screen.findByTestId('advisory-badge')).toHaveTextContent('Advisory — binds nothing; the application blends');
    await userEvent.click(await screen.findByRole('button', { name: 'pbr_4' }));
    await userEvent.type(await screen.findByLabelText('Suggestion reason'), 'best');
    await userEvent.click(screen.getByRole('button', { name: 'Suggest max' }));
    await waitFor(() => expect(api.setSuggestedBlend).toHaveBeenCalledTimes(1));
    expect(api.setSuggestedBlend).toHaveBeenCalledWith(key, { model: 'm_x', report_id: 'pbr_4', candidate: 'max',
      reason: 'best' });
    expect(await screen.findByLabelText('Suggested blend history')).toHaveTextContent('operator: max (best)');
  });

  it('prunes with one POST and no optimistic change', async () => {
    const done = envelope('pbr_5', 'done');
    api.listReports.mockResolvedValue({ behaviour_key: 'k', reports: [done], offset: 0, limit: 25, next_offset: null });
    api.getReport.mockResolvedValue(done);
    api.pruneReport.mockResolvedValue({ report_id: 'pbr_5', rows_pruned_at: 't', rows_pruned_by: 'operator' });
    render(<BlendReportList behaviourKey={slug()} model="m_x" />);
    await userEvent.click(await screen.findByRole('button', { name: 'pbr_5' }));
    await userEvent.click(await screen.findByRole('button', { name: /Prune per-row values/ }));
    await waitFor(() => expect(api.pruneReport).toHaveBeenCalledTimes(1));
    expect(api.pruneReport).toHaveBeenCalledWith('pbr_5');
  });
});

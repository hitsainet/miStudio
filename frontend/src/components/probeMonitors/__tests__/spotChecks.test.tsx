/**
 * Session spot-checks in the UI (Feature 038 FR-15, FR-17). Network mocked; requests asserted by their
 * exact body and call count; the report renders the statement, the 2 × 2 table and the disagreements.
 */
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

const api = vi.hoisted(() => ({
  availability: vi.fn(), list: vi.fn(), get: vi.fn(), start: vi.fn(), items: vi.fn(), cancel: vi.fn(), pin: vi.fn(),
  exportUrl: vi.fn((id: string) => `/api/v1/probe-monitors/spot-checks/${id}/disagreements.jsonl`),
}));
vi.mock('../../../api/spotChecks', () => ({ spotChecksApi: api }));

import { SpotCheckSection } from '../SpotCheckSection';
import { SpotCheckReport } from '../SpotCheckReport';
import type { SpotCheck } from '../../../api/spotChecks';

const PROMPTS = [{ id: 'jp_pii_presence_v1', key: 'pii-presence/v1', title: 'Personal data present (PII)', archived: false }];
const stats = (tp: number, fp: number, fn: number, tn: number) => ({
  n: tp + fp + fn + tn, tp, fp, fn, tn, agreement: (tp + tn) / (tp + fp + fn + tn),
  agreement_ci: { low: 0.7, high: 0.95, resamples: 1000, alpha: 0.05 }, kappa: 0.69,
});
const CHECK: SpotCheck = {
  id: 'psc_1', behaviour: 'pii', prompt_id: 'jp_pii_presence_v1', prompt_key: 'pii-presence/v1', endpoint: 'http://j/v1',
  model: 'judge', status: 'completed', progress: 1, error_message: null, partial: false, parse_failures: 0,
  pinned: false, excerpts_expire_at: null, created_at: null, sample_spec: {},
  metrics: {
    statement: 'The judge is a second opinion, not ground truth. Disagreement means one of the two is wrong, not which.',
    overall_note: 'Overall figures reflect the sampling design.', partial: false,
    shortfall: { near_threshold: { requested: 33, returned: 3, short: 30 } },
    probes: { pm_1: { millm_probe_id: 'pr_1', newest_turn_only: false, overall: stats(7, 2, 1, 10),
      strata: { fired: stats(7, 2, 0, 0), near_threshold: stats(0, 0, 0, 0), not_fired: stats(0, 0, 1, 10) },
      n_parse_failed: 0, n_not_recorded: 2, judge_saw: { same_window: 20 }, threshold_revisions: [1], disagreements: 3 } },
    newest_turn_probes: {},
  },
};

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockClear());
  api.availability.mockResolvedValue({ available: true, reason: null, url: 'http://millm' });
  api.list.mockResolvedValue({ spot_checks: [] });
  api.start.mockResolvedValue({ id: 'psc_2', status: 'pending' });
  api.items.mockResolvedValue({ total: 1, items: [{ id: 1, session_id: 'ps_9', request_id: 'r', turn_index: 0, stratum: 'fired',
    probe_id: 'pm_1', window: 'last_user', judge_saw: 'same_window', probe_score: 0.9, threshold: 0.5, probe_fired: true,
    not_recorded: false, judged_text: [{ role: 'user', content: 'call me at 555' }], judge_reply: '{}', judge_value: 'no',
    judge_positive: false, parse_failed: false, disagrees: true }] });
});

describe('the spot-check section', () => {
  it('posts exactly the form, once', async () => {
    render(<SpotCheckSection behaviourKey="pii" prompts={PROMPTS} />);
    const form = await screen.findByRole('form', { name: 'Start a spot-check' });
    await userEvent.type(within(form).getByLabelText('Judge endpoint'), 'http://millm-backend:8000/v1');
    await userEvent.type(within(form).getByLabelText('Judge model'), 'qwen');
    await userEvent.clear(within(form).getByLabelText('Turns'));
    await userEvent.type(within(form).getByLabelText('Turns'), '50');
    expect(screen.getByText(/live chats wait/)).toBeInTheDocument();
    await userEvent.click(within(form).getByRole('button', { name: 'Start spot-check' }));
    await waitFor(() => expect(api.start).toHaveBeenCalledTimes(1));
    const body = api.start.mock.calls[0][0];
    expect({ ...body, since: undefined }).toEqual({
      behaviour: 'pii', prompt_id: 'jp_pii_presence_v1', endpoint: 'http://millm-backend:8000/v1', model: 'qwen', size: 50, since: undefined,
    });
    expect(Date.now() - Date.parse(body.since)).toBeGreaterThan(6.9 * 86_400_000);
  });

  it('without miLLM it hides the form and gives the reason', async () => {
    api.availability.mockResolvedValue({ available: false, reason: 'miLLM at http://m does not serve /api/probes/sessions/sample yet', url: 'http://m' });
    render(<SpotCheckSection behaviourKey="pii" prompts={PROMPTS} />);
    expect(await screen.findByTestId('spot-unavailable')).toHaveTextContent('does not serve');
    expect(screen.queryByRole('form', { name: 'Start a spot-check' })).toBeNull();
  });
});

describe('the report', () => {
  it('leads with the statement and renders the 2 × 2 and the disagreements', async () => {
    render(<SpotCheckReport check={CHECK} />);
    expect(screen.getByText(/second opinion, not ground truth/)).toBeInTheDocument();
    const table = screen.getByRole('table', { name: 'Two by two table for pm_1' });
    const cells = within(table).getAllByRole('cell').map((c) => c.textContent);
    expect(cells).toEqual(['7', '2', '1', '10']);
    expect(screen.getByText(/Agreement 85.0%/)).toBeInTheDocument();
    expect(screen.getByText(/Only 3 of 33 near threshold turns/)).toBeInTheDocument();
    const list = await screen.findByRole('list', { name: 'Disagreements' });
    await waitFor(() => expect(within(list).getByText(/call me at 555/)).toBeInTheDocument());
    expect(api.items).toHaveBeenCalledTimes(1);
  });
});

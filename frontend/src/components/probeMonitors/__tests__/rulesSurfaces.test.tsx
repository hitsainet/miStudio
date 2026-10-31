/**
 * Feature 040 S3 (FTASKS 23.4): the rule-set panel and the rules ∪ probe evaluation, rendered from random-key
 * fixtures with only the network mocked. Each request is asserted by its arguments and call count; every figure
 * on screen is the fixture's; a refusal renders its message and its `next` call.
 */
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { APIError } from '../../../api/client';
import type { RulesEvaluation as RulesEvaluationBody } from '../../../types/probeMonitor';

const api = vi.hoisted(() => ({
  updateViewColumns: vi.fn(), getBreakdown: vi.fn(), createSuccessReport: vi.fn(), listSuccessReports: vi.fn(),
  readSuccessReport: vi.fn(), compareModels: vi.fn(), validateRuleSet: vi.fn(), evaluateRuleSignals: vi.fn(),
  startRulesEvaluation: vi.fn(), getRulesEvaluation: vi.fn(), listRulesEvaluations: vi.fn(), cancelRulesEvaluation: vi.fn(),
  revokeCodeRule: vi.fn(), reapproveCodeRule: vi.fn(),
}));
const probeApi = vi.hoisted(() => ({ getReport: vi.fn(), listDatasets: vi.fn() }));
vi.mock('../../../api/behaviourDetectors', () => ({ behaviourDetectorsApi: api }));
vi.mock('../../../api/probeMonitors', () => ({ probeMonitorsApi: probeApi }));

import { useBehaviourDetectorsStore } from '../../../stores/behaviourDetectorsStore';
import { RuleSetPanel } from '../RuleSetPanel';
import { RulesEvaluation } from '../RulesEvaluation';

const slug = () => {
  const part = () => Array.from({ length: 5 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');
  return `${part()}-${part()}`;
};

const rates = (pf: number, pn: number, nf: number, nn: number) => ({
  positives: { n: pn, flagged: pf, recall: pf / pn, ci: null }, negatives: { n: nn, flagged: nf, fpr: nf / nn, ci: null },
});

function evaluation(overrides: Partial<RulesEvaluationBody> = {}): RulesEvaluationBody {
  return {
    evaluation_id: `pre_${slug()}`, status: 'completed', probe_id: 'pm_1', view_ids: ['pmd_1'], behaviour_key: 'k',
    behaviour_version: 1, rule_set_digest: 'a'.repeat(64), task_queue_id: 'tq_1', progress: null, error: null,
    created_at: null,
    result: {
      reasons: [],
      views: [{ view_id: 'pmd_1', evaluation_id: 'pme_1', rows: 12, reasons: ['no_veto_signals'],
        unknown: { n: 1, share: 0.083, over_share: true, share_max: 0.01, by_reason: { rule_budget_exceeded: 1 } },
        detectors: { rules_alone: rates(3, 6, 1, 6), probe_alone: rates(3, 6, 2, 6), rules_or_probe: rates(4, 6, 2, 6) } }],
      calibration: { global: { threshold: 0.73, realised_union_fpr: 0.01, rule_share: 0.004, budget: 10 } },
    },
    stale: { stale: false, moved: [], next: [] },
    ...overrides,
  };
}

beforeEach(() => {
  Object.values(api).forEach((fn) => fn.mockReset());
  Object.values(probeApi).forEach((fn) => fn.mockReset());
  useBehaviourDetectorsStore.setState({ ruleValidation: null, ruleVerdict: null, rulesEvaluation: null,
    rulesEvaluations: null, busy: false, refusal: null });
  api.listRulesEvaluations.mockResolvedValue({ behaviour_key: 'x', rules_evaluations: [], total: 0, next_cursor: null });
});

describe('the rule-set panel', () => {
  it('validates the behaviour\'s set and names the failing cases', async () => {
    const key = slug();
    api.validateRuleSet.mockResolvedValue({ valid: false, code: 'rule_set_cases_failed', message: 'two cases fail',
      rule_set_digest: 'b'.repeat(64), engine_digest: 'c'.repeat(64), engine_version: 1,
      failing_cases: [{ case_index: 0, case_id: 'k-one', where: 'hit' }, { case_index: 3, case_id: 'k-two', where: 'vetoed' }] });
    render(<RuleSetPanel behaviourKey={key} />);
    await userEvent.click(screen.getByRole('button', { name: 'Validate rule set' }));
    expect(api.validateRuleSet).toHaveBeenCalledTimes(1);
    expect(api.validateRuleSet).toHaveBeenCalledWith({ behaviour_key: key });
    expect(await screen.findAllByTestId('failing-case')).toHaveLength(2);
    expect(screen.getByTestId('rule-validation')).toHaveTextContent('rule set cases failed');
  });

  it('tries the set on a text and highlights the spans, by code point', async () => {
    const key = slug();
    api.evaluateRuleSignals.mockResolvedValue({ rule_match: true, rule_veto: false, unknown: false, reason: null,
      fired: [{ signal_id: 'v_one', kind: 'pattern', pending: false, spans: [{ message_index: 0, start: 2, end: 4 }] }],
      expressions: {}, rule_set_digest: 'd', engine_digest: 'e', scope: 'all' });
    render(<RuleSetPanel behaviourKey={key} />);
    await userEvent.type(screen.getByLabelText('Text to try'), '日本12x');
    await userEvent.click(screen.getByRole('button', { name: 'Try on this text' }));
    expect(api.evaluateRuleSignals).toHaveBeenCalledWith({ behaviour_key: key, text: '日本12x' });
    expect(await screen.findByTestId('rule-verdict')).toHaveTextContent('Match: yes · Veto: no');
    const marks = screen.getByTestId('rule-spans').querySelectorAll('mark');
    expect([...marks].map((m) => m.textContent).join('')).toBe('12');
  });

  it('shows a refusal with its next call', async () => {
    api.validateRuleSet.mockRejectedValue(new APIError('no rule set', 409, {
      code: 'no_rule_set', message: 'version 2 names no rule set',
      next: [{ tool: 'new_behaviour_version', arguments: {}, why: 'attach one' }] }));
    render(<RuleSetPanel behaviourKey={slug()} />);
    await userEvent.click(screen.getByRole('button', { name: 'Validate rule set' }));
    expect(await screen.findByTestId('detector-refusal')).toHaveTextContent('new_behaviour_version');
  });
});

describe('rules and the probe together', () => {
  it('starts on the probe\'s evaluated views, renders the three detectors, polls and cancels', async () => {
    const key = slug();
    probeApi.getReport.mockResolvedValue({ evaluations: [{ id: 'pme_1', dataset_id: 'pmd_1' }, { id: 'pme_2', dataset_id: 'pmd_2' }] });
    api.startRulesEvaluation.mockResolvedValue(evaluation({ status: 'running', result: null }));
    render(<RulesEvaluation behaviourKey={key} probes={[{ id: 'pm_1' } as never]} />);
    await waitFor(() => expect(api.listRulesEvaluations).toHaveBeenCalledWith(key));
    await userEvent.click(screen.getByRole('button', { name: 'Use its evaluated views' }));
    await userEvent.click(await screen.findByRole('button', { name: /Evaluate rules with the probe \(2 views\)/ }));
    expect(api.startRulesEvaluation).toHaveBeenCalledTimes(1);
    expect(api.startRulesEvaluation).toHaveBeenCalledWith('pm_1', { view_ids: ['pmd_1', 'pmd_2'], calibration: true });
    expect(await screen.findByTestId('rules-status')).toHaveTextContent('running');
    api.cancelRulesEvaluation.mockResolvedValue({ evaluation_id: 'x', accepted: true });
    await userEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(api.cancelRulesEvaluation).toHaveBeenCalledTimes(1);
    const done = evaluation();
    api.getRulesEvaluation.mockResolvedValue(done);
    await userEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(await screen.findByTestId('rates-rules_or_probe')).toHaveTextContent('recall 66.7% (4/6)');
    expect(screen.getByTestId('rates-probe_alone')).toHaveTextContent('false positives 33.3% (2/6)');
    expect(screen.getByTestId('rules-view')).toHaveTextContent('unknown 1 (over the share)');
    expect(screen.getByTestId('union-bar')).toHaveTextContent('the union spends 1.0%, the rules 0.4%');
  });
});

describe('operator code rules on the rule-set panel (Feature 040 S4)', () => {
  it('shows each hash with its state and lets the operator revoke, with a reason, then re-validates', async () => {
    const key = slug();
    const approved = 'a'.repeat(63) + '1';
    const revoked = 'b'.repeat(63) + '2';
    const validation = {
      valid: true, rule_set_digest: 'd'.repeat(64), engine_digest: 'e'.repeat(64), engine_version: 1,
      cases: { passed: 0, unverified: {} }, match_expressions: ['m'], has_vetoes: false, pending_expressions: [],
      runner: 'available', code_rules: { [approved]: { state: 'approved', approval_row_id: 7 }, [revoked]: { state: 'revoked' } },
    };
    api.validateRuleSet.mockResolvedValue(validation);
    api.revokeCodeRule.mockResolvedValue({ code_sha256: approved, state: 'revoked' });
    render(<RuleSetPanel behaviourKey={key} />);
    await userEvent.click(screen.getByRole('button', { name: 'Validate rule set' }));
    const rows = await screen.findAllByTestId('code-rule');
    expect(rows.map((r) => r.textContent)).toEqual([
      expect.stringContaining('approved'), expect.stringContaining('revoked')]);
    const revoke = screen.getByRole('button', { name: 'Revoke' });
    expect((revoke as HTMLButtonElement).disabled).toBe(true); // no reason yet
    await userEvent.type(screen.getByLabelText('Reason for the decision'), 'no longer needed');
    await userEvent.click(revoke);
    await waitFor(() => expect(api.revokeCodeRule).toHaveBeenCalledTimes(1));
    expect(api.revokeCodeRule).toHaveBeenCalledWith(approved, 'no longer needed');
    await waitFor(() => expect(api.validateRuleSet).toHaveBeenCalledTimes(2));
    expect(api.validateRuleSet).toHaveBeenLastCalledWith({ behaviour_key: key });
    expect(screen.getByRole('button', { name: 'Re-approve' })).toBeTruthy();
  });

  it('re-approves a revoked hash with the reason given', async () => {
    const key = slug();
    const revoked = 'c'.repeat(64);
    api.validateRuleSet.mockResolvedValue({
      valid: true, rule_set_digest: 'd'.repeat(64), engine_digest: 'e'.repeat(64), engine_version: 1,
      runner: 'available', code_rules: { [revoked]: { state: 'revoked' } },
    });
    api.reapproveCodeRule.mockResolvedValue({ code_sha256: revoked, state: 'approved' });
    render(<RuleSetPanel behaviourKey={key} />);
    await userEvent.click(screen.getByRole('button', { name: 'Validate rule set' }));
    await screen.findByTestId('code-rule');
    await userEvent.type(screen.getByLabelText('Reason for the decision'), 'reviewed again');
    await userEvent.click(screen.getByRole('button', { name: 'Re-approve' }));
    await waitFor(() => expect(api.reapproveCodeRule).toHaveBeenCalledWith(revoked, 'reviewed again'));
    expect(api.reapproveCodeRule).toHaveBeenCalledTimes(1);
  });

  it('says a fired operator code rule has a count and no positions', async () => {
    const key = slug();
    api.evaluateRuleSignals.mockResolvedValue({
      rule_match: true, rule_veto: false, unknown: false, reason: null,
      fired: [{ signal_id: 'k_code', kind: 'code_rule', pending: false, spans: null, count: 2 }],
      expressions: {}, rule_set_digest: 'd'.repeat(64), engine_digest: 'e'.repeat(64), scope: 'all',
    });
    render(<RuleSetPanel behaviourKey={key} />);
    await userEvent.type(screen.getByLabelText('Text to try'), 'some text');
    await userEvent.click(screen.getByRole('button', { name: 'Try on this text' }));
    const verdict = await screen.findByTestId('rule-verdict');
    expect(verdict.textContent).toContain('count 2, no positions (operator code)');
  });
});

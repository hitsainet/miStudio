/**
 * ENH-006 D81: "Approve all code rules (N)" on the banner and the Approvals page. Only the network is mocked; the
 * banner and page are the real components reading the real store.
 *
 * Controls (0xcc/reviews/approve_all_settings_controls_2026-10-29.md): `BATCH_MIN = 1` → the one-request tests red;
 * send only the first id → the payload test red; drop the code-rule filter → the mixed-queue tests red; remove the
 * banner's `<ApproveAllCodeRules>` line → the banner test red.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { ApprovalRequest } from '../../types/featureGroups';

const listApprovals = vi.fn();
const listDecidedApprovals = vi.fn();
const approveBatch = vi.fn();
const getCodeRuleState = vi.fn();
vi.mock('../../api/featureGroups', () => ({
  listApprovals: (...a: unknown[]) => listApprovals(...a),
  listDecidedApprovals: (...a: unknown[]) => listDecidedApprovals(...a),
  getCodeRuleState: (...a: unknown[]) => getCodeRuleState(...a),
  approveBatch: (...a: unknown[]) => approveBatch(...a),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn() }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn().mockResolvedValue([]) } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn().mockResolvedValue({ gpus: [] }) }));
vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({ on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: false }),
}));

import { ApprovalsBanner } from '../layout/ApprovalsBanner';
import { ApprovalsPanel } from '../panels/ApprovalsPanel';
import { codeRuleLine } from './ApproveAllCodeRules';
import { resultText } from './approvalLabels';
import { APIError } from '../../api/client';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

const hex = (n: number) => Array.from({ length: n }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('');
const word = () => Array.from({ length: 8 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');

function codeRule(over: { passed?: boolean; statement?: string | null } = {}): ApprovalRequest {
  const fn = word();
  return {
    id: `cr-${hex(8)}`, tool_name: 'code_rule_approve', status: 'pending', reason: null, steering_task_id: null,
    created_at: new Date().toISOString(), resolved_at: null, requested_by: 'mcp',
    payload: {
      code_sha256: hex(64),
      document: {
        source: `def ${fn}(text, params):\n    return 0\n`, functions: [{ name: fn, output: { type: 'count' } }],
        ...(over.statement === null ? {} : { derived_from: { statement: over.statement ?? `rule ${word()}` } }),
      },
      facts: {
        case_results: [
          // Key order differs from `expect` on purpose: the server compares dicts, not strings.
          { id: 'c1', expect: { count: 2, kind: 'x' }, got: over.passed === false ? { count: 3 } : { kind: 'x', count: 2 }, same_twice: true },
        ],
      },
    },
  };
}

function steering(): ApprovalRequest {
  return {
    id: `st-${hex(8)}`, tool_name: 'steer_compare', status: 'pending', reason: null, steering_task_id: null,
    created_at: new Date().toISOString(), resolved_at: null, requested_by: 'mcp',
    payload: { sae_id: 's', prompt: 'hi', selected_features: [] },
  };
}

function setPending(rows: ApprovalRequest[]) {
  useApprovalsStore.setState({ pending: rows, loaded: true, error: null, history: [], historyLoaded: false });
  listApprovals.mockResolvedValue({ approvals: rows, total: rows.length });
}

beforeEach(() => {
  vi.clearAllMocks();
  getCodeRuleState.mockResolvedValue({ code_sha256: 'x', state: 'pending' });
  listDecidedApprovals.mockResolvedValue({ approvals: [], total: 0 });
  useUIStore.setState({ activePanel: 'datasets' });
});

describe('Approve all code rules', () => {
  it('is not offered for ONE waiting code rule, on the banner or the page', () => {
    setPending([codeRule()]);
    const { unmount } = render(<ApprovalsBanner />);
    expect(screen.queryByTestId('approve-all-code-rules-button')).toBeNull();
    unmount();
    useUIStore.setState({ activePanel: 'approvals' });
    render(<ApprovalsPanel />);
    expect(screen.queryByTestId('approve-all-code-rules-button')).toBeNull();
  });

  it('counts only code rules: one code rule beside two steering requests is still no button', () => {
    setPending([codeRule(), steering(), steering()]);
    render(<ApprovalsBanner />);
    expect(screen.queryByTestId('approve-all-code-rules-button')).toBeNull();
  });

  it('on the banner, names the action and counts every waiting code rule, beyond the 3 cards shown', () => {
    const rows = [codeRule(), codeRule(), codeRule(), codeRule(), steering()];
    setPending(rows);
    render(<ApprovalsBanner />);
    expect(screen.getByTestId('approve-all-code-rules-button')).toHaveTextContent('Approve all code rules (4)');
  });

  it('confirm lists each request, then sends ONE call with exactly the code-rule ids', async () => {
    const user = userEvent.setup();
    const a = codeRule();
    const b = codeRule({ passed: false, statement: null });
    const s = steering();
    setPending([a, s, b]);
    useUIStore.setState({ activePanel: 'approvals' });
    approveBatch.mockResolvedValue({
      batch_id: 'batch-1', kind: 'code_rules',
      results: [{ id: a.id, status: 'approved' }, { id: b.id, status: 'failed', error: 'digest mismatch' }],
    });
    render(<ApprovalsPanel />);
    await user.click(screen.getByTestId('approve-all-code-rules-button'));
    expect(approveBatch).not.toHaveBeenCalled(); // the button only opens the confirm

    const items = within(screen.getByTestId('approve-all-code-rules-list')).getAllByTestId('approve-all-code-rules-item');
    expect(items).toHaveLength(2);
    const la = codeRuleLine(a);
    const pa = a.payload as { code_sha256: string; document: { functions: { name: string }[]; derived_from: { statement: string } } };
    expect(items[0]).toHaveTextContent(pa.document.derived_from.statement);
    expect(items[0]).toHaveTextContent(pa.code_sha256.slice(0, 12));
    expect(items[0]).not.toHaveTextContent(pa.code_sha256.slice(0, 13));
    expect(items[0]).toHaveTextContent(pa.document.functions[0].name);
    expect(items[0]).toHaveTextContent('own cases passed at filing (1 of 1)');
    expect(items[0]).toHaveTextContent('asked by agent mcp');
    expect(la.casesPassed).toBe(true);
    expect(items[1]).toHaveTextContent('unnamed code rule');
    expect(items[1]).toHaveTextContent('own cases did NOT all pass at filing (0 of 1)');

    await user.click(screen.getByTestId('approve-all-code-rules-confirm-button'));
    await waitFor(() => expect(approveBatch).toHaveBeenCalledTimes(1));
    expect(approveBatch).toHaveBeenCalledWith('code_rules', [a.id, b.id]);
    const outcome = await screen.findByTestId('approve-all-code-rules-outcome');
    expect(outcome).toHaveTextContent('Approved 1 of 2 code rules together (batch batch-1)');
    expect(outcome).toHaveTextContent(`Not approved: ${b.id} — digest mismatch`);
    expect(listApprovals).toHaveBeenCalled(); // refreshed after
  });

  it('a refused batch says nothing was approved and keeps the list open', async () => {
    const user = userEvent.setup();
    setPending([codeRule(), codeRule()]);
    approveBatch.mockRejectedValue(new APIError(409, 'not pending code-rule requests: cr-x'));
    render(<ApprovalsBanner />);
    await user.click(screen.getByTestId('approve-all-code-rules-button'));
    await user.click(screen.getByTestId('approve-all-code-rules-confirm-button'));
    expect(await screen.findByTestId('approve-all-code-rules-problem')).toHaveTextContent('Nothing was approved');
    expect(screen.getByTestId('approve-all-code-rules-confirm')).toBeInTheDocument();
  });

  it('there is no reject-all', () => {
    setPending([codeRule(), codeRule()]);
    render(<ApprovalsBanner />);
    expect(screen.queryByText(/reject all/i)).toBeNull();
  });

  it('History names the batch each request was approved in', () => {
    const row = { ...codeRule(), status: 'approved' as const, batch_id: 'b-42' };
    expect(resultText(row)).toBe('approved together, batch b-42');
    expect(resultText({ ...row, batch_id: null })).toBe('—');
  });
});

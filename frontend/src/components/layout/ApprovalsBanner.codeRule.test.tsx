/**
 * Feature 040 S4 (FTDD Appendix C.6; FTASKS 25.10): an agent's request to run operator code is a card the operator
 * can READ IN FULL before approving — hash, functions, imports, the cases as run twice in the runner, and the whole
 * source, never cut. Random values throughout; only the network is mocked.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, within } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';

const listApprovals = vi.fn();
const getCodeRuleState = vi.fn();
vi.mock('../../api/featureGroups', () => ({
  listApprovals: (...args: unknown[]) => listApprovals(...args),
  listDecidedApprovals: vi.fn(),
  getCodeRuleState: (...args: unknown[]) => getCodeRuleState(...args),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn() }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn() } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn() }));
vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({ on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: false }),
}));

import { ApprovalsBanner } from './ApprovalsBanner';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

const word = () => Array.from({ length: 8 }, () => 'abcdefghijklmnopqrstuvwxyz'[Math.floor(Math.random() * 26)]).join('');

beforeEach(() => {
  listApprovals.mockReset();
  getCodeRuleState.mockReset();
  useUIStore.setState({ activePanel: 'datasets' });
});

describe('the operator-code approval card', () => {
  it('shows the hash, functions, imports, both case runs and the FULL source', async () => {
    const fn = word();
    const sha = Array.from({ length: 64 }, () => '0123456789abcdef'[Math.floor(Math.random() * 16)]).join('');
    // A long source: the card must never cut it.
    const source = `def ${fn}(text, params):\n` + Array.from({ length: 80 }, (_, i) => `    x${i} = ${i}  # ${word()}\n`).join('') + '    return 0\n';
    const request: ApprovalRequest = {
      id: 'req-1', tool_name: 'code_rule_approve', status: 'pending', reason: null, steering_task_id: null,
      created_at: null, resolved_at: null,
      payload: {
        code_sha256: sha,
        document: { source, functions: [{ name: fn, output: { type: 'count' } }], derived_from: { statement: word() } },
        facts: {
          inspect: { functions: [fn], imports: ['re'] },
          case_results: [{ id: 'c1', expect: { count: 2 }, got: { count: 2 }, same_twice: true }],
          midataworks_state: null, requested_by: 'agent:mcp',
        },
      },
    };
    getCodeRuleState.mockResolvedValue({ code_sha256: sha, state: 'pending', revoked_row_id: 4 });
    useApprovalsStore.setState({ pending: [request], loaded: true, error: null });
    render(<ApprovalsBanner />);
    const card = await waitFor(() => screen.getByTestId('code-rule-approval'));
    expect(within(card).getByText(sha)).toBeTruthy();
    expect(card.textContent).toContain(`Functions: ${fn}`);
    expect(card.textContent).toContain('Imports: re');
    expect(card.textContent).toContain('asked by agent:mcp');
    expect(within(card).getByTestId('code-rule-cases').textContent).toContain('same on two fresh instances');
    expect(within(card).getByTestId('code-rule-source').textContent).toBe(source);
    expect(card.textContent).toContain('information only');
    // D4: the hash's CURRENT state, read when the card is shown; the facts are labelled a snapshot.
    await waitFor(() => expect(within(card).getByTestId('code-rule-state').textContent).toContain('revoked earlier'));
    expect(getCodeRuleState).toHaveBeenCalledWith(sha);
    expect(card.textContent).toContain('Snapshot taken when the agent asked');
    expect(card.textContent).not.toMatch(/restor/i);
  });
});

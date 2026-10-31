/**
 * Hostile text renders as text (ENH-008 §2.7, S10). Markup in EVERY field a request carries — the tool name,
 * who asked, the summary/facts keys and values, the payload, the reason, the steering task, the code-rule source,
 * functions, imports and case results, and the id — creates no `img` or `script` element anywhere on the banner,
 * the page or History. A render test is the authority here; a source grep for dangerouslySetInnerHTML is not.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';

const H = '<img src=x onerror=alert(1)><script>alert(2)</script>';
vi.mock('../../api/featureGroups', () => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(() => new Promise(() => {})),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
  getCodeRuleState: vi.fn(() => Promise.resolve({ code_sha256: 'a'.repeat(64), state: H })),
  HISTORY_PAGE: 200,
}));
vi.mock('../../api/models', () => ({ getModel: vi.fn(() => Promise.resolve({ id: H, name: H })) }));
vi.mock('../../api/probeMonitors', () => ({ probeMonitorsApi: { listDatasets: vi.fn(() => Promise.resolve([])) } }));
vi.mock('../../api/system', () => ({ getGPUList: vi.fn(() => Promise.resolve({ gpus: [] })) }));

import { ApprovalsBanner } from '../layout/ApprovalsBanner';
import { ApprovalsPanel } from '../panels/ApprovalsPanel';
import { ApprovalHistory } from './ApprovalHistory';
import { useApprovalsStore } from '../../stores/approvalsStore';
import { useUIStore } from '../../stores/uiStore';

function hostile(tool: string, extra: Record<string, unknown> = {}): ApprovalRequest {
  return {
    id: `${tool}-${H}`, tool_name: tool, status: 'pending', reason: H, steering_task_id: H,
    created_at: H, resolved_at: H, requested_by: H, decided_by: H, expires_at: H,
    payload: { [H]: H, nested: { [H]: [H] }, prompt: H, model_id: H, train_dataset_id: H, eval_dataset_ids: [H],
      gpu: H, behaviour: H, behaviour_label: H, config: { [H]: H }, ...extra },
  };
}

const CODE = hostile('code_rule_approve', {
  code_sha256: 'a'.repeat(64),
  document: { source: H, functions: [{ name: H }], derived_from: { statement: H } },
  facts: { inspect: { imports: [H] }, case_results: [{ id: H, expect: { [H]: H }, got: { [H]: H }, same_twice: false }],
           midataworks_state: H, requested_by: H },
});
const ALL = [hostile(H), hostile('steer_compare', { selected_features: [H] }), hostile('start_probe_training_run'), CODE];

function noMarkup(container: HTMLElement) {
  expect(container.querySelectorAll('img')).toHaveLength(0);
  expect(container.querySelectorAll('script')).toHaveLength(0);
  expect(container.textContent).toContain(H);
}

beforeEach(() => {
  useUIStore.setState({ activePanel: 'datasets' });
  useApprovalsStore.setState({ pending: ALL, loaded: true, error: H });
});

describe('hostile text in every field renders as text', () => {
  it('on the banner', async () => {
    const { container } = render(<ApprovalsBanner />);
    await screen.findAllByTestId('approval-request');
    // the banner shows three cards; the code-rule card is fourth, so it is checked on the page below
    noMarkup(container);
  });

  it('on the page, including the code-rule source and its live state', async () => {
    const { container } = render(<ApprovalsPanel />);
    await screen.findByTestId('code-rule-approval');
    expect(screen.getByTestId('code-rule-source').textContent).toBe(H);
    noMarkup(container);
  });

  it('in History', () => {
    const decided = ALL.map((r) => ({ ...r, status: 'denied' as const }));
    const { container } = render(<ApprovalHistory rows={decided} />);
    noMarkup(container);
  });
});

/**
 * The agent-approvals banner renders every approvable tool (Feature 034, FR-20).
 *
 * A model load evicts the model miLLM serves, so its line must name the model
 * AND the eviction. A tool the banner does not know must still be readable — a
 * request the operator cannot read is one they approve blind.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import type { ApprovalRequest } from '../../types/featureGroups';

const listApprovals = vi.fn();
vi.mock('../../api/featureGroups', () => ({
  listApprovals: (...args: unknown[]) => listApprovals(...args),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
}));
vi.mock('../../contexts/WebSocketContext', () => ({
  useWebSocketContext: () => ({
    on: vi.fn(), off: vi.fn(), subscribe: vi.fn(), unsubscribe: vi.fn(), isConnected: false,
  }),
}));

import { ApprovalsBanner, renderSummary } from './ApprovalsBanner';

function request(tool_name: string, payload: Record<string, unknown>): ApprovalRequest {
  return {
    id: `id-${tool_name}`, tool_name, payload, status: 'pending', reason: null,
    steering_task_id: null, created_at: null, resolved_at: null,
  };
}

beforeEach(() => listApprovals.mockReset());

describe('ApprovalsBanner', () => {
  it('renders a model-load request with the model and the eviction', async () => {
    listApprovals.mockResolvedValue({
      approvals: [request('millm_load_model', { model_id: 7 })], total: 1,
    });
    render(<ApprovalsBanner />);
    expect(
      await screen.findByText('Load miLLM model #7 — evicts the resident model'),
    ).toBeInTheDocument();
    expect(screen.getByText('1 agent request awaiting approval')).toBeInTheDocument();
    expect(listApprovals).toHaveBeenCalledWith('pending');
  });

  it('keeps the steering line', async () => {
    listApprovals.mockResolvedValue({
      approvals: [
        request('steer_compare', { prompt: 'hello', selected_features: [1, 2] }),
        request('millm_load_model', { model_id: 3 }),
      ],
      total: 2,
    });
    render(<ApprovalsBanner />);
    expect(await screen.findByText('2 features · "hello"')).toBeInTheDocument();
    expect(screen.getByText('2 agent requests awaiting approval')).toBeInTheDocument();
  });

  it('renders NOTHING when the queue is empty — no wrapper, no padding', async () => {
    listApprovals.mockResolvedValue({ approvals: [], total: 0 });
    const { container } = render(<ApprovalsBanner />);
    await waitFor(() => expect(listApprovals).toHaveBeenCalled());
    expect(container).toBeEmptyDOMElement();
  });

  it('shows an UNKNOWN tool by name and payload rather than blank', () => {
    expect(renderSummary(request('future_tool', { a: 1 }))).toBe('{"a":1}');
  });
});

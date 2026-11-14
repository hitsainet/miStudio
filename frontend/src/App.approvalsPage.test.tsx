/**
 * REACHABILITY of the Approvals page (ENH-008 S8): the panel is in the registry, in the menu (with the badge) and in
 * the App switch; the shell runs the one poll that feeds the badge; the banner is on other panels and not on this
 * one. Each wiring line removed turns a test here red:
 *   * `'approvals'` out of PANEL_IDS     → "is a registered panel" and the restore test;
 *   * the navItems entry                → "the menu entry opens the page";
 *   * the App.tsx switch line           → "the menu entry opens the page" (no panel);
 *   * `useApprovalsLive()` in App.tsx   → the badge test (nothing reads the queue).
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import type { ApprovalRequest } from './types/featureGroups';

const api = vi.hoisted(() => ({
  listApprovals: vi.fn(),
  listDecidedApprovals: vi.fn(),
  approveRequest: vi.fn(),
  denyRequest: vi.fn(),
}));
vi.mock('./api/featureGroups', async (orig) => ({
  ...(await orig<typeof import('./api/featureGroups')>()),
  ...api,
  getCodeRuleState: vi.fn(() => new Promise(() => {})),
}));
vi.mock('./components/panels/DatasetsPanel', () => ({ DatasetsPanel: () => <div>datasets-panel</div> }));
vi.mock('./components/panels/ModelsPanel', () => ({ ModelsPanel: () => <div /> }));
vi.mock('./components/panels/TemplatesPanel', () => ({ TemplatesPanel: () => <div /> }));
vi.mock('./components/panels/TrainingPanel', () => ({ TrainingPanel: () => <div /> }));
vi.mock('./components/panels/ExtractionsPanel', () => ({ ExtractionsPanel: () => <div /> }));
vi.mock('./components/panels/LabelingPanel', () => ({ LabelingPanel: () => <div /> }));
vi.mock('./components/panels/CircuitsPanel', () => ({ CircuitsPanel: () => <div /> }));
vi.mock('./components/panels/ProbeMonitorsPanel', () => ({ ProbeMonitorsPanel: () => <div /> }));
vi.mock('./components/panels/JLensPanel', () => ({ JLensPanel: () => <div /> }));
vi.mock('./components/panels/SAEsPanel', () => ({ SAEsPanel: () => <div /> }));
vi.mock('./components/panels/SteeringPanel', () => ({ SteeringPanel: () => <div /> }));
vi.mock('./components/panels/FeatureGroupsPanel', () => ({ FeatureGroupsPanel: () => <div /> }));
vi.mock('./components/panels/SettingsPanel', () => ({ SettingsPanel: () => <div /> }));
vi.mock('./components/SystemMonitor/SystemMonitor', () => ({ SystemMonitor: () => <div /> }));
vi.mock('./components/layout/Header', () => ({ Header: () => <div /> }));
vi.mock('./hooks/useDatasetProgress', () => ({ useGlobalDatasetProgress: () => {} }));

import App from './App';
import { isActivePanel, PANEL_IDS } from './config/panels';
import { navItems } from './components/layout/Sidebar';
import { useApprovalsStore } from './stores/approvalsStore';

function waiting(id: string): ApprovalRequest {
  return {
    id, tool_name: 'evaluate_probe_monitor', payload: { probe_id: id }, status: 'pending', reason: null,
    steering_task_id: null, created_at: '2026-10-28T10:00:00Z', resolved_at: null, requested_by: 'mcp',
    expires_at: '2099-01-01T00:00:00Z',
  };
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  Object.values(api).forEach((f) => f.mockReset());
  api.listApprovals.mockResolvedValue({ approvals: [waiting('a'), waiting('b')], total: 2 });
  api.listDecidedApprovals.mockResolvedValue({ approvals: [], total: 0 });
  useApprovalsStore.setState({ pending: [], loaded: false, error: null, history: [], historyLoaded: false });
});

describe('the Approvals page is reachable', () => {
  it('is a registered panel, in the menu before Help', () => {
    expect(PANEL_IDS).toContain('approvals');
    expect(isActivePanel('approvals')).toBe(true);
    const ids = navItems.map((n) => n.id);
    expect(ids).toContain('approvals');
    expect(ids.indexOf('approvals')).toBe(ids.indexOf('help') - 1);
    expect(navItems.find((n) => n.id === 'approvals')?.label).toBe('Approvals');
  });

  it('the shell polls the queue and the menu badge shows the waiting count', async () => {
    localStorage.setItem('activePanel', 'datasets');
    render(<App />);
    const badge = await screen.findByTestId('nav-badge-approvals');
    expect(badge).toHaveTextContent('2');
    expect(badge).toHaveAttribute('aria-label', '2 waiting');
    expect(badge).toHaveAttribute('title', '2 waiting');
    expect(api.listApprovals).toHaveBeenCalledWith('pending');
    // The banner is on the other panels.
    expect(screen.getByTestId('approvals-banner')).toBeInTheDocument();
  });

  it('the badge is hidden at 0', async () => {
    api.listApprovals.mockResolvedValue({ approvals: [], total: 0 });
    render(<App />);
    await waitFor(() => expect(api.listApprovals).toHaveBeenCalled());
    expect(screen.queryByTestId('nav-badge-approvals')).toBeNull();
  });

  it('the menu entry opens the page, where the banner is not drawn', async () => {
    localStorage.setItem('activePanel', 'datasets');
    render(<App />);
    await screen.findByTestId('approvals-banner');
    const nav = screen.getAllByRole('button').find((b) => within(b).queryByText('Approvals'));
    expect(nav).toBeTruthy();
    fireEvent.click(nav!);
    const panel = await screen.findByTestId('approvals-panel');
    expect(within(panel).getByText('Waiting for you (2)')).toBeInTheDocument();
    expect(screen.queryByTestId('approvals-banner')).toBeNull();
    expect(within(panel).getAllByTestId('approval-request')).toHaveLength(2);
  });

  it('a saved Approvals panel is restored on reload', async () => {
    localStorage.setItem('activePanel', 'approvals');
    render(<App />);
    expect(await screen.findByTestId('approvals-panel')).toBeInTheDocument();
  });
});

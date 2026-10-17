/**
 * REACHABILITY: the agent-approvals banner is mounted in the APP SHELL, above
 * every panel (Feature 034, T-97), and nowhere else.
 *
 * A model load through MCP evicts the model miLLM serves. If the banner lived
 * only in the Steering panel, an operator on any other panel would never see
 * the request — the agent would wait on an approval nobody can find.
 *
 * MUTATION CONTROL (recorded 2026-10-16, task 2.12): deleting the
 * `<ApprovalsBanner />` mount from App.tsx -> both "on panel" tests fail.
 */
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
// Vite `?raw`: the source text, without Node globals the test type-check cannot see.
import steeringPanelSource from './components/panels/SteeringPanel.tsx?raw';
import App from './App';

vi.mock('./components/layout/ApprovalsBanner', () => ({
  ApprovalsBanner: () => <div data-testid="approvals-banner-stub" />,
}));
vi.mock('./components/panels/DatasetsPanel', () => ({ DatasetsPanel: () => <div>datasets-panel</div> }));
vi.mock('./components/panels/ModelsPanel', () => ({ ModelsPanel: () => <div>models-panel</div> }));
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

beforeEach(() => localStorage.clear());

describe('the approvals banner lives in the app shell', () => {
  it('is on screen on the Datasets panel', () => {
    localStorage.setItem('activePanel', 'datasets');
    render(<App />);
    expect(screen.getByText('datasets-panel')).toBeInTheDocument();
    expect(screen.getAllByTestId('approvals-banner-stub')).toHaveLength(1);
  });

  it('is on screen on the Models panel too', () => {
    localStorage.setItem('activePanel', 'models');
    render(<App />);
    expect(screen.getByText('models-panel')).toBeInTheDocument();
    expect(screen.getAllByTestId('approvals-banner-stub')).toHaveLength(1);
  });

  it('is no longer mounted by the Steering panel (one mount, one subscription)', () => {
    expect(steeringPanelSource).toContain('Feature Steering');
    expect(steeringPanelSource).not.toMatch(/ApprovalsBanner/);
  });
});

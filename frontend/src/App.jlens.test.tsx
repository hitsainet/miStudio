/**
 * REACHABILITY: the J-Lens panel must be reachable from the running app.
 *
 * The nav test asserts the tab is in the right place and the registry test
 * asserts the id routes — neither fails if the render line
 * `{activePanel === 'jlens' && <JLensPanel />}` is deleted from App.tsx. That
 * is the exact shape of this repo's worst shipped defect: 16 MCP tools fully
 * implemented, unit-tested and documented while never registered, with every
 * test passing by importing the module directly.
 *
 * So this test drives the real App: click the nav entry, assert the panel is
 * on screen; and restore from localStorage, assert the same.
 *
 * MUTATION CONTROLS:
 *   * delete the `activePanel === 'jlens'` render line from App.tsx -> both fail
 *   * remove 'jlens' from PANEL_IDS                                 -> restore test fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';

// Every other panel is stubbed: this test is about the ROUTE, and mounting the
// real panels would drag their fetches and timers in with them.
vi.mock('./components/panels/DatasetsPanel', () => ({ DatasetsPanel: () => <div /> }));
vi.mock('./components/panels/ModelsPanel', () => ({ ModelsPanel: () => <div /> }));
vi.mock('./components/panels/TemplatesPanel', () => ({ TemplatesPanel: () => <div /> }));
vi.mock('./components/panels/TrainingPanel', () => ({ TrainingPanel: () => <div /> }));
vi.mock('./components/panels/ExtractionsPanel', () => ({ ExtractionsPanel: () => <div /> }));
vi.mock('./components/panels/LabelingPanel', () => ({ LabelingPanel: () => <div /> }));
vi.mock('./components/panels/CircuitsPanel', () => ({ CircuitsPanel: () => <div /> }));
vi.mock('./components/panels/SAEsPanel', () => ({ SAEsPanel: () => <div /> }));
vi.mock('./components/panels/SteeringPanel', () => ({ SteeringPanel: () => <div /> }));
vi.mock('./components/panels/FeatureGroupsPanel', () => ({
  FeatureGroupsPanel: () => <div />,
}));
vi.mock('./components/panels/SettingsPanel', () => ({ SettingsPanel: () => <div /> }));
vi.mock('./components/SystemMonitor/SystemMonitor', () => ({ SystemMonitor: () => <div /> }));
vi.mock('./components/layout/Header', () => ({ Header: () => <div /> }));
vi.mock('./hooks/useDatasetProgress', () => ({ useGlobalDatasetProgress: () => {} }));

// The panel under test is NOT stubbed — but its data sources are, so the test
// exercises the route rather than the network.
// STABLE identities, defined once outside the hook. A mock that builds a fresh
// object per call hands back a new `fetchModels` on every render, which turns
// any effect keyed on it into an infinite loop — and that is a defect in the
// mock, not in the component.
const MODELS_STATE = { models: [], fetchModels: () => {} };
vi.mock('./stores/modelsStore', () => ({
  useModelsStore: Object.assign(
    (selector?: (s: unknown) => unknown) =>
      selector ? selector(MODELS_STATE) : MODELS_STATE,
    { getState: () => MODELS_STATE }
  ),
}));
vi.mock('./api/jlens', () => ({
  jlensApi: { readout: vi.fn(), listArtifacts: vi.fn().mockResolvedValue([]) },
}));

beforeEach(() => {
  localStorage.clear();
});

describe('J-Lens is reachable from the running app', () => {
  it('renders the panel when the nav entry is clicked', async () => {
    const user = userEvent.setup();
    render(<App />);

    await user.click(screen.getByRole('button', { name: /J-Lens/ }));

    expect(screen.getByRole('heading', { name: /J-Lens Readout/ })).toBeInTheDocument();
  });

  it('restores the panel from localStorage on reload', () => {
    localStorage.setItem('activePanel', 'jlens');
    render(<App />);

    expect(screen.getByRole('heading', { name: /J-Lens Readout/ })).toBeInTheDocument();
  });
});

/**
 * REACHABILITY: the Help screen is reachable from the running app, by every route a user has
 * (ENH-004). Drives the real <App/> with the real Sidebar, Header, UI store and Help panel; only
 * the other panels and their data sources are stubbed.
 *
 *   * the "Help" nav entry, which must be the LAST main-nav item;
 *   * the header's contextual Help link, which opens the topic for the screen you are on;
 *   * a `#/help/<topic>` address, on arrival and while the app is open;
 *   * leaving Help removes the `#/help...` address and never touches another fragment.
 *
 * MUTATION CONTROLS (2026-10-20, each restored from saved bytes; see the ENH-004 hand-back):
 * delete the Help navItems entry; delete the `activePanel === 'help'` render line in App.tsx;
 * remove 'help' from PANEL_IDS; delete the HelpLink render in Header.tsx; make openHelp skip
 * setActivePanel; drop the hash parse in initialPanel; drop the hashchange listener.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { act, render, screen, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import App from './App';
import { navItems, bottomNavItems } from './components/layout/Sidebar';
import { HELP_FOR_PANEL, HELP_TOPIC_IDS } from './config/help';
import { MANUAL_BASE_URL } from './config/manual';
import { PANEL_IDS } from './config/panels';
import { useHelpTopic } from './components/help/helpTopicStore';
import pkg from '../package.json';

vi.mock('./components/panels/DatasetsPanel', () => ({
  DatasetsPanel: () => <h1>Datasets stub</h1>,
}));
vi.mock('./components/panels/ModelsPanel', () => ({ ModelsPanel: () => <div /> }));
vi.mock('./components/panels/TemplatesPanel', () => ({ TemplatesPanel: () => <div /> }));
vi.mock('./components/panels/TrainingPanel', () => ({
  TrainingPanel: () => <h1>Training stub</h1>,
}));
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
vi.mock('./components/SystemMonitor/CompactGPUStatus', () => ({ CompactGPUStatus: () => null }));
vi.mock('./components/layout/ApprovalsBanner', () => ({ ApprovalsBanner: () => null }));
vi.mock('./hooks/useDatasetProgress', () => ({ useGlobalDatasetProgress: () => {} }));

function setHash(hash: string) {
  window.history.replaceState(null, '', `/${hash}`);
}

beforeEach(() => {
  localStorage.clear();
  setHash('');
  useHelpTopic.setState({ topic: null });
});

describe('Help in the navigation', () => {
  it('is the last main-nav item, and the foot is still Settings alone', () => {
    expect(navItems[navItems.length - 1]).toMatchObject({ id: 'help', label: 'Help' });
    expect(bottomNavItems.map(i => i.id)).toEqual(['settings']);
  });

  it('opens the Help screen when the nav entry is clicked, and the address says so', async () => {
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByText('Datasets stub')).toBeInTheDocument();

    await user.click(
      within(screen.getByRole('complementary')).getByRole('button', { name: 'Help' })
    );

    expect(screen.getByTestId('screen-help')).toBeInTheDocument();
    expect(screen.getByRole('heading', { level: 1, name: 'Help' })).toBeInTheDocument();
    expect(window.location.hash).toBe('#/help');
  });

  it('the foot keeps Settings, the book-icon manual link from config, and the package version', () => {
    render(<App />);
    expect(screen.getByTestId('sidebar-manual')).toHaveAttribute('href', MANUAL_BASE_URL);
    expect(screen.getByTestId('sidebar-version')).toHaveTextContent(`v${pkg.version}`);
  });
});

describe('the nav entry opens Help at the top', () => {
  it('clears a topic an earlier Help link asked for (mutation M23 survived without this)', async () => {
    localStorage.setItem('activePanel', 'training');
    const user = userEvent.setup();
    render(<App />);
    await user.click(within(screen.getByRole('banner')).getByTestId('help-link-training'));
    expect(document.getElementById('help-training')).toHaveAttribute('data-current', 'true');

    const nav = within(screen.getByRole('complementary'));
    await user.click(nav.getByRole('button', { name: 'Training' }));
    await user.click(nav.getByRole('button', { name: 'Help' }));

    expect(screen.getByTestId('screen-help')).toBeInTheDocument();
    expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(0);
    expect(window.location.hash).toBe('#/help');
  });
});

describe('the header Help link follows the screen', () => {
  it('maps every panel to a real topic', () => {
    for (const id of PANEL_IDS) expect(HELP_TOPIC_IDS).toContain(HELP_FOR_PANEL[id]);
  });

  it('on Training, opens Help at the training topic and marks it current', async () => {
    localStorage.setItem('activePanel', 'training');
    const user = userEvent.setup();
    render(<App />);
    expect(screen.getByText('Training stub')).toBeInTheDocument();

    const header = screen.getByRole('banner');
    await user.click(within(header).getByTestId('help-link-training'));

    expect(screen.getByTestId('screen-help')).toBeInTheDocument();
    expect(document.getElementById('help-training')).toHaveAttribute('data-current', 'true');
    expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(1);
    expect(window.location.hash).toBe('#/help/training');
    // The header link is for other screens; on Help itself there is none.
    expect(within(header).queryByRole('button', { name: 'Help' })).toBeNull();
  });

  it('carries exactly one manual link beside it, to the topic page', () => {
    localStorage.setItem('activePanel', 'training');
    render(<App />);
    const header = screen.getByRole('banner');
    expect(within(header).getAllByRole('link')).toHaveLength(1);
    expect(within(header).getByTestId('manual-link-training').getAttribute('href')).toMatch(
      /core-workflow\/sae-training#configuring-a-training-job$/
    );
  });
});

describe('linkable topics: #/help/<topic>', () => {
  it('an address on arrival opens Help at that topic', () => {
    setHash('#/help/circuits');
    render(<App />);
    expect(screen.getByTestId('screen-help')).toBeInTheDocument();
    expect(document.getElementById('help-circuits')).toHaveAttribute('data-current', 'true');
  });

  it('an unknown topic still opens Help, with nothing marked', () => {
    setHash('#/help/no-such-topic');
    render(<App />);
    expect(screen.getByTestId('screen-help')).toBeInTheDocument();
    expect(document.querySelectorAll('[data-current="true"]')).toHaveLength(0);
  });

  it('an address followed while the app is open switches to Help', () => {
    render(<App />);
    expect(screen.getByText('Datasets stub')).toBeInTheDocument();
    act(() => {
      setHash('#/help/probes');
      window.dispatchEvent(new HashChangeEvent('hashchange'));
    });
    expect(screen.getByTestId('screen-help')).toBeInTheDocument();
    expect(document.getElementById('help-probes')).toHaveAttribute('data-current', 'true');
  });

  it('a topic chip is a link to its own address and marks the topic', async () => {
    setHash('#/help');
    const user = userEvent.setup();
    render(<App />);
    await user.click(screen.getByTestId('help-chip-steering'));
    expect(window.location.hash).toBe('#/help/steering');
    expect(document.getElementById('help-steering')).toHaveAttribute('data-current', 'true');
  });

  it('leaving Help removes the help address', async () => {
    setHash('#/help/models');
    const user = userEvent.setup();
    render(<App />);
    await user.click(
      within(screen.getByRole('complementary')).getByRole('button', { name: 'Training' })
    );
    expect(screen.getByText('Training stub')).toBeInTheDocument();
    expect(window.location.hash).toBe('');
  });

  it('never touches another fragment (the J-Lens permalink)', () => {
    setHash('#jlens?repo=google%2Fgemma-2-2b&p=hi');
    render(<App />);
    expect(window.location.hash).toBe('#jlens?repo=google%2Fgemma-2-2b&p=hi');
  });
});

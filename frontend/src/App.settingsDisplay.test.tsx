/**
 * Settings -> Display provides display settings, wired to the running app.
 *
 * The tab said "Additional display settings will be added here in a future
 * update" and offered none. It now sets the theme (shared with the header's
 * sun/moon toggle, so App owns it) and the sidebar width (uiStore). This drives
 * the real App, so deleting the `theme={theme} onThemeChange={setTheme}` props
 * from App's SettingsPanel line leaves the Theme controls unrendered — red.
 *
 * MUTATION CONTROLS (2026-10-18, each restored by sha256):
 *   D1  drop `theme={theme} onThemeChange={setTheme}` from App            -> 1 red
 *   D2  Light button calls onThemeChange('dark')                          -> 1 red
 *   D3  Collapsed button calls setSidebarCollapsed(false)                 -> 1 red
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import { useSettingsStore } from './stores/settingsStore';
import { useUIStore } from './stores/uiStore';

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
vi.mock('./components/SystemMonitor/SystemMonitor', () => ({ SystemMonitor: () => <div /> }));
vi.mock('./components/layout/Header', () => ({ Header: () => <div /> }));
vi.mock('./components/layout/ApprovalsBanner', () => ({ ApprovalsBanner: () => null }));
vi.mock('./hooks/useDatasetProgress', () => ({ useGlobalDatasetProgress: () => {} }));
vi.mock('./components/panels/JLensPanel', () => ({ JLensPanel: () => <div /> }));

beforeEach(() => {
  window.scrollTo = vi.fn() as never;
  localStorage.clear();
  localStorage.setItem('activePanel', 'settings');
  useSettingsStore.setState({ fetchAll: vi.fn(), isLoading: false, settings: [] } as never);
  useUIStore.setState({ sidebar: { collapsed: false } });
});

async function openDisplay() {
  const user = userEvent.setup();
  render(<App />);
  await user.click(screen.getByRole('button', { name: 'Display' }));
  return user;
}

describe('Settings -> Display', () => {
  it('no longer promises settings for a future update', async () => {
    await openDisplay();
    expect(screen.queryByText(/future update/i)).not.toBeInTheDocument();
  });

  it('Theme switches the running app between dark and light', async () => {
    const user = await openDisplay();
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    await user.click(screen.getByRole('button', { name: 'Light' }));
    expect(document.documentElement.classList.contains('dark')).toBe(false);
    expect(localStorage.getItem('theme')).toBe('light');
    expect(screen.getByRole('button', { name: 'Light' })).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: 'Dark' }));
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });

  it('Sidebar collapses and expands the navigation', async () => {
    const user = await openDisplay();
    await user.click(screen.getByRole('button', { name: /Collapsed/ }));
    expect(useUIStore.getState().sidebar.collapsed).toBe(true);
    await user.click(screen.getByRole('button', { name: 'Expanded' }));
    expect(useUIStore.getState().sidebar.collapsed).toBe(false);
  });
});

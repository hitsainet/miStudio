/**
 * Nav placement and panel-registry integrity.
 *
 * Nav ORDER is array position — there is no sort key — so a test that merely
 * asserted "J-Lens is in the nav" would pass with the tab in the wrong place.
 * This asserts the RENDERED order.
 *
 * The registry assertions derive both sides from the authorities (navItems and
 * PANEL_IDS) rather than from a copy: a nav entry outside PANEL_IDS is a panel
 * that vanishes on reload, and a PANEL_ID absent from the nav is a panel no
 * user can reach.
 *
 * MUTATION CONTROLS:
 *   * move the jlens navItems entry after 'steering'    -> ordering test fails
 *   * remove 'jlens' from PANEL_IDS                      -> registry test fails
 *   * remove the jlens navItems entry                    -> both fail
 */

import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { Sidebar, navItems, bottomNavItems } from './Sidebar';
import { PANEL_IDS } from '../../config/panels';

describe('J-Lens nav placement', () => {
  it('renders J-Lens immediately before Steering', () => {
    render(<Sidebar activePanel="datasets" onPanelChange={() => {}} />);

    const labels = screen
      .getAllByRole('button')
      .map((b) => b.textContent?.trim())
      .filter((t): t is string => Boolean(t));

    const jlens = labels.indexOf('J-Lens');
    const steering = labels.indexOf('Steering');

    expect(jlens).toBeGreaterThanOrEqual(0);
    expect(steering).toBeGreaterThanOrEqual(0);
    expect(jlens).toBe(steering - 1);
  });

  it('places jlens between circuits and steering in the nav registry', () => {
    const ids = navItems.map((i) => i.id);
    expect(ids.indexOf('jlens')).toBe(ids.indexOf('circuits') + 1);
    expect(ids.indexOf('jlens')).toBe(ids.indexOf('steering') - 1);
  });
});

describe('panel registry', () => {
  it('routes every nav entry — a nav id outside PANEL_IDS is dropped on reload', () => {
    const navIds = [...navItems, ...bottomNavItems].map((i) => i.id);
    for (const id of navIds) {
      expect(PANEL_IDS).toContain(id);
    }
  });

  it('exposes every panel in the nav — an unlisted panel is unreachable', () => {
    const navIds = new Set([...navItems, ...bottomNavItems].map((i) => i.id));
    for (const id of PANEL_IDS) {
      expect(navIds.has(id)).toBe(true);
    }
  });
});

/**
 * The panel registry — one authority for which panels exist.
 *
 * This list used to be written out THREE times: the `ActivePanel` union in
 * Sidebar.tsx, the same union again in App.tsx, and a third hardcoded
 * `validPanels` array guarding the localStorage restore. Adding a panel and
 * missing the third copy produced a panel that worked when clicked and silently
 * vanished on reload, with no type error anywhere — because the two unions
 * agreed and the array was plain strings.
 *
 * Nav ORDER is not here: it is the array order of `navItems` in Sidebar.tsx,
 * which is the only thing that decides it.
 */

export const PANEL_IDS = [
  'models',
  'datasets',
  'training',
  'extractions',
  'labeling',
  'feature-groups',
  'circuits',
  'jlens',
  'templates',
  'saes',
  'steering',
  'system',
  'settings',
] as const;

export type ActivePanel = (typeof PANEL_IDS)[number];

/** Narrow an untrusted string (localStorage, a URL fragment) to a real panel. */
export function isActivePanel(value: string | null | undefined): value is ActivePanel {
  return value != null && (PANEL_IDS as readonly string[]).includes(value);
}

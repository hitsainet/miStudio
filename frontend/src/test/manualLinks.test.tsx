/**
 * Every manual link the app shows points at a page and heading that exist (ENH-004).
 *
 * The Help screen, every per-screen Help link and the sidebar's book icon point into the public
 * manual (manual/, published to GitHub Pages from the mirror). These checks read the manual's own
 * sources, so a renamed page, a renamed heading or a page dropped from the sidebar fails here, not
 * in a user's browser as a 404 or a page top.
 *
 * HOW IT READS THE MANUAL. Through Vite's glob with `?raw`, not node:fs: test files here have no
 * Node types, and fs/path/__dirname are type errors against the down-only ratchet in
 * typeCheckRatchet.test.ts (the lesson noUnfinishedWorkCopy.test.ts records). The glob is
 * relative to this file, so `../../../manual/` is the repository's manual/. Two fail-closed
 * guards: the glob must find the manual (a moved directory would otherwise match nothing and
 * check nothing), and the slug rule is pinned against ids Docusaurus 3.10 actually generated.
 *
 * Origin: miDataworks frontend/src/test/manualLinks.test.tsx @ 32cf406 (ENH-004), ported from
 * node:fs to import.meta.glob.
 *
 * MUTATION CONTROLS (2026-10-20, each restored from saved bytes): see the ENH-004 hand-back.
 */
import { describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';

import { HelpLink } from '../components/help/HelpLink';
import { HelpPanel } from '../components/help/HelpPanel';
import { HELP_TOPICS } from '../config/help';
import {
  DEFAULT_MANUAL_BASE_URL,
  MANUAL_BASE_URL,
  manualBaseUrl,
  manualUrl,
} from '../config/manual';

const DOCS = import.meta.glob('../../../manual/docs/**/*.{md,mdx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const CONFIG = import.meta.glob('../../../manual/{docusaurus.config,sidebars}.ts', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const DOCS_PREFIX = '../../../manual/docs/';
const DOCUSAURUS_CONFIG = CONFIG['../../../manual/docusaurus.config.ts'];
const SIDEBARS = CONFIG['../../../manual/sidebars.ts'];

/** The manual source for a page path ("core-workflow/circuits"), if it exists. */
function sourceOf(page: string): string | undefined {
  const path = page.split('#')[0];
  return DOCS[`${DOCS_PREFIX}${path}.md`] ?? DOCS[`${DOCS_PREFIX}${path}.mdx`];
}

/** Heading ids as Docusaurus makes them (github-slugger): lower case, punctuation dropped, spaces
 *  to "-", inline markdown reduced to its text. */
export function slug(heading: string): string {
  return heading
    .replace(/\[([^\]]*)\]\([^)]*\)/g, '$1')
    .trim()
    .toLowerCase()
    .replace(/[^\p{L}\p{M}\p{N}\p{Pc}\- ]/gu, '')
    .replace(/ /g, '-');
}

/** Every heading id on a page, with github-slugger's "-1", "-2" for repeats and `{#id}` honoured. */
export function headingSlugs(markdown: string): string[] {
  let fenced = false;
  const seen = new Map<string, number>();
  const out: string[] = [];
  for (const line of markdown.split('\n')) {
    if (/^\s*(```|~~~)/.test(line)) {
      fenced = !fenced;
      continue;
    }
    const m = !fenced && line.match(/^#{1,6}\s+(.*?)\s*$/);
    if (!m) continue;
    const explicit = m[1].match(/\{#([^}]+)\}\s*$/);
    let id = explicit ? explicit[1] : slug(m[1]);
    const n = seen.get(id) ?? 0;
    seen.set(id, n + 1);
    if (n > 0 && !explicit) id = `${id}-${n}`;
    out.push(id);
  }
  return out;
}

describe('the manual sources are where this test looks (fails closed, not open)', () => {
  it('finds the manual pages and its config', () => {
    expect(Object.keys(DOCS).length).toBeGreaterThan(40);
    expect(DOCUSAURUS_CONFIG).toContain('baseUrl');
    expect(SIDEBARS).toContain('manualSidebar');
  });
});

describe('the manual address comes from config', () => {
  it('the default agrees with the published site (manual/docusaurus.config.ts url + baseUrl)', () => {
    const url = DOCUSAURUS_CONFIG.match(/^\s*url:\s*'([^']+)'/m)?.[1];
    const baseUrl = DOCUSAURUS_CONFIG.match(/^\s*baseUrl:\s*'([^']+)'/m)?.[1];
    expect(url && baseUrl).toBeTruthy();
    expect(DEFAULT_MANUAL_BASE_URL).toBe(`${url}${baseUrl}`);
    expect(MANUAL_BASE_URL).toBe(DEFAULT_MANUAL_BASE_URL);
  });

  it('pages are served from the site root (routeBasePath "/"), which is what page paths assume', () => {
    expect(DOCUSAURUS_CONFIG).toMatch(/routeBasePath:\s*'\/'/);
  });

  it('the manual build fails on a broken anchor, not only on a broken page', () => {
    expect(DOCUSAURUS_CONFIG).toMatch(/^\s*onBrokenAnchors:\s*'throw'/m);
  });

  it('a configured base replaces the default host, with or without a trailing slash', () => {
    expect(manualBaseUrl('https://docs.example.test/ms')).toBe('https://docs.example.test/ms/');
    expect(manualBaseUrl('  ')).toBe(DEFAULT_MANUAL_BASE_URL);
    expect(manualBaseUrl(undefined)).toBe(DEFAULT_MANUAL_BASE_URL);
    expect(manualUrl('core-workflow/circuits#the-loop', 'https://docs.example.test/ms/')).toBe(
      'https://docs.example.test/ms/core-workflow/circuits#the-loop'
    );
    expect(manualUrl('/concepts/glossary', 'https://docs.example.test/ms/')).toBe(
      'https://docs.example.test/ms/concepts/glossary'
    );
  });
});

describe('every help topic names a manual page and section that exist', () => {
  it.each(HELP_TOPICS.map(t => [t.id, t.manual] as const))('%s -> %s', (_id, page) => {
    expect(page).toBeTruthy();
    const path = page.split('#')[0];
    const source = sourceOf(page);
    expect(source, `manual/docs/${path}.md is missing`).toBeDefined();
    // A frontmatter id or slug would move the page away from its file path.
    const front = source!.match(/^---\n([\s\S]*?)\n---/)?.[1] ?? '';
    expect(front, `${path} sets its own id or slug`).not.toMatch(/^(id|slug):/m);
    // Reachable from the manual's own navigation, not only by address.
    expect(SIDEBARS, `${path} is not in manual/sidebars.ts`).toContain(`'${path}'`);
    const anchor = page.split('#')[1];
    if (anchor) expect(headingSlugs(source!)).toContain(anchor);
  });

  it('the slug rule matches the ids Docusaurus generated for these headings', () => {
    // Read from manual/build/*.html after `npm run build` (2026-10-20).
    expect(slug('Step 4 — Train the SAE (~10–15 min)')).toBe('step-4--train-the-sae-1015-min');
    expect(slug('⚠ Score `token_ids`, not `messages`')).toBe('-score-token_ids-not-messages');
    expect(slug('Editing strengths, pins and the intensity dial λ')).toBe(
      'editing-strengths-pins-and-the-intensity-dial-λ'
    );
    expect(slug('What a readout does *not* prove')).toBe('what-a-readout-does-not-prove');
    expect(slug('Hazard warnings — compounding and cancellation')).toBe(
      'hazard-warnings--compounding-and-cancellation'
    );
  });

  it('repeats get a numbered suffix and fenced code is not a heading', () => {
    expect(headingSlugs('## Flags\n```\n## not a heading\n```\n## Flags\n')).toEqual([
      'flags',
      'flags-1',
    ]);
  });

  it('a missing heading is caught (the check is not vacuous)', () => {
    const source = sourceOf('core-workflow/circuits')!;
    expect(headingSlugs(source)).toContain('the-loop');
    expect(headingSlugs(source)).not.toContain('the-loops');
  });
});

describe('the Help screen links each summary to its manual page', () => {
  it('one "Read more in the manual" link per topic, to that topic\'s page, in a new tab', () => {
    render(<HelpPanel />);
    for (const t of HELP_TOPICS) {
      const link = screen.getByTestId(`help-manual-${t.id}`);
      expect(link).toHaveAttribute('href', `${MANUAL_BASE_URL}${t.manual}`);
      expect(link).toHaveAttribute('target', '_blank');
      expect(link.getAttribute('rel')).toContain('noopener');
      expect(link).toHaveTextContent('Read more in the manual');
      // Exactly one manual link inside each card.
      const card = document.getElementById(`help-${t.id}`)!;
      expect(card.querySelectorAll(`a[href^="${MANUAL_BASE_URL}"]`)).toHaveLength(1);
    }
    expect(screen.getAllByRole('link', { name: /Read more in the manual/ })).toHaveLength(
      HELP_TOPICS.length
    );
    expect(screen.getByTestId('manual-home')).toHaveAttribute('href', MANUAL_BASE_URL);
  });

  // Seventeen from the ENH-004 plan, plus "judge-prompts" (the judge-prompt library, 2026-10-21)
  // "behaviours" (the behaviour catalogue, Feature 037) and "probe-families" (Feature 041).
  it('has the twenty topics, each with a card and a chip', () => {
    expect(HELP_TOPICS).toHaveLength(20);
    expect(new Set(HELP_TOPICS.map(t => t.id)).size).toBe(20);
    render(<HelpPanel />);
    for (const t of HELP_TOPICS) {
      expect(screen.getByTestId(`help-chip-${t.id}`)).toHaveAttribute('href', `#/help/${t.id}`);
    }
  });
});

describe('a per-screen Help link carries its manual page', () => {
  it.each(HELP_TOPICS.map(t => [t.id, t.manual] as const))('%s', (id, page) => {
    render(<HelpLink topic={id} label="Help with this step" />);
    expect(screen.getByRole('button', { name: 'Help with this step' })).toBeInTheDocument();
    const link = screen.getByTestId(`manual-link-${id}`);
    expect(link).toHaveAttribute('href', `${MANUAL_BASE_URL}${page}`);
    expect(link).toHaveAttribute('target', '_blank');
    expect(screen.getAllByRole('link')).toHaveLength(1);
  });

  it('an unknown topic shows the Help button and no manual link', () => {
    render(<HelpLink topic="no-such-topic" />);
    expect(screen.queryByRole('link')).toBeNull();
  });
});

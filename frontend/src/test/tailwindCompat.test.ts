// @vitest-environment node
/**
 * Tailwind 3 -> 4 (2026-10-21): the compiled stylesheet keeps Tailwind 3's look.
 *
 * Tailwind 4 changed defaults that the markup relies on without saying so: the colour palette
 * (OKLCH), the space-x/space-y rule (gap moved from the first child's successors to every child but
 * the last, at zero specificity, so a child's own `mb-*` replaced the gap — the SAEs download card
 * lost 8px), placeholder colour, button cursor and the colourless ring. src/index.css restores
 * each. This compiles that file with the real Tailwind/PostCSS pipeline the build uses and reads
 * the OUTPUT, so a block that stops generating anything (a renamed directive, a Tailwind change in
 * how `@utility` overrides a core utility) fails here, not in a screenshot nobody takes.
 *
 * MUTATION CONTROLS (run 2026-10-21):
 *   * delete the `@utility space-y-*` block       -> the space-y test fails
 *   * delete the palette's emerald-500 line         -> the palette test fails
 *   * delete the placeholder block                  -> the placeholder test fails
 */
import { describe, it, expect, beforeAll } from 'vitest';
import postcss from 'postcss';
import tailwind from '@tailwindcss/postcss';
import source from '../index.css?raw';

// No `fs`/`path`: this package has no @types/node (see typeCheckRatchet.test.ts).
const ROOT = new URL('../..', import.meta.url).pathname;
const ENTRY = new URL('../index.css', import.meta.url).pathname;
let css = '';

beforeAll(async () => {
  const result = await postcss([tailwind({ base: ROOT })]).process(source, { from: ENTRY });
  css = result.css.replace(/\s+/g, ' ');
}, 60_000);

describe('the compiled stylesheet keeps the Tailwind 3 look', () => {
  it('compiled the app (sanity: utilities from the markup are present)', () => {
    expect(css).toContain('.bg-slate-950');
    expect(css).toContain('.space-y-4');
  });

  it('defines the Tailwind 3 palette in hex, and the app\'s own slate-950', () => {
    expect(css).toContain('--color-emerald-500: #10b981');
    expect(css).toContain('--color-slate-800: #1e293b');
    expect(css).toContain('--color-red-500: #ef4444');
    expect(css).toContain('--color-slate-950: #0f1629');
    expect(css).not.toMatch(/--color-emerald-500: oklch/);
  });

  it('puts space-y gaps on the following siblings, as Tailwind 3 did', () => {
    expect(css).toMatch(
      /\.space-y-4 > :not\(\[hidden\]\) ~ :not\(\[hidden\]\) \{ margin-top: calc\(var\(--spacing\) \* 4\); margin-bottom: 0; \}/
    );
    // ...and cancels Tailwind 4's margin-bottom at a specificity a child's own margin still beats.
    expect(css).toMatch(/:where\(\.space-y-4 > :not\(:last-child\)\):is\(html \*\) \{ margin-block-end: 0; \}/);
  });

  it('puts space-x gaps on the following siblings, as Tailwind 3 did', () => {
    expect(css).toMatch(
      /\.space-x-2 > :not\(\[hidden\]\) ~ :not\(\[hidden\]\) \{ margin-left: calc\(var\(--spacing\) \* 2\); margin-right: 0; \}/
    );
  });

  it('restores the gray-400 placeholder and the pointer cursor on buttons', () => {
    expect(css).toMatch(/input::placeholder, textarea::placeholder \{ color: var\(--color-gray-400\); \}/);
    expect(css).toMatch(/button:not\(:disabled\), \[role='button'\]:not\(:disabled\) \{ cursor: pointer; \}/);
  });

  it('keeps the class-based dark mode', () => {
    expect(css).toMatch(/\.dark\\:bg-slate-950 \{ &:is\(\.dark \*\) \{|\.dark\\:bg-slate-950:is\(\.dark \*\)/);
  });
});

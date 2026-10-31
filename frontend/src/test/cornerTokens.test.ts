// @vitest-environment node
/**
 * Corners are two named theme tokens (operator decision 2026-10-31, every miSuite app):
 * card = 12px, control = 8px.
 *
 * TO CHANGE THE LOOK: edit `--radius-card` / `--radius-control` in the `@theme` block of
 * src/index.css, the matching numbers in RADIUS (src/config/brand.ts), and DECIDED below. This
 * test compiles the real stylesheet with the Tailwind/PostCSS pipeline the build uses and reads
 * the OUTPUT, so it fails when the CSS and RADIUS disagree, when a token line is deleted (the
 * utilities then generate nothing), or when the values drift from the recorded decision.
 * noRawCornerClasses.test.ts is the other half: it keeps every element on these tokens.
 *
 * MUTATION CONTROLS: see 0xcc/reviews/corner_tokens_2026-10-31.md.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import postcss from 'postcss';
import tailwind from '@tailwindcss/postcss';
import source from '../index.css?raw';
import { RADIUS } from '../config/brand';

// No `fs`/`path`: this package has no @types/node (see typeCheckRatchet.test.ts).
const ROOT = new URL('../..', import.meta.url).pathname;
const ENTRY = new URL('../index.css', import.meta.url).pathname;

/** The operator's decision, recorded once. Changing the look changes this line too. */
const DECIDED = { cardPx: 12, controlPx: 8 } as const;

// Utilities the pins need, generated whether or not a component uses them today.
const PROBES = ['rounded-card', 'rounded-control', 'rounded-t-card', 'rounded-l-control'];
// Tailwind's raw scale, which `--radius-*: initial` removes; probed so the test sees whether it generates.
const RAW_SCALE = ['xs', 'sm', 'md', 'lg', 'xl', '2xl', '3xl', '4xl'];
const RAW_PROBES = [
  ...RAW_SCALE.map((s) => `rounded-${s}`),
  ...RAW_SCALE.map((s) => `rounded-t-${s}`),
  'rounded-full',
  'rounded-none',
];

let css = '';
let theme: Record<string, string> = {};

beforeAll(async () => {
  const input = `${source}\n@source inline(${JSON.stringify([...PROBES, ...RAW_PROBES].join(' '))});\n`;
  const result = await postcss([tailwind({ base: ROOT })]).process(input, { from: ENTRY });
  css = result.css.replace(/\s+/g, ' ');
  // The last declaration of each theme variable wins, as in the browser.
  theme = {};
  for (const m of css.matchAll(/(--radius-[\w-]+):\s*([^;]+);/g)) theme[m[1] ?? ''] = (m[2] ?? '').trim();
}, 60_000);

/** The body of the first rule whose selector is exactly `selector`. */
function ruleBody(selector: string): string {
  const i = css.indexOf(`${selector} {`);
  if (i === -1) return '';
  const open = css.indexOf('{', i);
  return css.slice(open + 1, css.indexOf('}', open));
}

describe('corner tokens', () => {
  it('RADIUS records the operator decision', () => {
    expect(RADIUS).toEqual(DECIDED);
  });

  it('the compiled theme defines --radius-card and --radius-control as RADIUS says', () => {
    expect(theme['--radius-card']).toBe(`${RADIUS.cardPx}px`);
    expect(theme['--radius-control']).toBe(`${RADIUS.controlPx}px`);
  });

  it('rounded-card / rounded-control and their side variants read the tokens', () => {
    expect(ruleBody('.rounded-card')).toContain('border-radius: var(--radius-card)');
    expect(ruleBody('.rounded-control')).toContain('border-radius: var(--radius-control)');
    expect(ruleBody('.rounded-t-card')).toContain('var(--radius-card)');
    expect(ruleBody('.rounded-l-control')).toContain('var(--radius-control)');
  });

  it("Tailwind's raw corner scale generates no CSS (`--radius-*: initial`)", () => {
    const generated = RAW_SCALE.flatMap((s) => [`.rounded-${s}`, `.rounded-t-${s}`]).filter(
      (sel) => ruleBody(sel) !== '',
    );
    expect(generated).toEqual([]);
    expect(Object.keys(theme).sort()).toEqual(['--radius-card', '--radius-control']);
    // full and none are not theme values and still work.
    expect(ruleBody('.rounded-full')).toContain('border-radius');
    expect(ruleBody('.rounded-none')).toContain('border-radius: 0');
  });
});

/**
 * Every corner class in the app is one of the two named tokens, `rounded-full` or `rounded-none`.
 *
 * Operator decision 2026-10-31: corners are `rounded-card` (12px: cards, panels, sections,
 * dialogs, drawers, popovers, large callouts, table frames) and `rounded-control` (8px: inputs,
 * selects, buttons, tabs, tooltips, menu items, small boxes inside a card), defined once in the
 * `@theme` block of src/index.css (pinned by cornerTokens.test.ts). Pills, badges, dots, toggles
 * and bars are `rounded-full`. A raw size (`rounded-lg`, `rounded-sm`, bare `rounded`, an
 * arbitrary `rounded-[3px]`) would bypass the tokens, so the look could no longer be changed by
 * editing two numbers.
 *
 * HOW IT READS THE SOURCE. Every non-test .ts/.tsx file under src/ is read from disk (Vite's
 * `?raw` glob — the test-file type-check has no node types, see typeCheckRatchet.test.ts) and
 * parsed with the TypeScript compiler. Every string literal and every piece of template-literal
 * text is checked, wherever it sits: a JSX className, a constant, an array, an object of
 * variants, an argument to cx(). A class assembled from pieces (`rounded-${size}`) leaves
 * `rounded-` in the template text and is refused too. Comments are not checked. CSS files are
 * checked on their `@apply` lines. Fail-open guards: the scan must cover a minimum number of
 * files and corner classes, and the extractor must find a planted class in each node kind.
 *
 * MUTATION CONTROLS: see 0xcc/reviews/corner_tokens_2026-10-31.md.
 */

import { describe, it, expect } from 'vitest';
import ts from 'typescript';

const RAW = import.meta.glob(['../**/*.{ts,tsx,css}', '!../**/*.test.{ts,tsx}', '!./**'], {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const SIDES = '(?:t|r|b|l|tl|tr|br|bl|s|e|ss|se|es|ee)';
/** The allowed corner classes, after any variant prefix (`md:`, `hover:`, `group-hover:`). */
export const ALLOWED = new RegExp(`^rounded(?:-${SIDES})?-(?:card|control|full|none)$`);
const ALLOWED_TEXT =
  'rounded[-side]-(card|control|full|none), side one of t r b l tl tr br bl s e ss se es ee';

/** A class-shaped token whose utility starts with `rounded`, with any variant prefix stripped. */
const TOKEN = /(?<=^|[\s"'`{}()[\],])!?((?:[\w[\]&.@/=-]+:)*)!?-?(rounded(?:-[\w[\]#%./-]*)?)!?(?=$|[\s"'`{}()\],])/g;

export interface Hit {
  file: string;
  line: number;
  token: string;
}

/** Corner classes in one piece of source text that is a string or template part. */
function tokensIn(text: string): { index: number; token: string; utility: string }[] {
  const out: { index: number; token: string; utility: string }[] = [];
  for (const m of text.matchAll(TOKEN)) {
    out.push({ index: m.index ?? 0, token: m[0], utility: m[2] ?? '' });
  }
  return out;
}

/** Every corner class in a .ts/.tsx source, with its line. */
export function cornerClassesInTs(file: string, source: string): (Hit & { utility: string })[] {
  const kind = file.endsWith('x') ? ts.ScriptKind.TSX : ts.ScriptKind.TS;
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, kind);
  const out: (Hit & { utility: string })[] = [];
  const visit = (node: ts.Node): void => {
    let start = -1;
    let text = '';
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      start = node.getStart(sf) + 1;
      text = source.slice(start, node.end - 1);
    } else if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
      // Head `...${`, middle `}...${`, tail `}...``: text between the delimiters.
      start = node.getStart(sf) + 1;
      text = source.slice(start, node.end - (ts.isTemplateTail(node) ? 1 : 2));
    }
    if (start >= 0) {
      for (const t of tokensIn(text)) {
        const line = sf.getLineAndCharacterOfPosition(start + t.index).line + 1;
        out.push({ file, line, token: t.token, utility: t.utility });
      }
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return out;
}

/** Corner classes on the `@apply` lines of a stylesheet. */
export function cornerClassesInCss(file: string, source: string): (Hit & { utility: string })[] {
  const out: (Hit & { utility: string })[] = [];
  source.split('\n').forEach((text, i) => {
    if (!text.includes('@apply')) return;
    for (const t of tokensIn(text.replace(/;/g, ' '))) {
      out.push({ file, line: i + 1, token: t.token, utility: t.utility });
    }
  });
  return out;
}

const ALL = Object.entries(RAW).flatMap(([key, source]) => {
  const file = key.replace(/^\.\.\//, 'src/');
  return key.endsWith('.css') ? cornerClassesInCss(file, source) : cornerClassesInTs(file, source);
});

describe('corner classes are the named tokens', () => {
  it('scans the app (fail-open guard)', () => {
    const files = Object.keys(RAW);
    expect(files.length).toBeGreaterThan(300);
    expect(files.some((f) => f.endsWith('components/panels/SettingsPanel.tsx'))).toBe(true);
    expect(files.some((f) => f.endsWith('index.css'))).toBe(true);
    // No test file and nothing under src/test/ is in the scan.
    expect(files.filter((f) => /\.test\.tsx?$/.test(f) || f.startsWith('./'))).toEqual([]);
    expect(ALL.length).toBeGreaterThan(1000);
    expect(ALL.some((h) => h.utility === 'rounded-card')).toBe(true);
    expect(ALL.some((h) => h.utility === 'rounded-control')).toBe(true);
  });

  it('finds a planted class in every node kind (fail-open guard)', () => {
    const planted = [
      'const a = "p-2 rounded-lg";',
      "const b = ['x', 'rounded-sm'];",
      'const c = `p-1 rounded ${x}`;',
      'const d = `${x} hover:rounded-md ${y} m-1`;',
      'const e = `${x} rounded-${size}`;',
      'const f = cx("a", cond && "rounded-xl");',
      'const g = { base: "rounded-[3px]" };',
      'const h = <div className="md:rounded-t-lg" />;',
    ].join('\n');
    const hits = cornerClassesInTs('planted.tsx', planted);
    expect(hits.map((h) => [h.line, h.utility])).toEqual([
      [1, 'rounded-lg'],
      [2, 'rounded-sm'],
      [3, 'rounded'],
      [4, 'rounded-md'],
      [5, 'rounded-'],
      [6, 'rounded-xl'],
      [7, 'rounded-[3px]'],
      [8, 'rounded-t-lg'],
    ]);
    expect(hits.filter((h) => !ALLOWED.test(h.utility))).toHaveLength(8);
    expect(cornerClassesInCss('x.css', '.a {\n  @apply p-2 rounded-lg;\n}').map((h) => h.line)).toEqual([2]);
  });

  it('accepts the tokens, their side variants, full and none, with any prefix', () => {
    for (const ok of ['rounded-card', 'rounded-control', 'rounded-t-card', 'rounded-bl-control', 'rounded-l-full', 'rounded-none'])
      expect([ok, ALLOWED.test(ok)]).toEqual([ok, true]);
    const prefixed = cornerClassesInTs('p.tsx', 'const a = "md:rounded-none hover:rounded-card";');
    expect(prefixed.every((h) => ALLOWED.test(h.utility))).toBe(true);
  });

  it('uses no other corner class anywhere in the source', () => {
    const offenders = ALL.filter((h) => !ALLOWED.test(h.utility)).map(
      (h) => `${h.file}:${h.line}  ${h.token}`,
    );
    expect(
      offenders,
      `Corner classes must be ${ALLOWED_TEXT}. Pick by what the element is: rounded-card for ` +
        'cards, panels, dialogs, popovers, large callouts and table frames; rounded-control for ' +
        'inputs, buttons, selects, tabs, tooltips and small boxes inside a card; rounded-full for ' +
        'pills, badges, dots and bars. The sizes are --radius-card / --radius-control in ' +
        `src/index.css. Offenders:\n${offenders.join('\n')}`,
    ).toEqual([]);
  });
});

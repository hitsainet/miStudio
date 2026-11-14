// @vitest-environment node
/**
 * The brand guard, frontend half (042 FR-042.13, FR-042.34; FTDD §6.7; FTID §3.14; TPD-15; FTASKS 7.10).
 *
 * WHAT IS SCANNED: every non-test `.ts`/`.tsx` under `src/`, parsed by the TypeScript compiler
 * (`ts.createSourceFile`), never a regex over file text: string literals, no-substitution templates,
 * template heads/middles/tails and JSX text. Comments and identifiers are never matched.
 *
 * WHAT IS ALLOWED: only what the census export says (`legacyExport.ts`): a non-retired entry's `pattern_js`
 * compiled WITH its `js_flags` (CF-16) or one of its literals; the `old` of a transition env/host/
 * history_tool record; the whole `<owner>/<name>` token of a transition repo record (this file holds no
 * repository name, TPD-33); an active allow guard. An active deny guard's match is a finding unless
 * covered. Guard activity keys to the header's `phase`; the mode (`brandMode.ts`) to its `closed`.
 *
 * The old-brand stems mirror `STEM` in `backend/tests/support/brand_surfaces.py`; `miSuite` is not one
 * (CF-15: the export's G-MISUITE deny record covers it from P5).
 */
import { describe, expect, it } from 'vitest';
import ts from 'typescript';
import { BRAND_GUARD_MODE, effectiveMode, failuresFor, type BrandGuardMode } from './brandMode';
import { loadExport, loadExports, phaseIndex, withHeader, type ExportData, type ExportRecord } from './legacyExport';

const SOURCES = import.meta.glob('../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const STEM = /mi(?:studio|llm)|mechinterp[ -]?studio/gi;
const TOKEN_CHAR = /[\w.]/;

export type ItemClass = 'string' | 'template' | 'template_head' | 'template_middle' | 'template_tail' | 'jsx_text';

export interface Item {
  readonly cls: ItemClass;
  readonly where: string;
  readonly text: string;
}

export interface Finding {
  readonly cls: ItemClass;
  readonly where: string;
  readonly token: string;
  readonly rules: string[];
}

export interface Rules {
  readonly patterns: RegExp[];
  readonly literals: string[];
  readonly oldsExact: string[];
  readonly oldsCi: string[];
  readonly repos: RegExp[];
  readonly deny: Array<[string, RegExp]>;
}

/** CF-16: every pattern_js is compiled with its record's js_flags (plus `g` to iterate). */
export function compileRecord(record: ExportRecord): RegExp {
  return new RegExp(String(record.pattern_js), `${String(record.js_flags)}g`);
}

export function isTestFile(file: string): boolean {
  return /\.(test|spec)\.tsx?$/.test(file) || /(^|\/)(test|__tests__)\//.test(file.replace(/^\.\.\//, ''));
}

export function sourceFiles(): string[] {
  return Object.keys(SOURCES).filter((f) => !isTestFile(f) && !f.endsWith('.d.ts')).sort();
}

export function tsItems(text: string, where: string): Item[] {
  const sf = ts.createSourceFile(where, text, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const items: Item[] = [];
  const add = (node: ts.Node, cls: ItemClass, value: string): void => {
    const line = sf.getLineAndCharacterOfPosition(node.getStart(sf)).line + 1;
    items.push({ cls, where: `${where}:${line}`, text: value });
  };
  const visit = (node: ts.Node): void => {
    // Literals are scanned in their SOURCE form, quotes included: register patterns anchor on them
    // (DR-25's storage keys, DR-01's quoted kinds), and a quote is never a token character.
    if (ts.isStringLiteral(node)) add(node, 'string', node.getText(sf));
    else if (ts.isNoSubstitutionTemplateLiteral(node)) add(node, 'template', node.getText(sf));
    else if (ts.isTemplateHead(node)) add(node, 'template_head', node.getText(sf));
    else if (ts.isTemplateMiddle(node)) add(node, 'template_middle', node.getText(sf));
    else if (ts.isTemplateTail(node)) add(node, 'template_tail', node.getText(sf));
    else if (ts.isJsxText(node) && node.text.trim() !== '') add(node, 'jsx_text', node.text);
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return items;
}

function escapeRe(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function activeGuards(data: ExportData): ExportRecord[] {
  const now = phaseIndex(data.header.phase);
  return data.records.filter((r) => {
    if (r.section !== 'guard') return false;
    const start = r.from_phase == null ? -1 : phaseIndex(String(r.from_phase));
    const end = r.until_phase == null ? 99 : phaseIndex(String(r.until_phase));
    return start <= now && now < end;
  });
}

export function rulesFor(data: ExportData): Rules {
  const of = (section: string): ExportRecord[] => data.records.filter((r) => r.section === section);
  const entries = of('entry').filter((e) => e.status !== 'retired');
  const transition = (section: string): ExportRecord[] => of(section).filter((r) => r.status === 'transition');
  const active = activeGuards(data);
  return {
    patterns: [
      ...entries.map(compileRecord),
      ...active.filter((g) => g.effect === 'allow').map(compileRecord),
    ],
    literals: entries.flatMap((e) => e.literals as string[]),
    oldsExact: [...transition('env'), ...of('history_tool')].map((r) => String(r.old)),
    oldsCi: transition('host').map((r) => String(r.old).toLowerCase()),
    repos: transition('repo').map((r) => new RegExp(`(?<![\\w-])${escapeRe(String(r.old))}(?![\\w-])`, 'gi')),
    deny: active.filter((g) => g.effect === 'deny').map((g) => [String(g.id), compileRecord(g)]),
  };
}

function allowedSpans(text: string, rules: Rules): Array<[number, number]> {
  const spans: Array<[number, number]> = [];
  for (const pattern of [...rules.patterns, ...rules.repos]) {
    for (const m of text.matchAll(pattern)) {
      if (m[0].length > 0) spans.push([m.index, m.index + m[0].length]);
    }
  }
  const low = text.toLowerCase();
  const find = (needle: string, hay: string): void => {
    let at = hay.indexOf(needle);
    while (at !== -1 && needle) {
      spans.push([at, at + needle.length]);
      at = hay.indexOf(needle, at + 1);
    }
  };
  for (const lit of [...rules.literals, ...rules.oldsExact]) find(lit, text);
  for (const host of rules.oldsCi) find(host, low);
  return spans;
}

const covered = (spans: Array<[number, number]>, s: number, e: number): boolean =>
  spans.some(([a, b]) => a <= s && e <= b);

function tokenAt(text: string, start: number, end: number): string {
  let s = start;
  let e = end;
  while (s > 0 && TOKEN_CHAR.test(text[s - 1]!)) s -= 1;
  while (e < text.length && TOKEN_CHAR.test(text[e]!)) e += 1;
  return text.slice(s, e).replace(/^\.+|\.+$/g, '');
}

export function scan(items: readonly Item[], rules: Rules): Finding[] {
  const findings: Finding[] = [];
  for (const item of items) {
    const spans = allowedSpans(item.text, rules);
    const brand: Array<[number, number, Finding]> = [];
    for (const m of item.text.matchAll(STEM)) {
      const end = m.index + m[0].length;
      if (covered(spans, m.index, end)) continue;
      const f: Finding = { cls: item.cls, where: item.where, token: tokenAt(item.text, m.index, end), rules: [`brand:${m[0].toLowerCase()}`] };
      brand.push([m.index, end, f]);
      findings.push(f);
    }
    for (const [id, pattern] of rules.deny) {
      for (const m of item.text.matchAll(pattern)) {
        const end = m.index + m[0].length;
        if (end === m.index || covered(spans, m.index, end)) continue;
        const hits = brand.filter(([s, e]) => s < end && m.index < e).map(([, , f]) => f);
        if (hits.length) hits.forEach((f) => f.rules.includes(id) || f.rules.push(id));
        else findings.push({ cls: item.cls, where: item.where, token: m[0], rules: [id] });
      }
    }
  }
  return findings;
}

/** The real-tree verdict: every finding printed; a failure only where the guard is enforced (R23). */
function realTreeCheck(constant: BrandGuardMode, data: ExportData, items: readonly Item[], label: string): void {
  const started = performance.now();
  const findings = scan(items, rulesFor(data));
  const seconds = ((performance.now() - started) / 1000).toFixed(2);
  const lines = findings.map((f) => `[${f.cls}] ${f.where}: ${f.token} (${f.rules.join(', ')})`).sort();
  const variant = String(data.header.variant);
  console.warn(
    `[brand guard ${effectiveMode(constant, variant)}] frontend/src: ${lines.length} finding(s) in ${label}, ` +
      `${items.length} items, ${seconds} s\n${lines.map((l) => `  ${l}`).join('\n')}`,
  );
  expect(failuresFor(lines, constant, variant)).toStrictEqual([]);
}

function realItems(): Item[] {
  return sourceFiles().flatMap((f) => tsItems(SOURCES[f] as string, f.replace(/^\.\.\//, 'frontend/src/')));
}

const PLANTED = [
  "const a = 'Open miStudio';",
  'const b = `miLLM settings`;',
  'const c = `miStudio ${x} plain ${y} plain`;',
  'const d = `plain ${x} MechInterp Studio ${y} plain`;',
  'const e = `plain ${x} plain ${y} millm tail`;',
  'const f = <p>Welcome to miStudio</p>;',
  '// miStudio in a comment is not a surface',
  'const miStudioIdentifier = 1;',
].join('\n');

describe('brand guard (frontend)', () => {
  const data = loadExport();

  it('finds one planted string per class', () => {
    const findings = scan(tsItems(PLANTED, 'planted.tsx'), rulesFor(data));
    const perClass: Record<string, number> = {};
    for (const f of findings) perClass[f.cls] = (perClass[f.cls] ?? 0) + 1;
    expect(perClass).toStrictEqual({
      string: 1,
      template: 1,
      template_head: 1,
      template_middle: 1,
      template_tail: 1,
      jsx_text: 1,
    });
  });

  it('compiles each record with js_flags', () => {
    const records = data.records.filter((r) => r.section === 'entry' || r.section === 'guard');
    expect(records.length).toBeGreaterThan(10);
    for (const r of records) {
      const rx = compileRecord(r);
      expect(rx.flags.replace('g', '')).toBe(String(r.js_flags));
      for (const lit of (r.literals as string[] | undefined) ?? []) {
        rx.lastIndex = 0;
        expect(rx.test(lit), `${String(r.id)} misses ${lit}`).toBe(true);
      }
    }
    // A case-insensitive deny record (js_flags "i") must catch an upper-case spelling, through the scanner.
    const ci = rulesFor(data).deny.filter(([, rx]) => rx.flags.includes('i'));
    expect(ci.length).toBeGreaterThan(0);
    const [id] = ci[0]!;
    const found = scan([{ cls: 'string', where: 'x', text: 'Open MISTUDIO now' }], rulesFor(data));
    expect(found).toHaveLength(1);
    expect(found[0]!.rules).toContain(id);
  });

  it('R23: a planted string fails with the full export and is only reported with the public one', () => {
    const all = loadExports();
    const pub = all.find((e) => e.header.variant === 'public');
    const full = all.find((e) => e.header.variant === 'full');
    // Both trees carry the public variant; only the private repository carries the full one.
    expect(pub, 'the public export is committed in every tree').toBeDefined();
    expect(pub!.header.withheld.length, 'the public export withholds records: the reason it cannot enforce').toBeGreaterThan(0);
    const lines = scan(tsItems("const a = 'Open miStudio';", 'planted.tsx'), rulesFor(pub!)).map((f) => f.where);
    expect(lines).toHaveLength(1);
    // The mirror (public export): reported, never a failure, whatever the constant says.
    expect(failuresFor(lines, 'enforce', String(pub!.header.variant))).toStrictEqual([]);
    expect(failuresFor(lines, 'report', String(pub!.header.variant))).toStrictEqual([]);
    // The private repository (full export): a failure once the constant says enforce, not before.
    expect(failuresFor(lines, 'enforce', 'full')).toStrictEqual(lines);
    expect(failuresFor(lines, 'report', 'full')).toStrictEqual([]);
    expect(effectiveMode('enforce', 'anything-else')).toBe('report');
    // THIS tree's verdict under `enforce` follows the export it actually loads.
    const here = String(loadExport().header.variant);
    expect(here).toBe(full ? 'full' : 'public');
    expect(effectiveMode('enforce', here)).toBe(full ? 'enforce' : 'report');
  });

  it('real tree in the current mode', () => {
    const files = sourceFiles();
    const items = realItems();
    expect(files.length).toBeGreaterThan(100);
    expect(items.length).toBeGreaterThan(1000);
    expect(files.some((f) => isTestFile(f))).toBe(false);
    realTreeCheck(BRAND_GUARD_MODE, data, items, `${files.length} files`);
  });

  it('R23: the real-tree check itself fails a plant with the full export and passes it with the public one', () => {
    // The same function the real-tree test runs, with the constant's value replaced by `enforce`.
    const planted = tsItems("const a = 'Open miStudio';", 'planted.tsx');
    const pub = loadExports().find((e) => e.header.variant === 'public')!;
    expect(() => realTreeCheck('enforce', pub, planted, 'plant')).not.toThrow();
    expect(() => realTreeCheck('enforce', withHeader(pub, { variant: 'full' }), planted, 'plant')).toThrow();
    expect(() => realTreeCheck('report', withHeader(pub, { variant: 'full' }), planted, 'plant')).not.toThrow();
  });
});

/**
 * The JavaScript half of the export checks (042 FR-042.6, FR-042.7; FTASKS 6.4; CF-25 m5).
 *
 * Controls (0xcc/reviews/042_implementation_controls_*.md): K-6.16 drops the literal check in
 * `legacyExport.ts` → the planted-miss case is red; V-skip removes the skip detection → the skip cases are red.
 */
import { describe, expect, it } from 'vitest';
import { literalProblems, loadExport, loadExports, vitestProblem } from './legacyExport';

const TEST_SOURCES = import.meta.glob('/src/**/*.test.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

describe('legacy-identifier export (JavaScript half)', () => {
  it('every literal matches new RegExp(pattern_js, js_flags)', () => {
    const variants = loadExports();
    expect(variants.length).toBeGreaterThan(0);
    for (const variant of variants) {
      const entries = variant.records.filter((r) => r.section === 'entry');
      expect(entries.length).toBeGreaterThan(10);
      const withLiterals = entries.filter((r) => ((r.literals as string[]) ?? []).length > 0);
      expect(withLiterals.length).toBeGreaterThan(5);
      expect(literalProblems(variant.records)).toEqual([]);
    }
    // A record whose pattern_js misses a literal the Python pattern matches is reported, by id and value.
    const planted = [{ section: 'entry', id: 'DR-X', pattern_js: '\\bmistudio-only\\b', js_flags: '',
                       literals: ['MISTUDIO-ONLY'] }];
    expect(literalProblems(planted)).toEqual(['DR-X: MISTUDIO-ONLY']);
    expect(literalProblems([{ ...planted[0], js_flags: 'i' }])).toEqual([]);
  });

  it('vitest exercises resolve to a test of that exact name', () => {
    const exercises = loadExport().records
      .filter((r) => r.section === 'entry' && typeof r.exercise === 'string')
      .map((r) => [String(r.id), r.exercise as string] as const)
      .filter(([, ex]) => ex.startsWith('vitest:'));
    expect(exercises.length).toBeGreaterThan(0);
    for (const [id, exercise] of exercises) {
      const [path, name] = exercise.slice('vitest:'.length).split('::');
      expect(path.startsWith('frontend/src/'), `${id}: ${path}`).toBe(true);
      const key = '/' + path.slice('frontend/'.length);
      const source = TEST_SOURCES[key];
      expect(source, `${id}: ${path} does not exist`).toBeTypeOf('string');
      expect(vitestProblem(source, name, path), id).toBeNull();
    }
    // The resolver refuses what would not run (miLLM P0 review, V01).
    const named = "describe('d', () => { it('the name', () => {}); });";
    expect(vitestProblem(named, 'the name')).toBeNull();
    expect(vitestProblem(named, 'another name')).toMatch(/no it\/test named/);
    for (const skipped of [
      "it.skip('the name', () => {});",
      "test.todo('the name');",
      "xit('the name', () => {});",
      "describe.skip('d', () => { it('the name', () => {}); });",
      "xdescribe('d', () => { test('the name', () => {}); });",
    ]) {
      expect(vitestProblem(skipped, 'the name'), skipped).toMatch(/skipped or todo/);
    }
    expect(vitestProblem(named + "\nit.only('x', () => {});", 'the name')).toMatch(/\.only/);
  });
});

// Review rev-mistudio-p0 F1 (the miLLM T03 finding): every form keeps the named test from running and read as a
// runnable exercise before; test_frozen_identifiers_coverage.py holds the Python resolver to the same list.
const SKIP_FORMS: Array<[string, string]> = [
  ['describe.skipIf', "describe.skipIf(true)('d', () => { it('the name', () => {}); });"],
  ['describe.runIf(false)', "describe.runIf(false)('d', () => { it('the name', () => {}); });"],
  ['it.skipIf', "it.skipIf(true)('the name', () => {});"],
  ['it.runIf(false)', "it.runIf(false)('the name', () => {});"],
  ['it {skip: true}', "it('the name', { skip: true }, () => {});"],
  ['it {todo: true}', "it('the name', { todo: true }, () => {});"],
  ['describe {skip: true}', "describe('d', { skip: true }, () => { it('the name', () => {}); });"],
  ['ctx.skip()', "it('the name', (ctx) => { ctx.skip(); });"],
  ['destructured skip()', "test('the name', ({ skip }) => { skip(); });"],
  ['if (false)', "if (false) { it('the name', () => {}); }"],
  ['never-called helper', "function later() { it('the name', () => {}); }"],
  ['describe.concurrent.skip', "describe.concurrent.skip('d', () => { it('the name', () => {}); });"],
  ['it.fails', "it.fails('the name', () => {});"],
];

describe('the vitest exercise resolver refuses every skip form', () => {
  it.each(SKIP_FORMS)('%s', (_form, source) => {
    expect(vitestProblem(source, 'the name')).not.toBeNull();
  });
  it('still accepts the real exercise shapes', () => {
    for (const ok of [
      "describe('d', () => { it('the name', () => { if (x) { y(); } }); });",
      "describe('d', () => it('the name', () => {}));",
      "it.concurrent('the name', () => {});",
      "function helper() { return 1; }\ndescribe('d', () => { it('the name', () => {}); });",
    ]) {
      expect(vitestProblem(ok, 'the name'), ok).toBeNull();
    }
  });
});

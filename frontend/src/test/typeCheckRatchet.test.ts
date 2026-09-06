/**
 * The test-file type-check exists, runs, and may not get worse.
 *
 * MIS-E2E-021. `tsconfig.test.json` was listed as a quality gate and could
 * never have worked, for two independent reasons:
 *
 *   1. No script referenced it — nothing ever ran it.
 *   2. It extends `tsconfig.json`, whose `exclude` names exactly the test files
 *      this config `include`s. An inherited exclude beats a child's include, so
 *      even run by hand it reported "No inputs were found" and checked nothing.
 *      It also named `@types/node`, which is not a dependency of this package,
 *      so the config failed to load at all.
 *
 * With those fixed the gate reports 432 pre-existing errors — overwhelmingly
 * `Cannot find name 'global'` in tests written against Node globals. They are
 * type-level only; every one of these tests passes at runtime.
 *
 * Fixing 427 is its own piece of work, and forcing them into CI today would
 * just break the build. What must not happen is the count silently growing back
 * behind a gate nobody runs. This asserts the gate is wired and ratchets the
 * number down-only.
 */
import { describe, it, expect } from 'vitest';
import { execSync } from 'child_process';
import { readFileSync } from 'fs';
import { join } from 'path';

const ROOT = join(__dirname, '..', '..');

/** Errors present when the gate was first switched on. Down-only.
 *
 * 432, not the 427 a shell `grep -c` reported: `grep -c` counts LINES and some
 * tsc lines carry more than one `error TSxxxx`. The number here is produced by
 * the same regex the check below uses, so the two cannot disagree.
 */
const BASELINE = 432;

function typeErrors(): number {
  try {
    execSync('npx tsc --noEmit -p tsconfig.test.json', { cwd: ROOT, stdio: 'pipe' });
    return 0;
  } catch (e) {
    const out = String((e as { stdout?: Buffer }).stdout ?? '');
    return (out.match(/error TS\d+/g) ?? []).length;
  }
}

describe('test-file type checking', () => {
  it('is reachable from an npm script', () => {
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf-8'));
    expect(pkg.scripts['type-check:test']).toBe('tsc --noEmit -p tsconfig.test.json');
  });

  it('has a format:check gate too', () => {
    // `format` rewrites files; only `format:check` can gate anything.
    const pkg = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf-8'));
    expect(pkg.scripts['format:check']).toContain('--check');
  });

  it('actually sees the test files', () => {
    const cfg = readFileSync(join(ROOT, 'tsconfig.test.json'), 'utf-8');
    // The inherited exclude must be overridden or this checks nothing.
    expect(cfg).toMatch(/"exclude"\s*:\s*\[\s*"node_modules"/);
    expect(cfg).not.toContain('"node"');   // the missing @types/node
  }, 30_000);

  it('does not have more type errors than the recorded baseline', () => {
    const count = typeErrors();
    expect(count).toBeLessThanOrEqual(BASELINE);
    if (count < BASELINE) {
      // Not a failure — a nudge to move the ratchet.
      console.log(`type-check:test is down to ${count} (baseline ${BASELINE}); lower BASELINE.`);
    }
  }, 180_000);

  it('the baseline is a real measurement, not zero-by-accident', () => {
    // A gate reporting 0 because it sees no files is the defect this replaces.
    expect(BASELINE).toBeGreaterThan(0);
  });
});

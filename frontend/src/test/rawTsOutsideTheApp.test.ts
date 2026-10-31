/**
 * The manual's TypeScript config files load as TEXT in the Frontend CI job (ENH-004).
 *
 * manualLinks.test.tsx reads manual/docusaurus.config.ts and manual/sidebars.ts through
 * `?raw`. Vite still hands a `.ts?raw` module to its esbuild plugin, which looks for the nearest
 * tsconfig.json: manual/tsconfig.json, which extends "@docusaurus/tsconfig". GitHub's Frontend CI
 * runs `npm ci` in frontend/ only, so that package is not installed there and the whole
 * manualLinks file failed to load ("failed to resolve extends"), while every local run, with
 * manual/node_modules present, stayed green. vite.config.ts therefore keeps esbuild off the
 * top-level .ts files of any `manual/` directory in test mode.
 *
 * This fixture reproduces the CI condition on every machine: a `manual/` directory whose
 * tsconfig.json extends a package that is never installed. Without the exclusion this file
 * fails to load.
 */
import { describe, expect, it } from 'vitest';

const FIXTURE = import.meta.glob('./fixtures/manual/*.ts', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

describe('a manual/ TypeScript file read with ?raw', () => {
  it('loads as its own text even when its tsconfig.json cannot be resolved', () => {
    const text = FIXTURE['./fixtures/manual/site.config.ts'];
    expect(text, 'fixture not found by the glob').toBeDefined();
    expect(text).toContain("export const marker = 'raw-fixture-loaded';");
  });
});

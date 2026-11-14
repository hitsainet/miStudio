// @vitest-environment node
/**
 * MIS-E2E-020: what the production build is allowed to ship.
 *
 * The config had `sourcemap: true` unconditionally, so every release published
 * 12 `.map` files that reconstruct the full original source — comments and all
 * — for anyone who opens devtools. 364 `console.log`/`debug`/`info` calls went
 * with it, some printing request payloads.
 *
 * Until 2026-10-21 this file only READ the config text. Vite 8 then started
 * ignoring the `esbuild.pure` option it was reading (Rolldown/Oxc build the
 * bundle; the esbuild block is warned about and skipped), and every assertion
 * here stayed green while a production build would have shipped the console
 * calls again. So the behaviour is now checked by BUILDING: a fixture module
 * goes through the real vite.config.ts in production mode (in memory, no
 * files written, about a second) and the emitted code is read.
 */
import { describe, it, expect } from 'vitest';
import { build, type Rolldown } from 'vite';
import config from '../../vite.config.ts?raw';

// No `fs`/`path`: this package has no @types/node (see typeCheckRatchet.test.ts).
const CONFIG_FILE = new URL('../../vite.config.ts', import.meta.url).pathname;
const FIXTURE = new URL('../test/fixtures/bundling/consoleCalls.ts', import.meta.url).pathname;

async function buildFixture(mode: 'production' | 'development') {
  const result = await build({
    configFile: CONFIG_FILE,
    mode,
    logLevel: 'silent',
    build: {
      write: false,
      minify: mode === 'production',
      rolldownOptions: {
        input: FIXTURE,
        preserveEntrySignatures: 'exports-only',
      },
    },
  });
  const outputs = (Array.isArray(result) ? result : [result]) as Rolldown.RolldownOutput[];
  const files = outputs.flatMap((o) => o.output);
  const code = files
    .filter((f): f is Rolldown.OutputChunk => f.type === 'chunk')
    .map((f) => f.code)
    .join('\n');
  return { code, files: files.map((f) => f.fileName) };
}

describe('production build hardening', () => {
  it('drops console.log/debug/info/trace and keeps console.warn/error in a production build', async () => {
    const { code } = await buildFixture('production');
    // The fixture must actually have been built, or every "not.toContain" below is vacuous.
    expect(code).toContain('marker-error');
    expect(code).toContain('marker-warn');
    for (const gone of ['marker-log', 'marker-debug', 'marker-info', 'marker-trace']) {
      expect(code, `${gone} survived the production build`).not.toContain(gone);
    }
  }, 60_000);

  it('emits no sourcemap in a production build', async () => {
    const { files } = await buildFixture('production');
    expect(files.length).toBeGreaterThan(0);
    expect(files.filter((f) => f.endsWith('.map'))).toEqual([]);
  }, 60_000);

  it('keeps the console calls and the sourcemap in a development build', async () => {
    // The whole fix keys on `mode`; this fails if it ever applies to every build.
    const { code, files } = await buildFixture('development');
    expect(code).toContain('marker-log');
    expect(files.some((f) => f.endsWith('.map'))).toBe(true);
  }, 60_000);

  it('reads the build mode, so dev builds keep their tooling', () => {
    // The object form of defineConfig has no `mode` in scope.
    expect(config).toMatch(/defineConfig\(\(\s*\{\s*mode\s*\}\s*\)\s*=>/);
  });
});

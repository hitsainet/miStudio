/**
 * MIS-E2E-020: what the production build is allowed to ship.
 *
 * The config had `sourcemap: true` unconditionally, so every release published
 * 12 `.map` files that reconstruct the full original source — comments and all
 * — for anyone who opens devtools. 364 `console.log`/`debug`/`info` calls went
 * with it, some printing request payloads.
 *
 * This reads the CONFIG rather than a built bundle, because a test that needs
 * `npm run build` to have run first passes vacuously on a clean checkout. The
 * built artifact was verified by hand once: 12 maps → 0, `console.log` → 0,
 * `console.error` → 116 still present.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';

const config = readFileSync(join(__dirname, '../../vite.config.ts'), 'utf-8');

describe('production build hardening', () => {
  it('does not emit sourcemaps unconditionally', () => {
    expect(config).not.toMatch(/sourcemap:\s*true\s*,/);
    expect(config).toMatch(/sourcemap:\s*mode !== 'production'/);
  });

  it('marks the chatty console methods pure so they are dropped', () => {
    const pure = config.match(/pure:[\s\S]*?\]/)?.[0] ?? '';
    for (const method of ['console.log', 'console.debug', 'console.info']) {
      expect(pure).toContain(method);
    }
  });

  it('keeps console.warn and console.error — a user is asked to read those', () => {
    const pure = config.match(/pure:[\s\S]*?\]/)?.[0] ?? '';
    expect(pure).not.toContain('console.warn');
    expect(pure).not.toContain('console.error');
  });

  it('reads the build mode, so dev builds keep their tooling', () => {
    // The object form of defineConfig has no `mode` in scope; the whole fix
    // silently does nothing if someone converts it back.
    expect(config).toMatch(/defineConfig\(\(\s*\{\s*mode\s*\}\s*\)\s*=>/);
  });
});

/**
 * The sidebar's version comes from package.json, which agrees with the repository's VERSION
 * file (ENH-004). The sidebar used to carry its own literal '0.5.0' that nothing tied to either.
 */
import { describe, expect, it } from 'vitest';

import pkg from '../../package.json';
import { APP_VERSION } from './version';

// Read through Vite's ?raw import, not node:fs (test files have no Node types; see
// typeCheckRatchet.test.ts).
import VERSION_FILE from '../../../VERSION?raw';

describe('the app version', () => {
  it('is package.json "version", defined at build time', () => {
    expect(APP_VERSION).toBe(pkg.version);
    expect(APP_VERSION).toMatch(/^\d+\.\d+\.\d+/);
  });

  it('agrees with the repository VERSION file', () => {
    expect(VERSION_FILE, 'VERSION not found').toBeDefined();
    expect(VERSION_FILE.trim()).toBe(pkg.version);
  });
});

/**
 * The sidebar's brand text must come FROM the brand config, not from a second
 * hardcoded copy beside it.
 *
 * Background: `BRAND` was imported by zero files while 25 imported `COMPONENTS`
 * from the same module. The sidebar hardcoded the name and tagline, so the two
 * copies were free to disagree — and `BRAND.version` had already drifted to
 * '0.1.0' while VERSION and package.json both said 0.5.0. Nothing caught it
 * because nothing read it.
 */
import { describe, it, expect } from 'vitest';
import { readFileSync } from 'fs';
import { join } from 'path';
import { BRAND } from '../../config/brand';

const source = readFileSync(join(__dirname, 'Sidebar.tsx'), 'utf-8');

describe('Sidebar brand wiring', () => {
  it('renders the tagline from BRAND rather than a literal', () => {
    expect(source).toContain('{BRAND.tagline}');
    // The literal must NOT also appear — that is the divergence this prevents.
    expect(source).not.toContain('AI Feature Discovery Workbench');
  });

  it('renders the product name from BRAND rather than a literal', () => {
    expect(source).toContain('{BRAND.name}');
    expect(source).not.toMatch(/>MechInterp Studio</);
    expect(source).not.toContain('alt="MechInterp Studio"');
  });

  it('carries the tagline the product actually ships', () => {
    expect(BRAND.tagline).toBe('AI Feature Discovery Workbench');
    expect(BRAND.name).toBe('MechInterp Studio');
  });

  it('does not hand-maintain a version that can drift from VERSION', () => {
    // 'version' in BRAND was stale by four minor releases and invisible.
    expect(BRAND).not.toHaveProperty('version');
  });
});

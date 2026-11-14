// @vitest-environment node
/**
 * The frontend mode constant agrees with the backend one and with the export's `closed` (042 T-48; CF-21).
 */
import { describe, expect, it } from 'vitest';
import { BRAND_GUARD_MODE } from './brandMode';
import { loadExport, phaseIndex, withHeader, type ExportData } from './legacyExport';

const BACKEND = import.meta.glob('../../../backend/tests/support/brand_surfaces.py', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

/** The backend constant: the one assignment at column 0, never a comment or a docstring line. */
export function backendMode(source: string): string {
  const hits = [...source.matchAll(/^BRAND_GUARD_MODE = "(report|enforce)"$/gm)];
  if (hits.length !== 1) throw new Error(`expected one BRAND_GUARD_MODE assignment, found ${hits.length}`);
  return hits[0]![1]!;
}

export function requiredMode(data: ExportData): 'report' | 'enforce' {
  return phaseIndex(data.header.closed) >= phaseIndex('P4') ? 'enforce' : 'report';
}

describe('brand guard mode (frontend)', () => {
  it('mode agrees with the backend constant', () => {
    const sources = Object.values(BACKEND);
    expect(sources).toHaveLength(1);
    expect(BRAND_GUARD_MODE).toBe(backendMode(sources[0]!));
  });

  it('mode is the one the export closed requires', () => {
    const data = loadExport();
    expect(BRAND_GUARD_MODE).toBe(requiredMode(data));
    // Keyed to `closed`, never to `phase` (CF-21).
    expect(requiredMode(withHeader(data, { phase: 'P4', closed: 'P3' }))).toBe('report');
    expect(requiredMode(withHeader(data, { phase: 'P5', closed: 'P4' }))).toBe('enforce');
  });
});

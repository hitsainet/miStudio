// @vitest-environment node
/**
 * docs/schemas/probe-identity-cases.json, recomputed by a SECOND language (Feature 039 FR-6, G1).
 *
 * The case file freezes `mistudio.probe-identity/1` — the detector and decision digests of a probe
 * definition. miStudio's Python module and a Python standard-library checker already agree with it;
 * this one shares no code with either. Its number formatting is the engine's own
 * `Number.prototype.toString`, which is the algorithm RFC 8785 cites, so a Python formatter that
 * drifted from ECMAScript would disagree here first.
 *
 * Everything is built from the FILE: the projections, the defaults an absent key is hashed as, the
 * framing. The JCS writer below follows RFC 8785 §3.2: `JSON.stringify` for strings (which escapes
 * exactly `"`, `\` and control characters), keys sorted by UTF-16 code units (JavaScript's default
 * string order), numbers by `String(n)`.
 *
 * HOW IT READS THE FILE: through Vite's `?raw`, like manualLinks.test.tsx — not node:fs, which this
 * package's type-check has no types for (typeCheckRatchet.test.ts). vite.config.ts allows the
 * repository root in test mode. Hashing is Web Crypto (`crypto.subtle`), global in Node.
 */
import { describe, expect, it } from 'vitest';

const FILES = import.meta.glob('../../../docs/schemas/probe-identity-cases.json', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

type Json = null | boolean | number | string | Json[] | { [key: string]: Json };
type JsonObject = { [key: string]: Json };
interface Spec {
  include: Record<string, string[]>;
  exclude: Record<string, string[]>;
  added_after_v1: Record<string, string[]>;
}
interface CaseEntry {
  name: string;
  document: JsonObject;
  detector_canonical: string;
  decision_canonical: string;
  detector_digest: string;
  decision_digest: string;
  relation?: { to: string; detector: 'equal' | 'differs'; decision: 'equal' | 'differs' };
}
interface CaseFile {
  version: number;
  algorithm: string;
  projection: { detector: Spec; decision: Spec; defaults: Record<string, Json> };
  numbers: { bits: string; jcs: string }[];
  cases: CaseEntry[];
}

function loadCases(text?: string): CaseFile {
  const raw = text ?? Object.values(FILES)[0];
  if (raw === undefined) throw new Error('probe-identity-cases.json was not found by the glob');
  return JSON.parse(raw) as CaseFile;
}

function numberFromBits(bits: string): number {
  const view = new DataView(new ArrayBuffer(8));
  for (let i = 0; i < 8; i += 1) view.setUint8(i, parseInt(bits.slice(i * 2, i * 2 + 2), 16));
  return view.getFloat64(0, false);
}

function compareUtf16(a: string, b: string): number {
  return a < b ? -1 : a > b ? 1 : 0; // JavaScript compares strings by UTF-16 code units
}

function jcs(value: Json): string {
  if (value === null) return 'null';
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${value} has no JSON form`);
    return String(value);
  }
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(jcs).join(',')}]`;
  const keys = Object.keys(value).sort(compareUtf16);
  return `{${keys.map((key) => `${JSON.stringify(key)}:${jcs(value[key])}`).join(',')}}`;
}

function join(path: string, key: string): string {
  return path === '' ? key : `${path}.${key}`;
}

function clone(value: Json): Json {
  return value === undefined ? null : (JSON.parse(JSON.stringify(value)) as Json);
}

function project(value: JsonObject, spec: Spec, defaults: Record<string, Json>, prefix: string,
  path = ''): JsonObject {
  const out: JsonObject = {};
  const nested = (key: string, item: Json): Json => {
    const child = join(path, key);
    if (item === null) return null;
    if (child in spec.include) return project(item as JsonObject, spec, defaults, prefix, child);
    if (`${child}[]` in spec.include) {
      return (item as JsonObject[]).map((entry) => project(entry, spec, defaults, prefix, `${child}[]`));
    }
    if (`${child}{}` in spec.include) {
      const map: JsonObject = {};
      for (const [name, entry] of Object.entries(item as Record<string, JsonObject>)) {
        map[name] = project(entry, spec, defaults, prefix, `${child}{}`);
      }
      return map;
    }
    return clone(item);
  };
  for (const key of spec.include[path] ?? []) {
    const full = join(prefix, join(path, key));
    const item = key in value ? value[key] : clone(defaults[full] ?? null);
    out[key] = nested(key, item);
  }
  for (const key of spec.added_after_v1[path] ?? []) {
    const full = join(prefix, join(path, key));
    const item = value[key];
    if (item === undefined || item === null || jcs(item) === jcs(defaults[full] ?? null)) continue;
    out[key] = nested(key, item);
  }
  return out;
}

async function sha256Hex(text: string): Promise<string> {
  const bytes = new TextEncoder().encode(text);
  const hash = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return Array.from(new Uint8Array(hash), (b) => b.toString(16).padStart(2, '0')).join('');
}

async function identityOf(document: JsonObject, file: CaseFile) {
  const { detector, decision, defaults } = file.projection;
  const frame = (kind: string, payload: JsonObject) =>
    jcs({ algorithm: file.algorithm, digest: kind, payload });
  const detectorCanonical = frame('detector', project(document, detector, defaults, 'detector'));
  const decisionCanonical = frame(
    'decision',
    project((document.decision ?? {}) as JsonObject, decision, defaults, 'decision'),
  );
  return {
    detector_canonical: detectorCanonical,
    decision_canonical: decisionCanonical,
    detector_digest: await sha256Hex(detectorCanonical),
    decision_digest: await sha256Hex(decisionCanonical),
  };
}

async function mismatches(file: CaseFile): Promise<string[]> {
  const wrong: string[] = [];
  for (const entry of file.numbers) {
    const got = String(numberFromBits(entry.bits));
    if (got !== entry.jcs) wrong.push(`number ${entry.bits}: ${got} != ${entry.jcs}`);
  }
  const keys = ['detector_canonical', 'decision_canonical', 'detector_digest', 'decision_digest'] as const;
  const byName = new Map(file.cases.map((c) => [c.name, c]));
  for (const entry of file.cases) {
    const got = await identityOf(entry.document, file);
    for (const key of keys) {
      if (got[key] !== entry[key]) wrong.push(`${entry.name}: ${key}`);
    }
    if (entry.relation) {
      const other = byName.get(entry.relation.to);
      if (!other) {
        wrong.push(`${entry.name}: relation to unknown case ${entry.relation.to}`);
        continue;
      }
      const theirs = await identityOf(other.document, file);
      for (const kind of ['detector', 'decision'] as const) {
        const same = got[`${kind}_digest`] === theirs[`${kind}_digest`];
        if (same !== (entry.relation[kind] === 'equal')) wrong.push(`${entry.name}: ${kind} relation`);
      }
    }
  }
  return wrong;
}

describe('probe-identity-cases.json, recomputed in TypeScript', () => {
  it('has the sections and the size the other checkers expect', () => {
    const file = loadCases();
    expect(file.version).toBe(1);
    expect(file.algorithm).toBe('mistudio.probe-identity/1');
    expect(file.numbers.length).toBeGreaterThanOrEqual(90);
    expect(file.cases.length).toBeGreaterThanOrEqual(40);
  });

  it('every number is written as Number#toString writes it', () => {
    const file = loadCases();
    const wrong = file.numbers.filter((entry) => String(numberFromBits(entry.bits)) !== entry.jcs);
    expect(wrong).toEqual([]);
  });

  it('every case recomputes to its canonical text and both digests', async () => {
    expect(await mismatches(loadCases())).toEqual([]);
  });

  it('a changed number string is caught (the checker is not vacuous)', async () => {
    const file = loadCases();
    file.numbers[3] = { ...file.numbers[3], jcs: `${file.numbers[3].jcs}0` };
    expect((await mismatches(file)).some((line) => line.startsWith('number '))).toBe(true);
  });

  it('a changed expected digest is caught', async () => {
    const file = loadCases();
    const first = file.cases[0];
    file.cases[0] = { ...first, detector_digest: `0${first.detector_digest.slice(1)}` === first.detector_digest
      ? `1${first.detector_digest.slice(1)}` : `0${first.detector_digest.slice(1)}` };
    expect(await mismatches(file)).toContain(`${first.name}: detector_digest`);
  });
});

// @vitest-environment node
/**
 * Generality guard, frontend half (Feature 040 FR-8 – FR-10; FTDD §6.8; FTASKS 1.3, 1.7).
 *
 * No source names a behaviour. A behaviour is data (a catalogue row, a seeded judge prompt, a pack);
 * a component that spells one key, or help text that explains the product through one behaviour,
 * works for that behaviour and reads wrong for every other.
 *
 * THE TOKENS ARE DERIVED, NEVER LISTED. They are read from the same files the backend half reads
 * (`backend/src/data/judge_prompt_seeds/*.json`, `backend/tests/fixtures/behaviour_packs/*.json`)
 * through Vite's glob — vite.config.ts allows the repository root in test mode — and the derivation
 * refuses fewer than MIN_FILES files or MIN_TOKENS tokens, so a wrong glob fails instead of matching
 * nothing. The rule for what a token is mirrors `backend/tests/support/behaviour_tokens.py`: a
 * behaviour key and label, and a pack's key, name, sub-question keys and rule ids; a judge prompt's
 * own `name` is not a token (FTID TI-15).
 *
 * WHAT IS SCANNED. Every non-test `.ts`/`.tsx` under `src/`, parsed by the TypeScript compiler (not a
 * regex over file text, which fails open on a layout it did not expect): string literals, template
 * literal parts, JSX text and every comment (leading and trailing comment ranges of every node, read
 * with the compiler's own comment scanner). The FTDD names `ts.createScanner`; a bare scanner cannot
 * tell a `//` inside a template literal from a comment without re-scanning template tokens, so the
 * parse tree plus `forEachLeadingCommentRange` is used — the same compiler, without that trap.
 *
 * Matching: case-insensitive, on the boundaries of the slug alphabet `[a-z0-9-]`.
 */
import { describe, expect, it } from 'vitest';
import ts from 'typescript';

const SOURCES = import.meta.glob('../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

const TOKEN_FILES = {
  ...(import.meta.glob('../../../backend/src/data/judge_prompt_seeds/*.json', {
    query: '?raw',
    import: 'default',
    eager: true,
  }) as Record<string, string>),
  ...(import.meta.glob('../../../backend/tests/fixtures/behaviour_packs/*.json', {
    query: '?raw',
    import: 'default',
    eager: true,
  }) as Record<string, string>),
};

/** Fixed when S1 landed (2026-10-23), equal to the backend half's. */
const MIN_FILES = 2;
const MIN_TOKENS = 4;

type Doc = Record<string, unknown>;

function strings(value: unknown): string[] {
  return typeof value === 'string' && value.trim() ? [value.trim()] : [];
}

export function tokensFromDocument(doc: Doc): string[] {
  const out = [...strings(doc.behaviour), ...strings(doc.behaviour_label)];
  if (String(doc.format ?? '').startsWith('mistudio.behaviour-pack/')) {
    // A pack holds one behaviour at its top level, or several under `behaviours`.
    for (const entry of (doc.behaviours as Doc[] | undefined) ?? [doc]) {
      out.push(...strings(entry.key), ...strings(entry.name));
      for (const sub of (entry.sub_questions as Doc[] | undefined) ?? []) out.push(...strings(sub?.key));
      for (const id of (entry.rule_ids as unknown[] | undefined) ?? []) out.push(...strings(id));
    }
  }
  return out;
}

export function deriveTokens(files: Record<string, string>): string[] {
  const names = Object.keys(files);
  if (names.length < MIN_FILES) {
    throw new Error(
      `the generality guard's globs matched ${names.length} file(s); at least ${MIN_FILES} are expected`
    );
  }
  const found = new Set<string>();
  for (const text of Object.values(files)) {
    for (const token of tokensFromDocument(JSON.parse(text) as Doc)) {
      found.add(token.toLowerCase().split(/\s+/).join(' '));
    }
  }
  if (found.size < MIN_TOKENS) {
    throw new Error(`the generality guard derived ${found.size} token(s); at least ${MIN_TOKENS} are expected`);
  }
  return [...found].sort();
}

function escape(word: string): string {
  return word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export function patternFor(token: string): RegExp {
  const body = token.split(' ').map(escape).join('\\s+');
  return new RegExp(`(?<![a-z0-9-])${body}(?![a-z0-9-])`, 'i');
}

export interface Hit { file: string; line: number; kind: string; token: string; text: string }

/** Every string, template part, JSX text and comment in one source, with the tokens it contains. */
export function scanSource(file: string, source: string, tokens: string[]): Hit[] {
  const patterns = tokens.map((token) => [token, patternFor(token)] as const);
  const sf = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true,
    file.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const hits: Hit[] = [];
  const check = (pos: number, kind: string, text: string) => {
    for (const [token, pattern] of patterns) {
      if (pattern.test(text)) {
        const line = sf.getLineAndCharacterOfPosition(pos).line + 1;
        hits.push({ file, line, kind, token, text: text.trim().slice(0, 120) });
      }
    }
  };
  const seenComments = new Set<number>();
  const comment = (pos: number, end: number) => {
    if (seenComments.has(pos)) return;
    seenComments.add(pos);
    check(pos, 'comment', source.slice(pos, end));
  };
  const visit = (node: ts.Node) => {
    ts.forEachLeadingCommentRange(source, node.pos, comment);
    ts.forEachTrailingCommentRange(source, node.end, comment);
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      check(node.getStart(sf), 'string', node.text);
    } else if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
      check(node.getStart(sf), 'template', node.text);
    } else if (ts.isJsxText(node)) {
      check(node.getStart(sf), 'jsx', node.text);
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  ts.forEachLeadingCommentRange(source, sf.endOfFileToken.pos, comment);
  return hits;
}

function isTest(file: string): boolean {
  return /\.test\.tsx?$/.test(file) || file.startsWith('../test/') || file.includes('/__tests__/');
}

const TOKENS = deriveTokens(TOKEN_FILES);

describe('no behaviour names in frontend source (FR-8)', () => {
  it('derives a token set from the data files, not a list', () => {
    expect(Object.keys(TOKEN_FILES).length).toBeGreaterThanOrEqual(MIN_FILES);
    expect(TOKENS.length).toBeGreaterThanOrEqual(MIN_TOKENS);
    for (const text of Object.values(TOKEN_FILES)) {
      const doc = JSON.parse(text) as Doc;
      if (typeof doc.behaviour === 'string') expect(TOKENS).toContain(doc.behaviour.toLowerCase());
      // every behaviour of every pack file (a pack lists several under `behaviours`)
      for (const entry of (doc.behaviours as Doc[] | undefined) ?? []) {
        expect(TOKENS).toContain(String(entry.key).toLowerCase());
      }
    }
  });

  it('finds no token in any non-test source', () => {
    const files = Object.keys(SOURCES).filter((f) => !isTest(f));
    expect(files.length).toBeGreaterThan(200);
    const hits = files.flatMap((file) => scanSource(file, SOURCES[file], TOKENS));
    const report = hits.map((h) => `${h.file}:${h.line} [${h.kind}] '${h.token}' in ${JSON.stringify(h.text)}`);
    expect(report, `${hits.length} hit(s) in ${new Set(hits.map((h) => h.file)).size} file(s)`).toEqual([]);
  });

  // ── the guard's own controls, permanent ──────────────────────────────────────────────────
  it('canary: a token in each node kind of a synthetic source is found', () => {
    const key = TOKENS.find((t) => !t.includes(' ')) as string;
    const label = TOKENS.find((t) => t.includes(' ')) ?? key;
    const source = [
      `// handles ${key}`,
      `/* ${label} */`,
      `const a = '${key.toUpperCase()}';`,
      'const b = `x ${a} ' + key + '`;',
      `export const C = () => <p>about ${label}</p>;`,
      `const d = '${key}-presence';`, // inside a longer slug: not a match
      'const e = `${a}// not a comment`;',
    ].join('\n');
    const kinds = scanSource('canary.tsx', source, [key, label]).map((h) => `${h.line}:${h.kind}`);
    expect(new Set(kinds)).toEqual(new Set(['1:comment', '2:comment', '3:string', '4:template', '5:jsx']));
  });

  it('a wrong glob fails instead of matching nothing', () => {
    expect(() => deriveTokens({})).toThrow(/matched 0 file/);
  });
});

/**
 * User-facing copy must not admit unfinished work.
 *
 * An inventory (2026-10-18) found five places where the UI explained a
 * capability's absence instead of providing it: "Feature details not available
 * for downloaded SAEs" (the route existed), circuits "coming with the mining
 * feature" (discovery had shipped), "Additional filters coming soon",
 * "Additional filters can be added here", and a Display tab whose only content
 * was "Additional display settings will be added here in a future update".
 * Each is now built. This keeps the next one from shipping.
 *
 * HOW IT READS THE SOURCE. Not with a regex over file text — a source scrape
 * matches comments and fails open on a layout it did not expect. Each
 * non-test .ts/.tsx file under src/ is parsed with the TypeScript compiler, and
 * only the nodes that can reach a screen are checked: string literals, template
 * literal text and JSX text. Comments are not copy and are not checked. Two
 * fail-open guards: the scan must cover a minimum number of files and strings,
 * and the extractor is run on a synthetic source that must produce a hit in
 * each node kind.
 *
 * ALLOWLIST. Only for a genuine refusal when a SIBLING PRODUCT is absent (the
 * suite rule: integrations are detected at runtime and refuse clearly), keyed
 * by file and an exact snippet, each with its reason. A stale entry (its
 * snippet no longer present) fails too, so the list cannot rot.
 *
 * MUTATION CONTROLS (2026-10-18, each restored by sha256):
 *   G1  put "Additional filters coming soon" back into TrainingTemplateList   -> 1 red
 *   G2  put "will be added here in a future update" back into SettingsPanel   -> 1 red
 *   G3  extractor skips JsxText (only the synthetic check sees JSX hits now)  -> 1 red
 *   G4  put "Feature details not available for downloaded SAEs" back into
 *       FeatureSelector                                                       -> 1 red
 */

import { describe, it, expect } from 'vitest';
import ts from 'typescript';

// Sources come from Vite's glob, not node:fs — the test-file type-check has no
// node types, and fs/path/__dirname are three errors against its down-only
// ratchet (the lesson probeClientIsReachable.test.ts records). Keys arrive
// relative to this file (`../components/...`).
const RAW = import.meta.glob('../**/*.{ts,tsx}', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;

/** Phrases that announce unfinished work rather than describe the product. */
export const UNFINISHED_PHRASES: RegExp[] = [
  /coming soon/i,
  /coming with the/i,
  /not (yet )?implemented( yet)?/i,
  /can be added here/i,
  /will be added (here|later|soon|in)/i,
  /in a future (update|release|version)/i,
  /not available in this build/i,
  /not available for (downloaded|external|imported)/i,
  /not (yet )?supported yet/i,
  /future enhancement/i,
  /placeholder for future/i,
];

/** file (relative to src, forward slashes) -> exact snippet -> reason. */
export const ALLOWLIST: Record<string, Record<string, string>> = {};

export interface CopyHit { file: string; line: number; kind: string; text: string }

export function extractCopy(fileName: string, source: string): CopyHit[] {
  const sf = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true,
    fileName.endsWith('.tsx') ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
  const out: CopyHit[] = [];
  const push = (node: ts.Node, kind: string, text: string) => {
    if (!text.trim()) return;
    const { line } = sf.getLineAndCharacterOfPosition(node.getStart(sf));
    out.push({ file: fileName, line: line + 1, kind, text });
  };
  const visit = (node: ts.Node) => {
    if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      push(node, 'string', node.text);
    } else if (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) || ts.isTemplateTail(node)) {
      push(node, 'template', node.text);
    } else if (ts.isJsxText(node)) {
      // JSX collapses whitespace runs; so must we, or a phrase wrapped across
      // two source lines would slip through.
      push(node, 'jsx', node.text.replace(/\s+/g, ' '));
    }
    ts.forEachChild(node, visit);
  };
  visit(sf);
  return out;
}

/** Non-test sources, keyed relative to src/ (e.g. `components/panels/X.tsx`). */
function sourceFiles(): [string, string][] {
  return Object.entries(RAW)
    .map(([key, text]) => [key.replace(/^\.\.\//, ''), text] as [string, string])
    .filter(([rel]) => !/(^|\/)(test|__tests__)\//.test(rel)
      && !/\.(test|spec)\.(ts|tsx)$/.test(rel)
      && !rel.endsWith('.d.ts'));
}

function scan() {
  const files = sourceFiles();
  const hits: CopyHit[] = [];
  let strings = 0;
  const seenAllowed = new Set<string>();
  for (const [rel, text] of files) {
    for (const c of extractCopy(rel, text)) {
      strings += 1;
      if (!UNFINISHED_PHRASES.some((re) => re.test(c.text))) continue;
      const allowed = Object.keys(ALLOWLIST[rel] ?? {}).find((snip) => c.text.includes(snip));
      if (allowed) { seenAllowed.add(`${rel}::${allowed}`); continue; }
      hits.push(c);
    }
  }
  return { files: files.length, strings, hits, seenAllowed };
}

describe('user-facing copy admits no unfinished work', () => {
  const result = scan();

  it('scans the whole frontend (fails closed, not open)', () => {
    expect(result.files).toBeGreaterThan(200);
    expect(result.strings).toBeGreaterThan(5000);
  });

  it('the extractor sees every kind of copy and ignores comments', () => {
    const src = [
      '// coming soon in a comment is not copy',
      'const a = "Filters coming soon";',
      'const b = `Shipping in a future update: ${x}`;',
      'const C = () => <p>Additional filters can be\n  added here</p>;',
    ].join('\n');
    const kinds = extractCopy('x.tsx', src)
      .filter((c) => UNFINISHED_PHRASES.some((re) => re.test(c.text)))
      .map((c) => c.kind)
      .sort();
    expect(kinds).toEqual(['jsx', 'string', 'template']);
  });

  it('no string, template or JSX text announces unfinished work', () => {
    const report = result.hits.map((h) => `${h.file}:${h.line} [${h.kind}] ${h.text.trim()}`);
    expect(report, 'Provide the capability, or say what the product does now').toEqual([]);
  });

  it('every allowlist entry is still present and carries a reason', () => {
    for (const [file, entries] of Object.entries(ALLOWLIST)) {
      for (const [snippet, reason] of Object.entries(entries)) {
        expect(reason.trim().length, `${file}: reason for "${snippet}"`).toBeGreaterThan(20);
        expect(result.seenAllowed.has(`${file}::${snippet}`), `stale allowlist entry ${file}`).toBe(true);
      }
    }
  });
});

/**
 * Copy audit for the Feature 016 surfaces of the Circuits screen.
 *
 * 016 produces rungs 0–1 only: a capture, a mined association and an attribution
 * score. None of them is an intervention, so none of the copy that presents them
 * may say or imply cause (016 FTASKS 6.2, "no causal language in 016 surfaces").
 * The backend half is `backend/tests/unit/test_causal_language_audit.py`; until
 * 2026-10-21 nothing read the frontend, where the user actually reads the copy.
 *
 * WHAT IS READ. The four 016 components of `CircuitsPanel.tsx` — the capture tab,
 * the discovery tab, a discovery run's detail and its report card — found BY NAME
 * in the TypeScript syntax tree. Their copy is every JSX text node and every string
 * or template literal inside them. Comments are not copy and are not read. The
 * rest of the file is 017/018 (validation, faithfulness), which earns rung 2 and
 * may use the word.
 *
 * FAILS CLOSED. A renamed or removed component is a red, not a smaller corpus: an
 * audit that silently reads nothing is the failure the backend audit's history is
 * full of. Each component must also yield a known anchor sentence, which proves
 * the extractor reaches its JSX.
 *
 * ALLOWED: a forbidden word whose own sentence denies it before it appears
 * ("never proof", "not causal"). Nothing else.
 *
 * MUTATION CONTROLS (2026-10-21, each restored by sha256): see 016 FTASKS 6.2.
 */
import { describe, it, expect } from 'vitest';
import ts from 'typescript';
import panelSource from './CircuitsPanel.tsx?raw';
import helpSource from '../../config/help.ts?raw';

/** The 016 components and one sentence each must contain (proves the read). */
const SURFACES: Record<string, string> = {
  CaptureTab: 'Capture co-activation events across layers',
  DiscoveryTab: 'Mine candidate associations from a capture',
  DiscoveryRunDetail: 'rank Δ',
  RunReportCard: 'Stage counts',
};

const FORBIDDEN =
  /\b(caus(?:e|es|ed|ing|al|ally|ation|ality)|driv(?:e|es|en|ing)|drove|mechanis(?:m|ms|tic)|influenc\w*|prov(?:e|es|ed|en|ing)|proof)\b/gi;
const NEGATOR = /\b(not|no|never|nor|without|isn't|aren't|cannot)\b/i;

/** Every user-visible string inside the named top-level functions of a TSX source. */
export function copyByFunction(source: string, fileName = 'x.tsx'): Record<string, string[]> {
  const sf = ts.createSourceFile(fileName, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX);
  const out: Record<string, string[]> = {};
  const collect = (node: ts.Node, sink: string[]) => {
    if (ts.isJsxText(node)) {
      const t = node.text.replace(/\s+/g, ' ').trim();
      if (t) sink.push(t);
    } else if (ts.isStringLiteral(node) || ts.isNoSubstitutionTemplateLiteral(node)) {
      sink.push(node.text);
    } else if (ts.isTemplateExpression(node)) {
      sink.push([node.head.text, ...node.templateSpans.map((s) => s.literal.text)].join(' … '));
    }
    ts.forEachChild(node, (c) => collect(c, sink));
  };
  for (const stmt of sf.statements) {
    if (ts.isFunctionDeclaration(stmt) && stmt.name) {
      const sink: string[] = [];
      collect(stmt, sink);
      out[stmt.name.text] = sink;
    }
  }
  return out;
}

/** `[sentence, word]` for every forbidden word not denied earlier in its sentence. */
export function overclaims(strings: string[]): Array<[string, string]> {
  const bad: Array<[string, string]> = [];
  for (const s of strings) {
    for (const sentence of s.split(/(?<=[.!?;—])\s+/)) {
      for (const m of sentence.matchAll(FORBIDDEN)) {
        if (!NEGATOR.test(sentence.slice(0, m.index))) bad.push([sentence, m[0]]);
      }
    }
  }
  return bad;
}

describe('016 copy makes no causal claim (rungs 0–1 only)', () => {
  const copy = copyByFunction(panelSource, 'CircuitsPanel.tsx');

  it.each(Object.keys(SURFACES))('%s exists and its copy is read', (name) => {
    expect(copy[name], `${name} is gone or renamed — the audit would read nothing`).toBeDefined();
    expect(copy[name].join(' ')).toContain(SURFACES[name]);
  });

  it.each(Object.keys(SURFACES))('%s says nothing causal', (name) => {
    expect(overclaims(copy[name] ?? [])).toEqual([]);
  });

  it('the help steps for capture and discovery say nothing causal', () => {
    const steps = copyByFunction(`function help() { return [${
      helpSource.split('\n').filter((l) => /^\s*'(Capture|Discovery):/.test(l)).join('\n')
    }]; }`).help;
    expect(steps).toHaveLength(2);
    expect(overclaims(steps)).toEqual([]);
  });
});

describe('the audit can fail', () => {
  it.each([
    'This edge is causal.',
    'The upstream feature drives the downstream one.',
    'Discovery found the mechanism.',
    'Attribution proves the link.',
    'A strong influence on the later layer.',
  ])('flags %j', (s) => {
    expect(overclaims([s])).not.toEqual([]);
  });

  it.each([
    'candidates and attribution support, never proof',
    'An association is not causal evidence.',
  ])('permits the denial %j', (s) => {
    expect(overclaims([s])).toEqual([]);
  });

  it('reads JSX text, attributes and template literals inside a named component', () => {
    const c = copyByFunction(
      'function DiscoveryTab() { const t = `a ${1} drives b`; return <div title="x causes y">z</div>; }',
    ).DiscoveryTab;
    expect(overclaims(c).map(([, w]) => w.toLowerCase()).sort()).toEqual(['causes', 'drives']);
  });
});

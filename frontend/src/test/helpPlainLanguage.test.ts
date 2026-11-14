/**
 * The in-app Help text is plain language (ENH-004, decision D8: Help topic text in v1).
 *
 * Help is for a person who has never seen the planning documents, so two things are banned:
 *
 *   1. INTERNAL IDS AND PLANNING LANGUAGE — requirement ids (FR-0xx., BRD-, ADR-, MIS-E2E),
 *      decision ids "(D12)", planning paths (0xcc/, FTASKS). The patterns are miForge's
 *      manual/publish-rules.json `internal` rules (the suite's one list), copied here because a
 *      test may not read a sibling repository; widened with this repo's own id shapes (IDL-,
 *      OSD-, T-97).
 *   2. A BARE ACRONYM — any run of two or more capitals (`SAE`, `SAEs`, `GPU`) must be spelled
 *      out in parentheses at its FIRST use in that topic, either "sparse autoencoder (SAE)" or
 *      "SAE (sparse autoencoder)", and must be listed on the manual's glossary page
 *      (manual/docs/concepts/glossary.md, "Abbreviations" table), which seeds the expansions.
 *
 * Fail-closed guards (the noUnfinishedWorkCopy.test.ts pattern): the scan must cover every topic
 * and a minimum amount of text, the glossary table must parse, and a synthetic topic carrying
 * `FR-009.1` and a bare `AUROC` must be caught.
 */
import { describe, expect, it } from 'vitest';

import { HELP_TOPICS, type HelpTopic } from '../config/help';

const GLOSSARY_GLOB = import.meta.glob('../../../manual/docs/concepts/glossary.md', {
  query: '?raw',
  import: 'default',
  eager: true,
}) as Record<string, string>;
const GLOSSARY = Object.values(GLOSSARY_GLOB)[0] ?? '';

/** Origin: miForge manual/publish-rules.json @ 36c1e50, category "internal" (+ miStudio's ids). */
export const INTERNAL_PATTERNS: { id: string; pattern: RegExp }[] = [
  { id: 'planning-tree', pattern: /0xcc|\bAGENT\.md\b|\bCLAUDE\.md\b|\bGEMINI\.md\b/ },
  {
    id: 'requirement-id',
    pattern:
      /\bR-0\d\.\d|\bBRD-|\b(?:PPRD|PADR|FPRD|FTDD|FTID|FTASKS)\b|\bADR-\d|\bMIS-E2E|\bFR-0\d\d\./,
  },
  { id: 'decision-id', pattern: /\(D\d{1,2}\)|\bfeature 0\d\d\b|\bF\d{1,2}'s\b/ },
  { id: 'mistudio-id', pattern: /\bIDL-\d|\bOSD-\d|\bT-\d{2,}\b|\bENH-\d/ },
];

/** Capitalised words that are names, not abbreviations. Keep it short; prefer rewording. */
export const NOT_ACRONYMS = new Set<string>([]);

/** Acronyms and their expansions from the glossary's "Abbreviations" table. */
export function glossaryAbbreviations(markdown: string): Map<string, string> {
  const out = new Map<string, string>();
  const section = markdown.split(/^## Abbreviations\s*$/m)[1] ?? '';
  for (const line of section.split('\n')) {
    const m = line.match(/^\|\s*\*\*([A-Z][A-Za-z0-9-]*)\*\*\s*\|\s*([^|]+?)\s*\|/);
    if (m) out.set(m[1], m[2]);
  }
  return out;
}

export interface Finding {
  topic: string;
  rule: string;
  text: string;
}

function topicText(t: HelpTopic): string {
  return [t.title, ...t.body, ...(t.steps ?? [])].join('\n');
}

/** Plain-language findings for one topic. */
export function checkTopic(t: HelpTopic, glossary: Map<string, string>): Finding[] {
  const text = topicText(t);
  const findings: Finding[] = [];
  for (const { id, pattern } of INTERNAL_PATTERNS) {
    const m = text.match(pattern);
    if (m) findings.push({ topic: t.id, rule: id, text: m[0] });
  }
  const seen = new Set<string>();
  // Two or more capitals as a whole word, optionally plural ("SAEs"), hyphenated parts included
  // ("TF-IDF").
  const re = /\b([A-Z]{2,}(?:-[A-Z]{2,})*)(s?)\b/g;
  for (const m of text.matchAll(re)) {
    const acronym = m[1];
    if (NOT_ACRONYMS.has(acronym) || seen.has(acronym)) continue;
    seen.add(acronym);
    const start = m.index ?? 0;
    const end = start + m[0].length;
    const after = text.slice(end);
    const before = text.slice(0, start);
    // "SAE (sparse autoencoder)" or "sparse autoencoder (SAE)".
    const expandedAfter = /^\s*\([a-z][^)]*\)/.test(after);
    const expandedBefore = /[a-z][\w\s'–-]*\($/.test(before) && after.startsWith(')');
    if (!expandedAfter && !expandedBefore) {
      findings.push({ topic: t.id, rule: 'acronym-not-spelled-out', text: acronym });
    }
    if (!glossary.has(acronym)) {
      findings.push({ topic: t.id, rule: 'acronym-not-in-glossary', text: acronym });
    }
  }
  return findings;
}

describe('Help text is plain language', () => {
  const glossary = glossaryAbbreviations(GLOSSARY);

  it('reads the glossary and every topic (fails closed, not open)', () => {
    expect(GLOSSARY, 'manual/docs/concepts/glossary.md not found').toContain('# Glossary');
    expect(glossary.size).toBeGreaterThanOrEqual(5);
    expect(glossary.get('SAE')).toBe('sparse autoencoder');
    expect(HELP_TOPICS.length).toBeGreaterThanOrEqual(17);
    const chars = HELP_TOPICS.map(topicText).join('').length;
    expect(chars).toBeGreaterThan(6000);
  });

  it('no topic carries an internal id or a bare acronym', () => {
    const findings = HELP_TOPICS.flatMap(t => checkTopic(t, glossary));
    expect(
      findings,
      'Spell the acronym out at first use, and list it on the glossary page'
    ).toEqual([]);
  });

  it('negative control: a planning id and a bare acronym are caught', () => {
    const bad: HelpTopic = {
      id: 'bad',
      title: 'Bad sample',
      body: ['Implements FR-009.1 (D12): the probe AUROC is shown on the card.'],
      manual: 'x',
    };
    const rules = checkTopic(bad, glossary).map(f => `${f.rule}:${f.text}`);
    expect(rules).toContain('requirement-id:FR-009.');
    expect(rules).toContain('decision-id:(D12)');
    expect(rules).toContain('acronym-not-spelled-out:AUROC');
  });

  it('accepts either spelling-out order, plurals, and later bare uses', () => {
    const ok: HelpTopic = {
      id: 'ok',
      title: 'Fine',
      body: [
        'A sparse autoencoder (SAE) has features. SAEs are trained here. AUROC (area under the receiver operating characteristic curve) grades a probe; a higher AUROC is better.',
      ],
      manual: 'x',
    };
    expect(checkTopic(ok, glossary)).toEqual([]);
  });

  it('an acronym missing from the glossary is caught even when spelled out', () => {
    const t: HelpTopic = {
      id: 't',
      title: 'T',
      body: ['A quasi-zonal widget (QZW) is not real.'],
      manual: 'x',
    };
    expect(checkTopic(t, glossary).map(f => f.rule)).toEqual(['acronym-not-in-glossary']);
  });

  it('every glossary abbreviation carries a lower-case expansion', () => {
    for (const [acronym, expansion] of glossary) {
      expect(expansion, acronym).toMatch(/^[a-zA-Z]/);
      expect(expansion.length, acronym).toBeGreaterThan(4);
    }
  });
});

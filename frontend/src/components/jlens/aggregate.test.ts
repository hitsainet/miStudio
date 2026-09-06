/**
 * Collapsing a readout into a ranked list.
 *
 * The count is only meaningful with its denominator: the same token scores
 * differently under a different layer range and a different top_n, so a bare
 * number invites a comparison that is not valid. These tests pin that the
 * range is honoured, that the filter reports what it removed, and that the
 * axis is indexed by POSITION and not by absolute layer number.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * ignore the range and count every layer     -> "SELECTED range" fails
 *   * index top_tokens by absolute layer number  -> "partial artifact" fails
 *   * drop tokens silently                       -> "says how many it hid" fails
 *   * sort by count alone                        -> "ties broken by best rank" fails
 */
import { describe, expect, it } from 'vitest';
import { aggregate, isNonWord, layersInRange } from './aggregate';

/** `axis` is passed at every call site so no test can agree with a default. */
function tok(position: number, type: string, rows: string[][]) {
  return {
    kind: 'token' as const,
    position,
    token: `t${position}`,
    id: position,
    is_generated: false,
    results: [{ type, top_tokens: rows, top_probs: rows.map((r) => r.map(() => 0.1)) }],
  };
}

describe('isNonWord', () => {
  it('removes punctuation fragments, whitespace and sentinels', () => {
    for (const t of ['^(@)', '$.', '_', '   ', '<unused0>', '[PAD]', '♪']) {
      expect(isNonWord(t), t).toBe(true);
    }
  });

  it('KEEPS letters in any script, including archaic and non-Latin', () => {
    /**
     * The filter declutters; it does not decide what counts as a real word in
     * a language it was not told about. Hiding these would remove legitimate
     * multilingual output and disguise a readout rather than tidy it.
     */
    for (const t of ['myſelf', 'Jefus', 'صوتيه', '醐', 'nahilalakip', 'dog', 'B12']) {
      expect(isNonWord(t), t).toBe(false);
    }
  });
});

describe('layersInRange', () => {
  it('is inclusive at both ends', () => {
    expect(layersInRange([0, 5, 10, 15], [5, 10])).toEqual([5, 10]);
  });

  it('tolerates a reversed range rather than returning nothing', () => {
    expect(layersInRange([0, 5, 10], [10, 5])).toEqual([5, 10]);
  });

  it('null means every layer, not none', () => {
    expect(layersInRange([0, 5, 10], null)).toEqual([0, 5, 10]);
  });
});

describe('aggregate', () => {
  it('counts a token once per (layer, position) cell it reaches', () => {
    const axis = [0, 1];
    const tokens = [
      tok(0, 'JACOBIAN_LENS', [['dog', 'cat'], ['dog', 'pet']]),
      tok(1, 'JACOBIAN_LENS', [['dog', 'x'], ['y', 'z']]),
    ];
    const { ranked } = aggregate(tokens, 'JACOBIAN_LENS', axis, null);
    const dog = ranked.find((r) => r.token === 'dog')!;
    expect(dog.count).toBe(3); // (L0,p0) (L1,p0) (L0,p1)
    expect(dog.layers).toEqual([0, 1]);
    expect(dog.bestRank).toBe(0);
  });

  it('counts only the SELECTED range', () => {
    // MUTATION CONTROL: ignore `range` and this fails at 3.
    const axis = [0, 1];
    const tokens = [
      tok(0, 'JACOBIAN_LENS', [['dog'], ['dog']]),
      tok(1, 'JACOBIAN_LENS', [['dog'], ['dog']]),
    ];
    const { ranked } = aggregate(tokens, 'JACOBIAN_LENS', axis, [1, 1]);
    expect(ranked.find((r) => r.token === 'dog')!.count).toBe(2);
  });

  it('indexes the axis by POSITION, so a partial artifact reads its own rows', () => {
    /**
     * `top_tokens[layerIdx]` is indexed by position in the axis, NOT by the
     * model's absolute layer number. On a partial fit the two differ, and
     * indexing with the absolute number reads the wrong row — producing a
     * plausible ranked list rather than an error.
     *
     * MUTATION CONTROL: index by absolute layer and this throws or mis-counts.
     */
    const axis = [20, 21]; // absolute layers; rows are still [0] and [1]
    const tokens = [tok(0, 'JACOBIAN_LENS', [['high'], ['higher']])];
    const { ranked } = aggregate(tokens, 'JACOBIAN_LENS', axis, null);
    expect(ranked.map((r) => r.token).sort()).toEqual(['high', 'higher']);
    expect(ranked.find((r) => r.token === 'high')!.layers).toEqual([20]);
  });

  it('says how many non-words it hid rather than hiding them silently', () => {
    // MUTATION CONTROL: drop the token without counting and this fails at 0.
    const axis = [0];
    const tokens = [tok(0, 'JACOBIAN_LENS', [['dog', '^(@)', '$.', '_']])];
    const out = aggregate(tokens, 'JACOBIAN_LENS', axis, null, { hideNonWords: true });
    expect(out.ranked.map((r) => r.token)).toEqual(['dog']);
    expect(out.hiddenNonWords).toBe(3);
  });

  it('shows everything when the filter is off, and reports nothing hidden', () => {
    const axis = [0];
    const tokens = [tok(0, 'JACOBIAN_LENS', [['dog', '^(@)']])];
    const out = aggregate(tokens, 'JACOBIAN_LENS', axis, null, { hideNonWords: false });
    expect(out.ranked).toHaveLength(2);
    expect(out.hiddenNonWords).toBe(0);
  });

  it('breaks ties on best rank, so a top-1 outranks an equally common top-8', () => {
    /**
     * Two tokens seen equally often are not equally prominent if one of them
     * was top-1.
     *
     * THE INSERTION ORDER IS DELIBERATELY OPPOSITE TO THE RANK ORDER. An
     * earlier version of this put the better-ranked token in the same cell at
     * rank 0, so it was inserted first and Array.sort — which is stable — kept
     * it first with or without the tie-break. The test could not fail, and
     * sorting by count alone survived the mutation.
     *
     * Here `worse` (best rank 1) is inserted BEFORE `better` (best rank 0), and
     * all three tie at count 1, so only the tie-break can reorder them.
     *
     * MUTATION CONTROL: sort by count alone and `better` stays behind `worse`.
     */
    const axis = [0, 1];
    const tokens = [
      tok(0, 'JACOBIAN_LENS', [
        ['first', 'worse'], // first: rank 0, worse: rank 1
        ['better'], //         better: rank 0, inserted LAST
      ]),
    ];
    const { ranked } = aggregate(tokens, 'JACOBIAN_LENS', axis, null);
    expect(ranked.map((r) => r.count)).toEqual([1, 1, 1]);
    expect(ranked.map((r) => r.token)).toEqual(['first', 'better', 'worse']);
  });

  it('returns nothing for a lens type the readout does not carry', () => {
    const axis = [0];
    const tokens = [tok(0, 'LOGIT_LENS', [['a']])];
    expect(aggregate(tokens, 'JACOBIAN_LENS', axis, null).ranked).toEqual([]);
  });

  it('tolerates an axis longer than the rows the response actually holds', () => {
    /** A short slice must not throw; it means those layers were not served. */
    const axis = [0, 1, 2];
    const tokens = [tok(0, 'JACOBIAN_LENS', [['dog']])];
    expect(() => aggregate(tokens, 'JACOBIAN_LENS', axis, null)).not.toThrow();
  });
});

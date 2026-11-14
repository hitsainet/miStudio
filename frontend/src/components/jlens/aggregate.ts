/**
 * Collapse a readout into a ranked list of tokens.
 *
 * The readout is a rank-4 structure — `tokens[position].results[type]
 * .top_tokens[layerIdx][k]` — and the question a reader actually has is flatter
 * than that: what is this model poised to say across the stack, and where do the
 * two lenses disagree? Answering it from a layer x position grid means
 * collapsing four dimensions by eye.
 *
 * DERIVED ON THE CLIENT, DELIBERATELY. `schemas/jlens.py` opens with BR-029 /
 * PADR IDL-45: the wire format is fixed by an external contract, and "adding a
 * miStudio-shaped field here would silently break that property". A count field
 * on the stream would end the interchangeability the format exists to provide.
 * Every top-k is already in the response, so the aggregate is derivable — and
 * because it is derivable, it must be derived rather than requested.
 */

import type { LensTokenMessage, LensTypeSlice } from '../../types/jlens';

export interface RankedToken {
  token: string;
  /** `(layer, position)` cells in range where this token reached the top-k. */
  count: number;
  /** ABSOLUTE layer numbers it appeared at — drives the per-layer strip. */
  layers: number[];
  /** Best (lowest) rank it achieved anywhere in range. 0-based. */
  bestRank: number;
}

export interface AggregateResult {
  ranked: RankedToken[];
  /**
   * How many distinct tokens the non-word filter removed.
   *
   * ALWAYS REPORTED, never merely applied. A filter that silently drops rows
   * from a ranked list is how a readout starts lying about what the model is
   * poised to say — the reader cannot tell a model that surfaced nothing from
   * one whose output was tidied away.
   */
  hiddenNonWords: number;
}

/**
 * Tokens with no letter and no digit, plus tokenizer sentinel forms.
 *
 * SCOPE, STATED HONESTLY. This removes punctuation fragments (`^(@)`, `$.`,
 * `_`), whitespace-only pieces and sentinels (`<unused0>`, `[PAD]`). It does
 * NOT remove tokens that are letters in another script or archaic spellings —
 * `myſelf`, `Jefus`, `صوتيه`, `醐` all survive, because they ARE letters and a
 * filter that guessed at "not a word in the language I expected" would hide
 * legitimate multilingual output.
 *
 * So this declutters a readout; it does not explain one. The junk bands seen on
 * gemma at L0-L3 and L12-L17 are mostly letter tokens and will still be there.
 */
export function isNonWord(token: string): boolean {
  const t = token.trim();
  if (!t) return true;
  if (/^<[^>]*>$/.test(t)) return true;
  if (/^\[[^\]]*\]$/.test(t)) return true;
  return !/\p{L}|\p{N}/u.test(t);
}

/** Layers of `axis` inside an inclusive absolute range. `null` means all. */
export function layersInRange(
  axis: number[],
  range: [number, number] | null,
): number[] {
  if (!range) return axis;
  const [lo, hi] = range[0] <= range[1] ? range : [range[1], range[0]];
  return axis.filter((l) => l >= lo && l <= hi);
}

/**
 * Rank the tokens a lens surfaces across the selected layers and every position.
 *
 * Sorted by count desc, ties broken by best rank — two tokens seen equally often
 * are not equally prominent if one of them was top-1.
 */
export function aggregate(
  tokens: LensTokenMessage[],
  type: string,
  axis: number[],
  range: [number, number] | null,
  opts: { hideNonWords: boolean } = { hideNonWords: true },
): AggregateResult {
  const inRange = new Set(layersInRange(axis, range));
  const acc = new Map<string, { count: number; layers: Set<number>; best: number }>();
  const hidden = new Set<string>();

  for (const tk of tokens) {
    const slice: LensTypeSlice | undefined = tk.results.find((s) => s.type === type);
    if (!slice) continue;

    axis.forEach((layer, layerIdx) => {
      // INDEXED BY POSITION IN THE AXIS, not by absolute layer number. The two
      // differ on any partial artifact, and indexing with the absolute number
      // reads the wrong row and produces a plausible list rather than an error.
      if (!inRange.has(layer)) return;
      const row = slice.top_tokens[layerIdx];
      if (!row) return;

      row.forEach((raw, rank) => {
        const token = raw;
        if (opts.hideNonWords && isNonWord(token)) {
          hidden.add(token);
          return;
        }
        const cur = acc.get(token);
        if (cur) {
          cur.count += 1;
          cur.layers.add(layer);
          if (rank < cur.best) cur.best = rank;
        } else {
          acc.set(token, { count: 1, layers: new Set([layer]), best: rank });
        }
      });
    });
  }

  const ranked = [...acc.entries()]
    .map(([token, v]) => ({
      token,
      count: v.count,
      layers: [...v.layers].sort((a, b) => a - b),
      bestRank: v.best,
    }))
    .sort((a, b) => b.count - a.count || a.bestRank - b.bestRank);

  return { ranked, hiddenNonWords: hidden.size };
}


/**
 * How to describe the layers a count was taken over, honestly.
 *
 * `L{first}–L{last}` overstates a NON-CONTIGUOUS axis: [8, 12, 20] reads as
 * "L8–L20", implying thirteen contributing layers when three contributed. A
 * partial fit produces exactly that shape, and the count next to it is then
 * read against a denominator four times too large.
 *
 * The COUNT leads because it is the denominator; the extent follows because it
 * locates the reading in the stack.
 */
export function describeSpan(axis: number[]): string {
  if (!axis.length) return 'no layers';
  if (axis.length === 1) return `L${axis[0]}`;
  const contiguous = axis.every((l, i) => i === 0 || l === axis[i - 1] + 1);
  if (contiguous) return `${axis.length} layers, L${axis[0]}\u2013L${axis[axis.length - 1]}`;
  const shown = axis.slice(0, 4).map((l) => `L${l}`).join(', ');
  return `${axis.length} layers (${shown}${axis.length > 4 ? ', \u2026' : ''})`;
}

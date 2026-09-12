/**
 * `dataset_weights` is POSITIONAL, and nothing on the backend validates the pairing.
 *
 * The worker builds its `extractions` list by iterating the `extraction_ids` it
 * was given, in order, then indexes `dataset_weights[i]` against
 * `extractions[i]`. A weight array built in a different order is not an error —
 * it trains on the wrong mixture, and the worker's log line reports the
 * REALISED split truthfully, which reads like confirmation that the request was
 * applied.
 *
 * So the ordering lives in one function and these tests are about that ordering
 * first and the values second.
 */

import { describe, it, expect } from 'vitest';
import {
  buildMixtureBlock,
  serialiseDatasetWeights,
} from './trainingMixture';

describe('serialiseDatasetWeights', () => {
  it('emits weights in extraction_ids order, NOT map insertion order', () => {
    // The map is deliberately built back-to-front. An implementation that
    // iterated Object.keys would pass a test whose map happened to agree.
    const weights = serialiseDatasetWeights(
      ['ext_web', 'ext_chat', 'ext_code'],
      { ext_code: 0.15, ext_chat: 0.25, ext_web: 0.6 }
    );
    expect(weights).toEqual([0.6, 0.25, 0.15]);
  });

  it('follows a reordering of the ids', () => {
    const byId = { ext_a: 0.9, ext_b: 0.1 };
    expect(serialiseDatasetWeights(['ext_a', 'ext_b'], byId)).toEqual([0.9, 0.1]);
    expect(serialiseDatasetWeights(['ext_b', 'ext_a'], byId)).toEqual([0.1, 0.9]);
  });

  it('defaults a source the operator never touched to 1', () => {
    expect(
      serialiseDatasetWeights(['ext_a', 'ext_b'], { ext_a: 3 })
    ).toEqual([3, 1]);
  });

  it('always produces one weight per id, never a shorter array', () => {
    const weights = serialiseDatasetWeights(
      ['a', 'b', 'c', 'd'],
      { b: 2 }
    );
    expect(weights).toHaveLength(4);
  });

  describe('when it should omit rather than send', () => {
    it('omits when no weights were set at all', () => {
      expect(serialiseDatasetWeights(['a', 'b'], undefined)).toBeUndefined();
    });

    it('omits an all-equal array, because uniform ≠ the default', () => {
      // Omitted means "proportional to real tokens". A uniform array means
      // "equal shares regardless of size". Someone who never touched the
      // control wants the former.
      expect(serialiseDatasetWeights(['a', 'b'], { a: 1, b: 1 })).toBeUndefined();
      expect(serialiseDatasetWeights(['a', 'b'], { a: 5, b: 5 })).toBeUndefined();
    });

    it('omits all-zero rather than asking the backend to divide by zero', () => {
      expect(serialiseDatasetWeights(['a', 'b'], { a: 0, b: 0 })).toBeUndefined();
    });

    it('omits when no extractions are selected', () => {
      expect(serialiseDatasetWeights([], { a: 1 })).toBeUndefined();
      expect(serialiseDatasetWeights(undefined, { a: 1 })).toBeUndefined();
    });
  });

  describe('unusable values', () => {
    it.each([NaN, Infinity, -1, undefined as unknown as number])(
      'replaces %p with 1 rather than sending it',
      (bad) => {
        const weights = serialiseDatasetWeights(['a', 'b'], { a: bad, b: 4 });
        expect(weights).toEqual([1, 4]);
      }
    );

    it('keeps a legitimate zero, which means "exclude this source"', () => {
      expect(serialiseDatasetWeights(['a', 'b'], { a: 0, b: 1 })).toEqual([0, 1]);
    });
  });
});

describe('buildMixtureBlock', () => {
  it('carries both fields when both are set', () => {
    expect(
      buildMixtureBlock({
        extractionIds: ['ext_web', 'ext_chat'],
        weightsByExtraction: { ext_web: 0.7, ext_chat: 0.3 },
        holdoutFraction: 0.05,
      })
    ).toEqual({ dataset_weights: [0.7, 0.3], holdout_fraction: 0.05 });
  });

  it('NEGATIVE CONTROL — an untouched form produces an EMPTY block', () => {
    // The historical request must be reproducible byte-for-byte, or every
    // existing training becomes incomparable for a reason nobody chose.
    expect(
      buildMixtureBlock({
        extractionIds: ['ext_web', 'ext_chat'],
        weightsByExtraction: undefined,
        holdoutFraction: 0,
      })
    ).toEqual({});
  });

  it.each([0, 1, 1.5, -0.1, NaN])(
    'omits holdout_fraction at %p, since the backend requires 0 <= f < 1 and 0 is the historical default',
    (value) => {
      const block = buildMixtureBlock({
        extractionIds: ['a'],
        weightsByExtraction: undefined,
        holdoutFraction: value,
      });
      expect('holdout_fraction' in block).toBe(false);
    }
  );

  it('accepts a holdout just under the limit', () => {
    expect(
      buildMixtureBlock({
        extractionIds: ['a'],
        weightsByExtraction: undefined,
        holdoutFraction: 0.99,
      })
    ).toEqual({ holdout_fraction: 0.99 });
  });

  it('the target corpus mixture serialises as intended', () => {
    // ~35% web / 25% chat / 15% code / 15% financial / 10% mixed
    const ids = ['ext_web', 'ext_chat', 'ext_code', 'ext_bloomberg', 'ext_pile'];
    const block = buildMixtureBlock({
      extractionIds: ids,
      weightsByExtraction: {
        ext_pile: 0.1,
        ext_bloomberg: 0.15,
        ext_code: 0.15,
        ext_chat: 0.25,
        ext_web: 0.35,
      },
      holdoutFraction: 0.02,
    });
    expect(block.dataset_weights).toEqual([0.35, 0.25, 0.15, 0.15, 0.1]);
    expect(block.dataset_weights!.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
    expect(block.holdout_fraction).toBe(0.02);
  });
});

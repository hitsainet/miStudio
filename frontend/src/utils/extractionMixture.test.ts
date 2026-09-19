/**
 * The mixture builder for feature extraction.
 *
 * The load-bearing property is ORDER: `dataset_weights` is positional over
 * `dataset_ids`, and a mis-ordered pair is not an error — it silently extracts
 * the wrong mixture. So the map here is deliberately built BACK TO FRONT in the
 * ordering test: an implementation that iterated `Object.keys(weights)` instead
 * of `datasetIds` would pass a same-order fixture and fail this one.
 */

import { describe, it, expect } from 'vitest';

import { buildExtractionMixture, primaryDatasetId } from './extractionMixture';

describe('buildExtractionMixture', () => {
  it('sends dataset_ids in the order given', () => {
    const block = buildExtractionMixture({
      datasetIds: ['ds_code', 'ds_web', 'ds_chat'],
      weightsByDataset: undefined,
    });

    expect(block.dataset_ids).toEqual(['ds_code', 'ds_web', 'ds_chat']);
  });

  it('orders weights by datasetIds, not by the map insertion order', () => {
    // Built back-to-front on purpose: an Object.keys() implementation returns
    // ['ds_chat','ds_web','ds_code'] here and produces [0.1, 0.3, 0.6].
    const weightsByDataset: Record<string, number> = {};
    weightsByDataset['ds_chat'] = 0.1;
    weightsByDataset['ds_web'] = 0.3;
    weightsByDataset['ds_code'] = 0.6;

    const block = buildExtractionMixture({
      datasetIds: ['ds_code', 'ds_web', 'ds_chat'],
      weightsByDataset,
    });

    expect(block.dataset_weights).toEqual([0.6, 0.3, 0.1]);
  });

  it('omits dataset_weights when every weight is equal', () => {
    // All-equal carries nothing the server default does not already express,
    // and sending it would claim a preference the operator never expressed.
    const block = buildExtractionMixture({
      datasetIds: ['a', 'b', 'c'],
      weightsByDataset: { a: 1, b: 1, c: 1 },
    });

    expect(block.dataset_weights).toBeUndefined();
    expect(block.dataset_ids).toEqual(['a', 'b', 'c']);
  });

  it('omits dataset_weights when no weights were supplied at all', () => {
    const block = buildExtractionMixture({
      datasetIds: ['a', 'b'],
      weightsByDataset: undefined,
    });

    expect(block.dataset_weights).toBeUndefined();
  });

  it('treats a missing per-dataset weight as 1 rather than dropping the slot', () => {
    const block = buildExtractionMixture({
      datasetIds: ['a', 'b', 'c'],
      weightsByDataset: { a: 2, c: 3 },
    });

    expect(block.dataset_weights).toEqual([2, 1, 3]);
  });

  it('omits weights that are all zero, which would ask for an empty corpus', () => {
    const block = buildExtractionMixture({
      datasetIds: ['a', 'b'],
      weightsByDataset: { a: 0, b: 0 },
    });

    expect(block.dataset_weights).toBeUndefined();
  });

  it('still sends dataset_ids for a single dataset, with no weights', () => {
    const block = buildExtractionMixture({
      datasetIds: ['only'],
      weightsByDataset: { only: 0.5 },
    });

    expect(block.dataset_ids).toEqual(['only']);
    // One source has no mixture to express.
    expect(block.dataset_weights).toBeUndefined();
  });

  it('returns an empty block when nothing is selected', () => {
    const block = buildExtractionMixture({ datasetIds: [], weightsByDataset: { a: 1 } });

    expect(block).toEqual({});
  });

  it('does not alias the caller’s array', () => {
    const ids = ['a', 'b'];
    const block = buildExtractionMixture({ datasetIds: ids, weightsByDataset: undefined });

    ids.push('c');
    expect(block.dataset_ids).toEqual(['a', 'b']);
  });
});

describe('primaryDatasetId', () => {
  it('is the first selected id, matching what the server stores as dataset_id', () => {
    expect(primaryDatasetId(['ds_code', 'ds_web'])).toBe('ds_code');
  });

  it('is undefined when nothing is selected', () => {
    expect(primaryDatasetId([])).toBeUndefined();
  });
});

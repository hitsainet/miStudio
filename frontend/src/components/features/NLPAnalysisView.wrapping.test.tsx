/**
 * The AI-generated feature summary must wrap, not overflow the NLP tab.
 *
 * `summary_for_prompt` is written by the labeling model, so it can arrive as one
 * unbroken run — a repeated token with no space, a pasted identifier, a
 * degenerate continuation. The summary block had `whitespace-pre-wrap` and
 * nothing else, and pre-wrap only wraps where a break opportunity already
 * exists.
 *
 * It matters here specifically because the NLP tab renders inside
 * FeatureDetailModal's `flex-1 overflow-y-auto` pane: per spec `overflow-y:
 * auto` makes `overflow-x` compute to `auto` too, so an unbroken summary grows
 * that pane its own horizontal scrollbar rather than being clipped.
 *
 * MUTATION CONTROL: recorded in the commit message.
 */

import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { NLPAnalysisView } from './NLPAnalysisView';
import type { NLPAnalysis } from '../../types/features';
import { expectWrapsLongRuns } from '../../test/expectWrapsLongRuns';

/** A degenerate summary: newline structure AND a run with no space in it. */
const SUMMARY =
  'Activates on financial risk language.\n' + 'RISKRISKRISK'.repeat(30);

const analysis: NLPAnalysis = {
  prime_token_analysis: {
    unique_count: 2,
    total_count: 4,
    unique_tokens: ['risk', 'loss'],
    frequency_distribution: { risk: 3, loss: 1 },
    lowercase_distribution: { risk: 3, loss: 1 },
    pos_distribution: { NOUN: 4 },
    ner_entities: [],
    token_types: { word: 4 },
    most_common_token: ['risk', 3],
    concentration_ratio: 0.75,
  },
  context_patterns: {
    prefix_bigrams: [],
    prefix_trigrams: [],
    suffix_bigrams: [],
    suffix_trigrams: [],
    immediately_before: {},
    immediately_after: {},
    syntactic_patterns: [],
  },
  activation_stats: {
    mean: 1.2,
    std: 0.4,
    min: 0.1,
    max: 3.3,
    median: 1.1,
    skewness: 0.2,
    distribution_type: 'symmetric',
    high_activation_tokens: [],
    activation_range_buckets: {},
    coefficient_of_variation: 0.33,
  },
  semantic_clusters: [],
  summary_for_prompt: SUMMARY,
  num_examples_analyzed: 20,
  computed_at: '2026-09-27T00:00:00Z',
};

describe('the NLP summary wraps a space-free run', () => {
  it('breaks inside the run instead of scrolling the NLP tab sideways', () => {
    render(
      <NLPAnalysisView
        nlpAnalysis={analysis}
        nlpProcessedAt="2026-09-27T00:00:00Z"
        featureId="f_1"
      />,
    );

    // Found by its text, so the assertion cannot be satisfied by the selector.
    // The identity normalizer matters: the default collapses the newline that
    // pre-wrap is being kept for, and then nothing matches.
    const summary = screen.getByText(SUMMARY, { normalizer: (t) => t });
    expect(summary.tagName).toBe('P');
    expectWrapsLongRuns(summary);
  });
});

/**
 * Feature 012 regression test: blended batch results must render their BAKED
 * title (feature_config.label from the adapter), never a live selectedFeatures
 * lookup — otherwise relabeling/removing features retitles finished results,
 * and cluster provenance never reaches the batch UI at all.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ComparisonResults } from './ComparisonResults';
import { useSteeringStore } from '../../stores/steeringStore';
import type { BatchPromptResult } from '../../types/steering';

const blendedBatchResult: BatchPromptResult = {
  prompt: 'The old lighthouse keeper retired.',
  promptIndex: 0,
  status: 'completed',
  error: null,
  comparison: {
    comparison_id: 'cmb_test1',
    sae_id: 'sae-1',
    model_id: 'm-1',
    prompt: 'The old lighthouse keeper retired.',
    unsteered: { text: 'baseline text', metrics: null },
    steered: [
      {
        text: 'blended output text',
        feature_config: {
          instance_id: 'x',
          feature_idx: 10091,
          layer: 12,
          strength: 0.8,
          label: 'fear — Blended (3 features)', // baked by combinedToComparison
          color: 'teal',
          feature_id: null,
        },
        metrics: null,
      },
    ],
    steered_multi: null,
    applied_features: [
      { feature_idx: 10091, layer: 12, strength: 0.8, label: null, color: 'teal' },
      { feature_idx: 2262, layer: 12, strength: 0.7, label: null, color: 'blue' },
      { feature_idx: 9157, layer: 12, strength: 0.6, label: null, color: 'purple' },
    ],
    metrics_summary: null,
    total_time_ms: 1000,
    created_at: new Date().toISOString(),
  },
};

describe('ComparisonResults blended titles (Feature 012)', () => {
  beforeEach(() => {
    // A live selection whose labels MUST NOT leak into the finished result:
    useSteeringStore.setState({
      selectedFeatures: [
        {
          instance_id: 'live1',
          feature_idx: 10091,
          layer: 12,
          strength: 5,
          label: 'LIVE LABEL SHOULD NOT RENDER',
          color: 'teal',
          feature_id: null,
        },
      ],
    } as never);
  });

  it('renders the baked cluster title on blended batch cards', () => {
    render(<ComparisonResults batchResults={[blendedBatchResult]} />);
    expect(screen.getByText('fear — Blended (3 features)')).toBeInTheDocument();
    expect(screen.queryByText('LIVE LABEL SHOULD NOT RENDER')).not.toBeInTheDocument();
    expect(screen.queryByText('Feature #10091')).not.toBeInTheDocument();
  });

  it('shows the applied-features expander on blended batch cards', () => {
    render(<ComparisonResults batchResults={[blendedBatchResult]} />);
    expect(screen.getByRole('button', { name: /Applied features \(3\)/ })).toBeInTheDocument();
  });
});

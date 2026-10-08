/**
 * The feature modal names the score for what it measures.
 *
 * It was labelled "Interpretability". The value is activation consistency:
 * similarity of activation SHAPE across a feature's top examples, which never
 * inspects which tokens fire and is highest for evidence repeating one token.
 *
 * MUTATION CONTROLS:
 *   M1  restore the "Interpretability" label      -> the label test fails
 *   M2  read only interpretability_score          -> the new-field test fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

vi.mock('../../hooks/useEnhancedLabeling', () => ({
  useEnhancedLabeling: () => ({ isRunning: false, start: vi.fn(), progress: null }),
}));

import { FeatureDetailModal } from './FeatureDetailModal';
import { useFeaturesStore } from '../../stores/featuresStore';

function seed(feature: Record<string, unknown>) {
  const now = new Date().toISOString();
  useFeaturesStore.setState({
    selectedFeature: {
      id: 'feat_1', neuron_index: 7, name: 'a label', description: null,
      label_source: 'auto', activation_frequency: 0.4, max_activation: 5,
      mean_activation: 2, is_favorite: false, star_color: null, notes: null,
      created_at: now, updated_at: now,
      ...feature,
    },
    featureExamples: [],
    fetchFeatureDetail: vi.fn().mockResolvedValue(undefined),
    fetchFeatureExamples: vi.fn().mockResolvedValue(undefined),
  } as never);
}

describe('FeatureDetailModal — activation consistency', () => {
  beforeEach(() => vi.clearAllMocks());

  it('labels the score as activation consistency, not interpretability', () => {
    seed({ interpretability_score: 0.714, activation_consistency: 0.714 });
    render(<FeatureDetailModal featureId="feat_1" trainingId={null} onClose={vi.fn()} />);

    expect(screen.getByText('Activation consistency')).toBeInTheDocument();
    expect(screen.queryByText('Interpretability')).not.toBeInTheDocument();
  });

  it('reads the new field', () => {
    // Different values, so reading the deprecated field shows a different number.
    seed({ interpretability_score: 0.1, activation_consistency: 0.714 });
    render(<FeatureDetailModal featureId="feat_1" trainingId={null} onClose={vi.fn()} />);

    expect(screen.getByText('71.4%')).toBeInTheDocument();
  });

  it('falls back to the deprecated field for older responses', () => {
    seed({ interpretability_score: 0.5 });
    render(<FeatureDetailModal featureId="feat_1" trainingId={null} onClose={vi.fn()} />);

    expect(screen.getByText('50.0%')).toBeInTheDocument();
  });
});

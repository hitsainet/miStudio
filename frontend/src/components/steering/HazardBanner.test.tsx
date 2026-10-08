/**
 * Tests for HazardBanner (Feature 015, IDL-35 copy discipline).
 *
 * The banner renders cross-layer steering hazards from the store. Its central
 * contract is COPY DISCIPLINE: a validated (rung ≥ 2) hazard may read as
 * established and show its ES; a heuristic (weight-prior) hazard MUST label
 * itself a heuristic and MUST NEVER read as causal. It is a warning only and
 * dismissible per selection (dismissal resets when the selection changes).
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { HazardBanner } from './HazardBanner';
import { useSteeringStore } from '../../stores/steeringStore';
import type { Hazard, SelectedFeature } from '../../types/steering';

const feature = (idx: number, layer: number): SelectedFeature =>
  ({
    instance_id: `i-${idx}-${layer}`,
    feature_idx: idx,
    layer,
    strength: 1,
    color: 'teal',
    label: null,
    feature_id: null,
  }) as SelectedFeature;

const validatedHazard: Hazard = {
  type: 'compounding',
  up: { layer: 13, feature_idx: 100 },
  down: { layer: 14, feature_idx: 200 },
  evidence: 'validated:ES=0.80',
  rung: 3,
  quantified_effect: 0.8,
};

const heuristicHazard: Hazard = {
  type: 'compounding',
  up: { layer: 13, feature_idx: 300 },
  down: { layer: 14, feature_idx: 400 },
  evidence: 'heuristic:weight_prior=0.62',
  rung: 0,
  quantified_effect: null,
};

describe('HazardBanner', () => {
  beforeEach(() => {
    useSteeringStore.setState({
      hazards: null,
      selectedFeatures: [],
    });
  });

  it('renders nothing without hazards', () => {
    const { container } = render(<HazardBanner />);
    expect(container).toBeEmptyDOMElement();
  });

  it('renders a validated hazard with its ES', () => {
    useSteeringStore.setState({
      hazards: [validatedHazard],
      selectedFeatures: [feature(100, 13), feature(200, 14)],
    });
    render(<HazardBanner />);
    // Layer-labeled pair
    expect(screen.getByText(/L13 #100/)).toBeInTheDocument();
    expect(screen.getByText(/L14 #200/)).toBeInTheDocument();
    // Reads as validated and shows the ES (specific headline phrasing).
    expect(screen.getByText(/compounding \(validated, ES=0\.80\)/)).toBeInTheDocument();
  });

  it('renders a heuristic hazard WITH "heuristic" and WITHOUT "causal"', () => {
    useSteeringStore.setState({
      hazards: [heuristicHazard],
      selectedFeatures: [feature(300, 13), feature(400, 14)],
    });
    const { container } = render(<HazardBanner />);
    // Must label itself a heuristic (headline + raw evidence both carry it).
    expect(screen.getAllByText(/heuristic/).length).toBeGreaterThan(0);
    // The human headline says "weight prior" (space-separated, non-causal).
    expect(screen.getByText(/possible compounding \(heuristic — weight prior/)).toBeInTheDocument();
    // Must NEVER read as causal
    expect(container.textContent?.toLowerCase()).not.toContain('causal');
  });

  it('is dismissible for the current selection', () => {
    useSteeringStore.setState({
      hazards: [validatedHazard],
      selectedFeatures: [feature(100, 13), feature(200, 14)],
    });
    const { container } = render(<HazardBanner />);
    expect(screen.getByText(/L13 #100/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Dismiss hazard warning/ }));
    expect(container).toBeEmptyDOMElement();
  });
});

/**
 * Tests for ClusterBudgetBar (Feature 013).
 *
 * The bar is the budget-model's visibility surface: it must show consumption
 * against B, surface coherence flags, warn when over budget, and — when the
 * model is gated (low cohesion) — show ONLY the notice, never a budget.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ClusterBudgetBar } from './ClusterBudgetBar';
import { useSteeringStore } from '../../stores/steeringStore';

const budget = {
  B: 2.4,
  B_dir: 2.4,
  G: 0.87,
  flags: [] as string[],
  approximate: false,
  weightsByInstance: {} as Record<string, number>,
};

const feature = (idx: number, strength: number) =>
  ({
    feature_idx: idx,
    layer: 6,
    strength,
    color: 'teal',
    instance_id: `i-${idx}`,
    strengthSource: 'cluster',
  }) as never;

describe('ClusterBudgetBar', () => {
  beforeEach(() => {
    useSteeringStore.setState({
      clusterBudget: null,
      clusterNotice: null,
      selectedFeatures: [],
    });
  });

  it('renders nothing without a budget or notice', () => {
    const { container } = render(<ClusterBudgetBar />);
    expect(container).toBeEmptyDOMElement();
  });

  it('shows consumption / B and the gain G', () => {
    useSteeringStore.setState({
      clusterBudget: budget,
      selectedFeatures: [feature(1, 1.2), feature(2, -0.6)],
    });
    render(<ClusterBudgetBar />);
    // |1.2| + |−0.6| = 1.8 of 2.4
    expect(screen.getByText(/1\.8 \/ 2\.4/)).toBeInTheDocument();
    expect(screen.getByText(/G 0\.87/)).toBeInTheDocument();
  });

  it('turns amber when over budget', () => {
    useSteeringStore.setState({
      clusterBudget: budget,
      selectedFeatures: [feature(1, 2.0), feature(2, 1.0)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/3\.0 \/ 2\.4/)).toHaveClass('text-amber-400');
  });

  it('surfaces coherence flags with their copy', () => {
    useSteeringStore.setState({
      clusterBudget: { ...budget, flags: ['cancellation', 'approximate'] },
      selectedFeatures: [feature(1, 1.0)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/partially cancel/)).toBeInTheDocument();
    expect(screen.getByText(/constant-budget approximation/)).toBeInTheDocument();
  });

  it('ignores unknown flags from a future server', () => {
    useSteeringStore.setState({
      clusterBudget: { ...budget, flags: ['grain_limited', 'brand_new_flag'] },
      selectedFeatures: [feature(1, 1.0)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.queryByText(/brand_new_flag/)).not.toBeInTheDocument();
  });

  it('gated cluster: renders the notice only, no budget numbers', () => {
    useSteeringStore.setState({
      clusterBudget: null,
      clusterNotice: 'Low cluster cohesion — kept per-feature baselines (budget model gated)',
      selectedFeatures: [feature(1, 10)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/Low cluster cohesion/)).toBeInTheDocument();
    expect(screen.queryByText(/Cluster budget/)).not.toBeInTheDocument();
  });
});

// Feature 015: per-layer budget bars (multi-layer cross-SAE steering).
const layerFeature = (idx: number, layer: number, strength: number) =>
  ({
    feature_idx: idx,
    layer,
    strength,
    color: 'teal',
    instance_id: `i-${idx}-${layer}`,
    strengthSource: 'cluster',
  }) as never;

const layerBudget = (B: number, G: number) => ({
  B,
  B_dir: B,
  G,
  flags: [] as string[],
  approximate: false,
  weightsByInstance: {} as Record<string, number>,
});

describe('ClusterBudgetBar — multi-layer (Feature 015)', () => {
  beforeEach(() => {
    useSteeringStore.setState({
      clusterBudget: null,
      layerBudgets: null,
      hazards: null,
      clusterNotice: null,
      selectedFeatures: [],
    });
  });

  it('renders one budget row per layer with layer chips and per-layer B/λ', () => {
    useSteeringStore.setState({
      // Multi-layer: clusterBudget stays null; layerBudgets governs.
      clusterBudget: null,
      layerBudgets: { 13: layerBudget(2.4, 0.9), 14: layerBudget(3.0, 0.85) },
      selectedFeatures: [
        layerFeature(100, 13, 1.2),
        layerFeature(200, 13, 0.6),
        layerFeature(300, 14, 2.0),
      ],
    });
    render(<ClusterBudgetBar />);

    // Layer chips, one per layer.
    expect(screen.getByText('L13')).toBeInTheDocument();
    expect(screen.getByText('L14')).toBeInTheDocument();

    // Per-layer consumption: L13 = |1.2|+|0.6| = 1.8 / 2.4; L14 = 2.0 / 3.0.
    expect(screen.getByText(/1\.8 \/ 2\.4/)).toBeInTheDocument();
    expect(screen.getByText(/2\.0 \/ 3\.0/)).toBeInTheDocument();

    // Each layer shows its own gain G.
    expect(screen.getByText(/G 0\.90/)).toBeInTheDocument();
    expect(screen.getByText(/G 0\.85/)).toBeInTheDocument();
  });

  it('single-layer layerBudgets (mirrored) still renders the classic single bar', () => {
    // When only one layer is present, the mirrored clusterBudget drives the
    // classic single-bar path (byte-identical to 013).
    useSteeringStore.setState({
      clusterBudget: layerBudget(2.4, 0.87),
      layerBudgets: { 6: layerBudget(2.4, 0.87) },
      selectedFeatures: [layerFeature(1, 6, 1.2), layerFeature(2, 6, -0.6)],
    });
    render(<ClusterBudgetBar />);
    expect(screen.getByText(/Cluster budget/)).toBeInTheDocument();
    expect(screen.getByText(/1\.8 \/ 2\.4/)).toBeInTheDocument();
    // No per-layer chip in the single-layer path.
    expect(screen.queryByText('L6')).not.toBeInTheDocument();
  });
});

/**
 * 015 FPRD §8.4 — "rebalance stays within-layer", through the panel.
 *
 * The store test (`steeringStore.test.ts`, "rebalance within one layer when two
 * layers are selected") pins `rebalance(layer, …)`. This file pins the line that
 * CALLS it: a strength edit on a member of a two-layer selection must reach the
 * store as `rebalance(<that member's layer>, <its instance>, <value>)`, once.
 *
 * MUTATION CONTROLS (run 2026-10-21, each restored by sha256): see
 * 0xcc/tasks/015_FTASKS|MultiSAE_Steering.md, Task 6.2.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, screen } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

vi.mock('axios', () => ({
  default: { get: vi.fn().mockResolvedValue({ data: {} }), post: vi.fn().mockResolvedValue({ data: {} }) },
}));
vi.mock('../../hooks/useEnhancedLabeling', () => ({
  useEnhancedLabeling: () => ({ isRunning: false, start: vi.fn(), progress: null }),
}));
vi.mock('./ClusterBudgetBar', () => ({ ClusterBudgetBar: () => null }));
vi.mock('./ProfilesMenu', () => ({ ProfilesMenu: () => null }));
vi.mock('./SaveProfileDialog', () => ({ SaveProfileDialog: () => null }));
vi.mock('./FeatureBrowser', () => ({ FeatureBrowser: () => null }));

import { FeatureSelector } from './FeatureSelector';
import { useSteeringStore } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { SAEStatus } from '../../types/sae';

const sae = {
  id: 'sae-l13', name: 'L13 SAE', status: SAEStatus.READY, training_id: null,
  layer: 13, hook_type: 'resid_post', n_features: 8192, model_name: 'm',
} as never;

const member = (id: string, idx: number, layer: number, strength: number) => ({
  instance_id: id, feature_idx: idx, layer, strength, label: `f${idx}`, color: 'teal',
  feature_id: `feat_${idx}`, sae_id: layer === 13 ? 'sae-l13' : 'sae-l14',
  strengthSource: 'cluster', pinned: false,
});

const budget = (B: number, weights: Record<string, number>) => ({
  B, B_dir: B, G: 1, flags: [], approximate: false, weightsByInstance: weights,
});

const realRebalance = useSteeringStore.getState().rebalance;

function seed() {
  useSAEsStore.setState({ saes: [sae] } as never);
  useSteeringStore.setState({
    selectedSAE: sae,
    selectedFeatures: [
      member('i13a', 100, 13, 1.5),
      member('i13b', 101, 13, 0.5),
      member('i14a', 200, 14, 1.75),
      member('i14b', 201, 14, 1.25),
    ],
    clusterBudget: null,
    layerBudgets: {
      13: budget(2.0, { i13a: 0.75, i13b: 0.25 }),
      14: budget(3.0, { i14a: 0.6, i14b: 0.4 }),
    },
    clusterContext: null,
    rebalance: realRebalance,
  } as never);
}

function editStrength(current: string, next: string) {
  const input = screen.getByDisplayValue(current);
  fireEvent.focus(input);
  fireEvent.change(input, { target: { value: next } });
  fireEvent.blur(input);
}

const strengths = () =>
  Object.fromEntries(useSteeringStore.getState().selectedFeatures.map((f) => [f.instance_id, f.strength]));

describe('FeatureSelector — two-layer selection rebalances within the edited layer', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seed();
  });

  it('an L14 edit calls rebalance once with layer 14, that member and the value', () => {
    const spy = vi.fn();
    useSteeringStore.setState({ rebalance: spy } as never);
    render(<FeatureSelector />);
    editStrength('1.75', '1');
    expect(spy).toHaveBeenCalledTimes(1);
    expect(spy).toHaveBeenCalledWith(14, 'i14a', 1);
  });

  it('end to end through the real store: L14 rebalances to its budget, L13 is untouched', () => {
    render(<FeatureSelector />);
    editStrength('1.75', '1');
    expect(strengths()).toEqual({ i13a: 1.5, i13b: 0.5, i14a: 1, i14b: 2 });
  });
});

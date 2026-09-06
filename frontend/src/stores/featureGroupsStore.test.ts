/**
 * Tests for featureGroupsStore selection stamping + provenance derivation
 * (Features 012/013). The stamps are the foundation the steering hand-off's
 * cluster identity trusts — mixed or unstamped selections must never yield a
 * cluster claim.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { act } from '@testing-library/react';
import {
  useFeatureGroupsStore,
  deriveSourceCluster,
  type SelectedMember,
} from './featureGroupsStore';

const member = (over: Partial<SelectedMember> = {}): SelectedMember => ({
  neuron_index: 1,
  max_activation: 2.5,
  activation_frequency: 0.2,
  group_id: 'gA',
  display_token: 'fear',
  similarity: 0.8,
  cohesion: 0.7,
  ...over,
});

describe('deriveSourceCluster', () => {
  it('returns the cluster for a clean single-cluster selection', () => {
    expect(deriveSourceCluster([member(), member({ neuron_index: 2 })])).toEqual({
      group_id: 'gA',
      display_token: 'fear',
    });
  });

  it('returns null for mixed-cluster selections (US-5)', () => {
    expect(
      deriveSourceCluster([member(), member({ group_id: 'gB', display_token: 'joy' })]),
    ).toBeNull();
  });

  it('returns null for empty, unstamped, or blank-token selections', () => {
    expect(deriveSourceCluster([])).toBeNull();
    expect(deriveSourceCluster([member({ group_id: '' as never })])).toBeNull();
    expect(deriveSourceCluster([member({ display_token: '   ' })])).toBeNull();
  });

  it('trims the display token', () => {
    expect(deriveSourceCluster([member({ display_token: ' fear ' })])?.display_token).toBe('fear');
  });
});

describe('featureGroupsStore selection stamping', () => {
  beforeEach(() => {
    act(() => {
      useFeatureGroupsStore.setState({ selection: new Map() });
    });
  });

  it('toggleSelect stores the full stamp and toggles off', () => {
    const store = useFeatureGroupsStore.getState();
    act(() => store.toggleSelect('f1', member()));
    let sel = useFeatureGroupsStore.getState().selection;
    expect(sel.get('f1')).toMatchObject({
      group_id: 'gA',
      display_token: 'fear',
      similarity: 0.8,
      cohesion: 0.7,
    });
    act(() => useFeatureGroupsStore.getState().toggleSelect('f1', member()));
    sel = useFeatureGroupsStore.getState().selection;
    expect(sel.has('f1')).toBe(false);
  });

  it('setSelected bulk-stamps and bulk-clears', () => {
    const store = useFeatureGroupsStore.getState();
    const rows = [
      { feature_id: 'f1', ...member({ neuron_index: 1 }) },
      { feature_id: 'f2', ...member({ neuron_index: 2 }) },
    ];
    act(() => store.setSelected(rows, true));
    const sel = useFeatureGroupsStore.getState().selection;
    expect(sel.size).toBe(2);
    expect(sel.get('f2')?.neuron_index).toBe(2);
    expect(sel.get('f2')?.group_id).toBe('gA');
    act(() => useFeatureGroupsStore.getState().setSelected(rows, false));
    expect(useFeatureGroupsStore.getState().selection.size).toBe(0);
  });

  it('cross-cluster accumulation keeps distinct stamps (derivation then rejects)', () => {
    const store = useFeatureGroupsStore.getState();
    act(() => {
      store.toggleSelect('f1', member());
      useFeatureGroupsStore.getState().toggleSelect(
        'f2',
        member({ group_id: 'gB', display_token: 'joy' }),
      );
    });
    const members = [...useFeatureGroupsStore.getState().selection.values()];
    expect(deriveSourceCluster(members)).toBeNull();
  });
});

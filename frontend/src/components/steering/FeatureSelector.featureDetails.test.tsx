/**
 * "View Feature Details" works for BOTH kinds of SAE.
 *
 * The context-menu item was disabled for downloaded SAEs (`!selectedSAE?.training_id`)
 * with "Feature details not available for downloaded SAEs", although the handler
 * already looked a downloaded SAE's feature up by its own id
 * (GET /api/v1/saes/{sae_id}/features/by-index/{idx}). The details modal reads a
 * feature by feature_id alone, so it renders for either kind once the id is found.
 *
 * MUTATION CONTROLS (run 2026-10-18, each restored by sha256):
 *   M1  restore `disabled={!selectedSAE?.training_id}` on the menu button  -> 3 red
 *   M2  replace the `/saes/${saeId}/features/by-index/` call with a null    -> 2 red
 *   M3  drop the "Could not look up" alert (a failed lookup then reads as
 *       "not extracted yet")                                                -> 1 red
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { fireEvent, screen, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

vi.mock('axios', () => ({
  default: { get: vi.fn(), post: vi.fn().mockResolvedValue({ data: {} }) },
}));
vi.mock('../../hooks/useEnhancedLabeling', () => ({
  useEnhancedLabeling: () => ({ isRunning: false, start: vi.fn(), progress: null }),
}));
// Panels unrelated to the context menu, which fetch on mount.
vi.mock('./ClusterBudgetBar', () => ({ ClusterBudgetBar: () => null }));
vi.mock('./ProfilesMenu', () => ({ ProfilesMenu: () => null }));
vi.mock('./SaveProfileDialog', () => ({ SaveProfileDialog: () => null }));
vi.mock('./FeatureBrowser', () => ({ FeatureBrowser: () => null }));

import axios from 'axios';
import { FeatureSelector } from './FeatureSelector';
import { useSteeringStore } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { useFeaturesStore } from '../../stores/featuresStore';
import { SAEStatus } from '../../types/sae';

const get = axios.get as unknown as ReturnType<typeof vi.fn>;

function sae(trainingId: string | null) {
  return {
    id: 'sae_dl_1',
    name: 'Downloaded SAE',
    status: SAEStatus.READY,
    training_id: trainingId,
    layer: 12,
    hook_type: 'resid_post',
    n_features: 16384,
    model_name: 'gemma-2-2b',
  } as never;
}

function seed(trainingId: string | null) {
  const s = sae(trainingId);
  useSAEsStore.setState({ saes: [s] } as never);
  useSteeringStore.setState({
    selectedSAE: s,
    selectedFeatures: [
      {
        instance_id: 'inst_1',
        feature_idx: 42,
        layer: 12,
        strength: 10,
        label: 'feature 42',
        color: 'teal',
        feature_id: null, // added by index: no id yet, so the lookup must run
      },
    ],
    clusterBudget: null,
    layerBudgets: null,
    clusterContext: null,
  } as never);

  const now = new Date().toISOString();
  const fetchFeatureDetail = vi.fn().mockImplementation(async (id: string) => {
    useFeaturesStore.setState({
      selectedFeature: {
        id, neuron_index: 42, name: 'detail of the looked-up feature', description: null,
        label_source: 'auto', activation_frequency: 0.1, max_activation: 3,
        mean_activation: 1, is_favorite: false, star_color: null, notes: null,
        created_at: now, updated_at: now,
      },
    } as never);
  });
  useFeaturesStore.setState({
    selectedFeature: null,
    featureExamples: [],
    fetchFeatureDetail,
    fetchFeatureExamples: vi.fn().mockResolvedValue(undefined),
    clearSelectedFeature: vi.fn(),
  } as never);
  return fetchFeatureDetail;
}

function openMenuAndView() {
  fireEvent.contextMenu(screen.getByText('feature 42'));
  const item = screen.getByRole('button', { name: /View Feature Details/ });
  expect(item).not.toBeDisabled();
  fireEvent.click(item);
}

describe('FeatureSelector — View Feature Details', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.spyOn(window, 'alert').mockImplementation(() => undefined);
  });

  it('downloaded SAE: looks the feature up by SAE id and opens the details view', async () => {
    const fetchFeatureDetail = seed(null);
    get.mockResolvedValue({ data: { feature_id: 'feat_sae_sae_dl_1_42' } });

    render(<FeatureSelector />);
    openMenuAndView();

    await waitFor(() =>
      expect(screen.getByText('detail of the looked-up feature')).toBeInTheDocument(),
    );
    expect(get).toHaveBeenCalledTimes(1);
    expect(get).toHaveBeenCalledWith('/api/v1/saes/sae_dl_1/features/by-index/42');
    expect(fetchFeatureDetail).toHaveBeenCalledWith('feat_sae_sae_dl_1_42');
    expect(window.alert).not.toHaveBeenCalled();
  });

  it('downloaded SAE: the menu no longer says details are unavailable', () => {
    seed(null);
    render(<FeatureSelector />);
    fireEvent.contextMenu(screen.getByText('feature 42'));
    expect(screen.queryByText(/not available for downloaded SAEs/)).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /View Feature Details/ })).not.toBeDisabled();
  });

  it('trained SAE: looks the feature up by training id and opens the details view', async () => {
    const fetchFeatureDetail = seed('train_abc');
    get.mockResolvedValue({ data: { feature_id: 'feat_train_abc_42' } });

    render(<FeatureSelector />);
    openMenuAndView();

    await waitFor(() =>
      expect(screen.getByText('detail of the looked-up feature')).toBeInTheDocument(),
    );
    expect(get).toHaveBeenCalledTimes(1);
    expect(get).toHaveBeenCalledWith('/api/v1/trainings/train_abc/features/by-index/42');
    expect(fetchFeatureDetail).toHaveBeenCalledWith('feat_train_abc_42');
  });

  it('a feature with no extraction says so, and a failed lookup says it failed', async () => {
    seed(null);
    get.mockResolvedValueOnce({ data: { feature_id: null } });
    render(<FeatureSelector />);
    openMenuAndView();
    await waitFor(() => expect(window.alert).toHaveBeenCalledTimes(1));
    expect((window.alert as unknown as ReturnType<typeof vi.fn>).mock.calls[0][0]).toMatch(
      /has not been extracted yet/,
    );

    get.mockRejectedValueOnce(new Error('Request failed with status code 500'));
    openMenuAndView();
    await waitFor(() => expect(window.alert).toHaveBeenCalledTimes(2));
    expect((window.alert as unknown as ReturnType<typeof vi.fn>).mock.calls[1][0]).toMatch(
      /Could not look up feature #42: Request failed with status code 500/,
    );
  });
});

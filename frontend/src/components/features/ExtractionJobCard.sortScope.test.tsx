/**
 * Dataset-wide sorting ("all" scope) asks the server for the order.
 *
 * The state's comment said "'all' not yet implemented - requires backend
 * support" while the handler already sent sort_by/sort_order to
 * GET /extractions/{id}/features. What WAS wrong: Label and Category were mapped
 * to `feature_id`, although the route sorts by `name` and `category`, so a
 * dataset-wide Label sort came back in ID order under a Label arrow.
 *
 * MUTATION CONTROLS (2026-10-18, each restored by sha256):
 *   S1  map label back to 'feature_id' in SERVER_SORT                -> 1 red
 *   S3  make every header sortable in "all" scope                    -> 1 red
 *   (S2, a second `if (!serverSort) return` guard inside handleSort, SURVIVED:
 *    no UI path reaches it once the header is not clickable, so it was removed
 *    rather than kept as untested code. S3 pins the one gate that matters.)
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

vi.mock('../../api/models', () => ({
  triggerNlpAnalysis: vi.fn(),
  cancelNlpAnalysis: vi.fn(),
  resetNlpAnalysis: vi.fn(),
}));

const FEATURES = [0, 1].map((i) => ({
  id: `feat_${i}`, neuron_index: i, name: `feature ${i}`, category: null,
  description: null, label_source: 'auto', activation_frequency: 0.1 * (i + 1),
  max_activation: 2 + i, mean_activation: 1, is_favorite: false, star_color: null,
  notes: null, created_at: '', updated_at: '',
}));

vi.mock('axios', () => ({
  default: {
    get: vi.fn(),
    post: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

import axios from 'axios';
import { ExtractionJobCard } from './ExtractionJobCard';
import { useFeaturesStore } from '../../stores/featuresStore';

const get = vi.mocked(axios.get);

const completed = {
  id: 'extr_sort', status: 'completed', progress: 1, features_extracted: 2,
  total_features: 2, sae_name: 'SAE', created_at: new Date().toISOString(),
  config: {}, nlp_status: 'completed', statistics: { total_features: 2 },
};

const featureCalls = () => get.mock.calls.filter(([url]) => String(url).includes('/features'));

async function openAllScope() {
  render(<ExtractionJobCard extraction={completed as never} onDelete={vi.fn()} onCancel={vi.fn()} />);
  fireEvent.click(screen.getByText('Browse 2 features'));
  await screen.findByText('feature 1');
  fireEvent.click(screen.getByTitle('Sorting current page only (click to sort all)'));
}

describe('ExtractionJobCard dataset-wide sort', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useFeaturesStore.setState({ featuresByExtraction: {}, featureListMetadata: {}, searchFilters: {} } as never);
    get.mockResolvedValue({ data: { features: FEATURES, total: 2, limit: 50, offset: 0 } });
  });

  it('Label sorts the whole extraction by name, once', async () => {
    await openAllScope();
    const before = featureCalls().length;
    fireEvent.click(screen.getByText('Label'));
    await waitFor(() => expect(featureCalls().length).toBe(before + 1));
    const params = (featureCalls()[before][1] as { params: Record<string, unknown> }).params;
    expect(params).toMatchObject({ sort_by: 'name', sort_order: 'asc', offset: 0 });
  });

  it('Category sorts the whole extraction by category', async () => {
    await openAllScope();
    fireEvent.click(screen.getByText('Category'));
    await waitFor(() => expect(
      (featureCalls()[featureCalls().length - 1][1] as { params: Record<string, unknown> }).params,
    ).toMatchObject({ sort_by: 'category' }));
  });

  it('a column the server cannot order is not a dataset-wide sort', async () => {
    await openAllScope();
    const before = featureCalls().length;
    const header = screen.getByText('Description').closest('th')!;
    expect(header.getAttribute('title')).toMatch(/switch the sort scope to page/);
    fireEvent.click(header);
    await new Promise((r) => setTimeout(r, 50));
    expect(featureCalls().length).toBe(before);
  });
});

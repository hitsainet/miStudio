/**
 * The template lists' Filters panels filter, through the list routes.
 *
 * Both panels had a Filters button whose panel only said "Additional filters
 * coming soon (architecture type, date range, etc.)" / "Additional filters can
 * be added here (sort order, hook types, etc.)". The lists are paginated by the
 * server (50 per page), so the filters are list-route query params, and these
 * tests assert the request each control sends: its params and its call count.
 *
 * Also pinned (pre-existing defects in the same code): a page change kept no
 * filter (`fetchTemplates({ page })`), and the Favorites tab showed the All
 * tab's pagination over a different list.
 *
 * MUTATION CONTROLS (2026-10-18, each restored by sha256):
 *   F1  drop `encoder_type` from trainingFiltersToParams                 -> 3 red
 *   F2  send created_before for the To day itself (no +1 day)            -> 2 red
 *   F3  drop `onFiltersChange={setFilters}` from TrainingTemplatesPanel  -> 4 red
 *   F4  page change sends `{ page }` only (training panel)               -> 1 red
 *   F5  drop `hook_type` from extractionFiltersToParams                  -> 2 red
 *   F6  drop the sort mapping from extractionFiltersToParams             -> 2 red
 *   F7  drop `onFiltersChange={setFilters}` from ExtractionTemplatesPanel-> 3 red
 *   F8  drop the Favorites tab's `is_favorite: true`                     -> 1 red
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

vi.mock('../../api/trainingTemplates');
vi.mock('../../api/extractionTemplates');

import * as trainingApi from '../../api/trainingTemplates';
import * as extractionApi from '../../api/extractionTemplates';
import { TrainingTemplatesPanel } from './TrainingTemplatesPanel';
import { ExtractionTemplatesPanel } from './ExtractionTemplatesPanel';
import { useTrainingTemplatesStore } from '../../stores/trainingTemplatesStore';
import { useExtractionTemplatesStore } from '../../stores/extractionTemplatesStore';
import {
  trainingFiltersToParams,
  extractionFiltersToParams,
} from '../../utils/templateListFilters';

const page = (total: number, total_pages = 1, p = 1) => ({
  page: p, limit: 50, total, total_pages, has_next: p < total_pages, has_prev: p > 1,
});

const TT = {
  id: 't1', name: 'Jump template', description: null, encoder_type: 'jumprelu',
  model_id: null, dataset_ids: [], dataset_id: null, is_favorite: false, extra_metadata: {},
  hyperparameters: { hidden_dim: 8, latent_dim: 16, l1_alpha: 0.001, learning_rate: 0.001,
    batch_size: 8, total_steps: 10, warmup_steps: 0, weight_decay: 0, checkpoint_interval: 5,
    log_interval: 1, architecture_type: 'jumprelu', training_layers: [1] },
  created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z',
};

const ET = {
  id: 'e1', name: 'Residual template', description: null, layer_indices: [1],
  hook_types: ['residual'], max_samples: 10, batch_size: 2, micro_batch_size: null,
  top_k_examples: 5, context_prefix_tokens: 25, context_suffix_tokens: 25,
  is_favorite: false, extra_metadata: {},
  created_at: '2026-10-01T00:00:00Z', updated_at: '2026-10-01T00:00:00Z',
};

const lastCall = (fn: unknown) => {
  const calls = vi.mocked(fn as (...a: unknown[]) => unknown).mock.calls;
  return calls[calls.length - 1]?.[0] as Record<string, unknown>;
};

beforeEach(() => {
  vi.clearAllMocks();
  vi.mocked(trainingApi.getTrainingTemplates).mockResolvedValue({
    data: [TT as never], pagination: page(120, 3),
  } as never);
  vi.mocked(trainingApi.getFavoriteTrainingTemplates).mockResolvedValue({ data: [] } as never);
  vi.mocked(extractionApi.getExtractionTemplates).mockResolvedValue({
    data: [ET as never], pagination: page(1),
  } as never);
  vi.mocked(extractionApi.getFavoriteExtractionTemplates).mockResolvedValue({ data: [] } as never);
  useTrainingTemplatesStore.setState({ templates: [], favorites: [], loading: false, error: null, pagination: null });
  useExtractionTemplatesStore.setState({ templates: [], favorites: [], loading: false, error: null, pagination: null });
});

describe('filter -> params (pure)', () => {
  it('training: architecture, and a To day that includes its whole day', () => {
    const p = trainingFiltersToParams({
      encoder_type: 'jumprelu', created_from: '2026-10-03', created_to: '2026-10-07',
    });
    expect(p.encoder_type).toBe('jumprelu');
    expect(p.created_after).toBe(new Date(2026, 9, 3).toISOString());
    expect(p.created_before).toBe(new Date(2026, 9, 8).toISOString());
  });

  it('extraction: hook type and sort', () => {
    expect(extractionFiltersToParams({ hook_type: 'mlp', sort: 'name_asc' }))
      .toMatchObject({ hook_type: 'mlp', sort_by: 'name', order: 'asc' });
    expect(extractionFiltersToParams({}))
      .toMatchObject({ sort_by: 'created_at', order: 'desc' });
  });
});

describe('Training templates panel', () => {
  async function openFilters() {
    render(<TrainingTemplatesPanel />);
    await screen.findByText('Jump template');
    expect(screen.queryByText(/coming soon/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /Filters/ }));
  }

  it('the architecture filter refetches page 1 with encoder_type, once', async () => {
    await openFilters();
    const before = vi.mocked(trainingApi.getTrainingTemplates).mock.calls.length;
    fireEvent.change(screen.getByLabelText('Architecture'), { target: { value: 'jumprelu' } });
    await waitFor(() =>
      expect(vi.mocked(trainingApi.getTrainingTemplates).mock.calls.length).toBe(before + 1));
    expect(lastCall(trainingApi.getTrainingTemplates)).toMatchObject({ encoder_type: 'jumprelu' });
    expect(lastCall(trainingApi.getTrainingTemplates).page).toBeUndefined();
  });

  it('the date range sends created_after and created_before', async () => {
    await openFilters();
    fireEvent.change(screen.getByLabelText('Created from'), { target: { value: '2026-10-03' } });
    fireEvent.change(screen.getByLabelText('Created to'), { target: { value: '2026-10-07' } });
    await waitFor(() => expect(lastCall(trainingApi.getTrainingTemplates)).toMatchObject({
      created_after: new Date(2026, 9, 3).toISOString(),
      created_before: new Date(2026, 9, 8).toISOString(),
    }));
  });

  it('a page change keeps the filters', async () => {
    await openFilters();
    fireEvent.change(screen.getByLabelText('Architecture'), { target: { value: 'topk' } });
    await waitFor(() =>
      expect(lastCall(trainingApi.getTrainingTemplates)).toMatchObject({ encoder_type: 'topk' }));
    fireEvent.click(await screen.findByRole('button', { name: /Next/ }));
    await waitFor(() => expect(lastCall(trainingApi.getTrainingTemplates))
      .toMatchObject({ encoder_type: 'topk', page: 2 }));
  });

  it('search is sent to the route (debounced), and the count is the server total', async () => {
    await openFilters();
    expect(screen.getByText(/Showing 1 of 120 templates/)).toBeInTheDocument();
    fireEvent.change(screen.getByLabelText('Search templates'), { target: { value: 'jump' } });
    await waitFor(() =>
      expect(lastCall(trainingApi.getTrainingTemplates)).toMatchObject({ search: 'jump' }));
  });

  it('the Favorites tab is the filtered list restricted to favorites', async () => {
    render(<TrainingTemplatesPanel />);
    await screen.findByText('Jump template');
    fireEvent.click(screen.getByRole('button', { name: /^Favorites/ }));
    await waitFor(() =>
      expect(lastCall(trainingApi.getTrainingTemplates)).toMatchObject({ is_favorite: true }));
  });
});

describe('Extraction templates panel', () => {
  async function openFilters() {
    render(<ExtractionTemplatesPanel />);
    await screen.findByText('Residual template');
    expect(screen.queryByText(/can be added here/i)).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Toggle filters' }));
  }

  it('the hook-type filter refetches with hook_type, once', async () => {
    await openFilters();
    const before = vi.mocked(extractionApi.getExtractionTemplates).mock.calls.length;
    fireEvent.change(screen.getByLabelText('Hook type'), { target: { value: 'attention' } });
    await waitFor(() =>
      expect(vi.mocked(extractionApi.getExtractionTemplates).mock.calls.length).toBe(before + 1));
    expect(lastCall(extractionApi.getExtractionTemplates)).toMatchObject({ hook_type: 'attention' });
  });

  it('the sort order is sent to the route', async () => {
    await openFilters();
    fireEvent.change(screen.getByLabelText('Sort'), { target: { value: 'name_desc' } });
    await waitFor(() => expect(lastCall(extractionApi.getExtractionTemplates))
      .toMatchObject({ sort_by: 'name', order: 'desc' }));
  });

  it('no match offers Clear filters, which refetches unfiltered', async () => {
    await openFilters();
    vi.mocked(extractionApi.getExtractionTemplates).mockResolvedValueOnce({
      data: [], pagination: page(0),
    } as never);
    fireEvent.change(screen.getByLabelText('Hook type'), { target: { value: 'mlp' } });
    expect(await screen.findByText('No templates match these filters')).toBeInTheDocument();
    fireEvent.click(screen.getAllByRole('button', { name: 'Clear filters' })[0]);
    await waitFor(() =>
      expect(lastCall(extractionApi.getExtractionTemplates).hook_type).toBeUndefined());
  });
});

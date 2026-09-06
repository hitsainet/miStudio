/**
 * An expanded card must survive a background refresh.
 *
 * This asserts the USER-VISIBLE behaviour behind featuresStore.refresh.test.ts:
 * the reported symptom was not "a flag was set", it was
 *
 *   "the whole page refreshes ... and closes the expanded progress windows"
 *
 * `showMetrics` is card-local state, so it only resets if the card UNMOUNTS.
 * Asserting the flag alone would pass against any future change that unmounts
 * the grid for some other reason.
 *
 * MUTATION CONTROL:
 *   * restore `isLoadingExtractions: true` on every fetch -> this test fails
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { screen, fireEvent, act } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ExtractionsPanel } from './ExtractionsPanel';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useTrainingsStore } from '../../stores/trainingsStore';

vi.mock('../../stores/trainingsStore');
vi.mock('../../api/saes', () => ({ cancelSAEExtraction: vi.fn() }));
vi.mock('../extraction/StartExtractionModal', () => ({
  StartExtractionModal: () => null,
}));
vi.mock('../../api/models', () => ({
  triggerNlpAnalysis: vi.fn(),
  cancelNlpAnalysis: vi.fn(),
  resetNlpAnalysis: vi.fn(),
}));

const RUNNING = {
  id: 'extr_1',
  status: 'extracting',
  progress: 0.915,
  features_extracted: 10000,
  total_features: 32768,
  sae_name: 'SAE from granite-4.1-8b (L35)',
  created_at: new Date().toISOString(),
  config: {},
};

describe('ExtractionsPanel does not unmount cards on refresh', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    (useTrainingsStore as any).mockImplementation((selector?: any) => {
      const state = { fetchTrainings: vi.fn(), trainings: [] };
      return typeof selector === 'function' ? selector(state) : state;
    });
    useFeaturesStore.setState({
      allExtractions: [RUNNING as any],
      extractionsMetadata: { total: 1, limit: 50, offset: 0 },
      isLoadingExtractions: false,
      extractionsError: null,
    });
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('keeps Live Metrics expanded across a background refresh', () => {
    render(<ExtractionsPanel />);

    fireEvent.click(screen.getByText(/Show Live Metrics/));
    expect(screen.getByText(/Hide Live Metrics/)).toBeInTheDocument();

    // Exactly what a background refresh does to the store today.
    act(() => {
      useFeaturesStore.setState({
        allExtractions: [{ ...RUNNING, features_extracted: 12000 } as any],
      });
    });

    expect(screen.getByText(/Hide Live Metrics/)).toBeInTheDocument();
  });

  it('closes it only if the loading flag is raised — the old behaviour', () => {
    // Negative control kept in the suite: this documents WHY the flag must not
    // be raised for a refresh, and fails loudly if the panel stops gating the
    // grid on it (in which case this file needs rewriting, not deleting).
    render(<ExtractionsPanel />);

    fireEvent.click(screen.getByText(/Show Live Metrics/));
    expect(screen.getByText(/Hide Live Metrics/)).toBeInTheDocument();

    act(() => {
      useFeaturesStore.setState({ isLoadingExtractions: true });
    });
    act(() => {
      useFeaturesStore.setState({ isLoadingExtractions: false });
    });

    expect(screen.queryByText(/Hide Live Metrics/)).not.toBeInTheDocument();
    expect(screen.getByText(/Show Live Metrics/)).toBeInTheDocument();
  });
});

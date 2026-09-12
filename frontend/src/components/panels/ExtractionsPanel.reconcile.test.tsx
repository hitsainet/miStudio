/**
 * A dropped WebSocket event must not leave the panel permanently stale.
 *
 * Reported 2026-07-26: "one extraction job finished without updating the ui.
 * that sucks. and then after I refreshed the ui, the next two jobs stay queued."
 *
 * The panel fetched on mount and on filter change only, so a single missed
 * `extraction:completed` stuck the card on "extracting" until the user reloaded
 * by hand. Every other live surface here pairs WS with an HTTP fallback (see
 * systemMonitorStore); this one did not.
 *
 * MUTATION CONTROLS:
 *   * delete the reconcile useEffect          -> "keeps polling" test fails
 *   * drop the hasActiveExtraction condition  -> "stops when idle" test fails
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ExtractionsPanel } from './ExtractionsPanel';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useTrainingsStore } from '../../stores/trainingsStore';

vi.mock('../../stores/featuresStore');
vi.mock('../../stores/trainingsStore');
vi.mock('../../api/saes', () => ({ cancelSAEExtraction: vi.fn() }));
vi.mock('../extraction/StartExtractionModal', () => ({
  StartExtractionModal: () => null,
}));
vi.mock('../features/ExtractionJobCard', () => ({
  ExtractionJobCard: ({ extraction }: { extraction: any }) => (
    <div data-testid={`card-${extraction.id}`} />
  ),
}));

const fetchAllExtractions = vi.fn();

function mountWith(extractions: any[]) {
  (useFeaturesStore as any).mockReturnValue({
    allExtractions: extractions,
    extractionsMetadata: {},
    isLoadingExtractions: false,
    extractionsError: null,
    fetchAllExtractions,
    deleteExtraction: vi.fn(),
  });
  // The panel also selects individual slices via useFeaturesStore(selector).
  (useFeaturesStore as any).mockImplementation((selector?: any) => {
    const state = {
      allExtractions: extractions,
      extractionsMetadata: {},
      isLoadingExtractions: false,
      extractionsError: null,
      fetchAllExtractions,
      deleteExtraction: vi.fn(),
      updateExtractionById: vi.fn(),
    };
    return typeof selector === 'function' ? selector(state) : state;
  });
  (useTrainingsStore as any).mockImplementation((selector?: any) => {
    const state = { fetchTrainings: vi.fn(), trainings: [] };
    return typeof selector === 'function' ? selector(state) : state;
  });
  return render(<ExtractionsPanel />);
}

describe('ExtractionsPanel reconciliation poll', () => {
  beforeEach(() => {
    vi.useFakeTimers();
    fetchAllExtractions.mockClear();
  });
  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
  });

  it('keeps refetching while an extraction is still running', () => {
    mountWith([{ id: 'e1', status: 'extracting', nlp_status: null }]);
    const afterMount = fetchAllExtractions.mock.calls.length;

    vi.advanceTimersByTime(45_000);

    expect(fetchAllExtractions.mock.calls.length).toBeGreaterThan(afterMount);
  });

  it('keeps refetching while a queued job is waiting for its turn', () => {
    // The reported case: job 001 done, 002/003 queued. Without a poll the
    // panel never learns that 002 started.
    mountWith([
      { id: 'e1', status: 'completed', nlp_status: 'processing' },
      { id: 'e2', status: 'queued', nlp_status: null },
    ]);
    const afterMount = fetchAllExtractions.mock.calls.length;

    vi.advanceTimersByTime(45_000);

    expect(fetchAllExtractions.mock.calls.length).toBeGreaterThan(afterMount);
  });

  it('issues no extra requests once everything is terminal', () => {
    mountWith([{ id: 'e1', status: 'completed', nlp_status: 'completed' }]);
    const afterMount = fetchAllExtractions.mock.calls.length;

    vi.advanceTimersByTime(120_000);

    expect(fetchAllExtractions.mock.calls.length).toBe(afterMount);
  });
});

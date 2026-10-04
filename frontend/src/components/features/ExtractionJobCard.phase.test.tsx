/**
 * A near-full bar must say which phase it is in.
 *
 * Extraction has two long phases. Sampling fills the first 90% of the bar; then
 * feature records are committed to the database, which took ~20 minutes for a
 * 32,768-latent SAE on 2026-07-26 and looked, on screen, exactly like a hung
 * job at 100%.
 *
 * Phase 2 is identified from PERSISTED columns — status still `extracting`
 * while `total_features` has become non-null — because the extractions refetch
 * replaces store entries wholesale and would wipe a WebSocket-only marker.
 *
 * MUTATION CONTROLS:
 *   * drop the total_features condition from isWritingFeatures -> phase-1 test fails
 *   * remove the phase banner                                  -> banner test fails
 *   * return the sampling eta during phase 2                    -> ETA test fails
 */

import { describe, it, expect, vi, _beforeEach, afterEach } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ExtractionJobCard } from './ExtractionJobCard';

vi.mock('../../api/models', () => ({
  triggerNlpAnalysis: vi.fn(),
  cancelNlpAnalysis: vi.fn(),
  resetNlpAnalysis: vi.fn(),
}));

const base = {
  id: 'extr_1',
  status: 'extracting',
  progress: 0.915,
  features_extracted: 10000,
  total_features: null as number | null,
  sae_name: 'SAE from granite-4.1-8b (L35)',
  created_at: new Date().toISOString(),
  config: {},
};

function renderCard(overrides: Record<string, unknown>) {
  return render(
    <ExtractionJobCard
      extraction={{ ...base, ...overrides } as any}
      onDelete={vi.fn()}
      onCancel={vi.fn()}
    />,
  );
}

describe('ExtractionJobCard write-phase indication', () => {
  afterEach(() => vi.clearAllMocks());

  it('announces phase 2 once feature records are being written', () => {
    renderCard({ total_features: 32768, features_extracted: 10000 });

    expect(screen.getByText('Phase 2 of 2')).toBeInTheDocument();
    expect(screen.getByText('Writing to database')).toBeInTheDocument();
  });

  it('shows the real feature count, not the sample count', () => {
    renderCard({
      total_features: 32768,
      features_extracted: 10000,
      samples_processed: 2000,
      total_samples: 2000,
    });

    expect(screen.getByText('10,000 / 32,768 features')).toBeInTheDocument();
    expect(
      screen.queryByText('2,000 / 2,000 samples'),
    ).not.toBeInTheDocument();
  });

  it('stays silent during phase 1, when total_features is still null', () => {
    renderCard({
      total_features: null,
      progress: 0.5,
      samples_processed: 1000,
      total_samples: 2000,
    });

    expect(screen.queryByText('Phase 2 of 2')).not.toBeInTheDocument();
    expect(screen.getByText('1,000 / 2,000 samples')).toBeInTheDocument();
  });

  it('does not claim a phase for a completed job', () => {
    renderCard({ status: 'completed', total_features: 32759, progress: 1 });
    expect(screen.queryByText('Phase 2 of 2')).not.toBeInTheDocument();
  });

  it('suppresses the stale sampling ETA during phase 2', () => {
    // eta_seconds is left over from sampling and is meaningless now. Showing
    // it would promise a finish time for work that already ended.
    renderCard({
      total_features: 32768,
      features_extracted: 10000,
      eta_seconds: 5,
    });

    expect(screen.queryByText(/ETA: 5s/)).not.toBeInTheDocument();
  });

  it('marks the sampling metrics as final rather than live', () => {
    renderCard({ total_features: 32768, features_extracted: 10000 });

    // The metrics grid is collapsed by default; the note lives inside it.
    fireEvent.click(screen.getByText(/Show Live Metrics/));

    expect(
      screen.getByText(/Sampling metrics below are final/),
    ).toBeInTheDocument();
  });
});

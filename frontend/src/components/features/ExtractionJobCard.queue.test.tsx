/**
 * The feature browser's two new controls (2026-10-16):
 *  - the J-space label-disagreement QUEUE as a filter over this list (024 task 4.3) —
 *    it must send `jlens_disagreement` and sort worst-first, or the backend filter is
 *    unreachable from the screen where features live;
 *  - batch enhanced labeling over the STARRED subset (004), offered only while the
 *    starred filter is on, so the subset the button acts on is the one on screen.
 *
 * MUTATION CONTROLS:
 *   * drop `jlens_disagreement` from the toggle's filters -> the param test fails
 *   * send starred_only:false                             -> the batch test fails
 */
import { describe, it, expect, vi, afterEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ExtractionJobCard } from './ExtractionJobCard';

vi.mock('../../api/models', () => ({
  triggerNlpAnalysis: vi.fn(),
  cancelNlpAnalysis: vi.fn(),
  resetNlpAnalysis: vi.fn(),
}));
vi.mock('axios', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: { features: [], total: 0 } }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}));
vi.mock('../../api/enhancedLabeling', () => ({
  startEnhancedLabelingBatch: vi.fn().mockResolvedValue({
    extraction_id: 'extr_1', queued: ['elj_1', 'elj_2'], already_active: ['f3'], skipped: [], considered: 3,
  }),
}));

import axios from 'axios';
import { startEnhancedLabelingBatch } from '../../api/enhancedLabeling';

const completed = {
  id: 'extr_1', status: 'completed', progress: 1, features_extracted: 10, total_features: 10,
  sae_name: 'SAE', created_at: new Date().toISOString(), config: {}, nlp_status: 'completed',
  statistics: { total_features: 10 },
};

function openBrowser() {
  render(<ExtractionJobCard extraction={completed as any} onDelete={vi.fn()} onCancel={vi.fn()} />);
  fireEvent.click(screen.getByText('Browse 10 features'));
}

describe('feature browser: disagreement queue and batch enhanced labeling', () => {
  afterEach(() => vi.clearAllMocks());

  it('the queue toggle asks the server for disagreements, worst first', async () => {
    openBrowser();
    fireEvent.click(screen.getByLabelText('Filter to lens/label disagreements'));
    await waitFor(() => {
      const calls = vi.mocked(axios.get).mock.calls;
      const last = calls[calls.length - 1];
      expect(last[0]).toBe('/api/v1/extractions/extr_1/features');
      expect(last[1]?.params).toMatchObject({
        jlens_disagreement: true, sort_by: 'jlens_disagreement', sort_order: 'desc', offset: 0,
      });
    });
    expect(screen.getByLabelText('Filter to lens/label disagreements')).toHaveAttribute('aria-pressed', 'true');
  });

  it('batch enhanced labeling appears with the starred filter and queues the starred subset', async () => {
    openBrowser();
    expect(screen.queryByLabelText('Enhanced-label starred features')).toBeNull();
    fireEvent.click(screen.getByTitle('Filter favorites'));
    fireEvent.click(await screen.findByLabelText('Enhanced-label starred features'));
    await waitFor(() => expect(startEnhancedLabelingBatch).toHaveBeenCalledTimes(1));
    expect(startEnhancedLabelingBatch).toHaveBeenCalledWith('extr_1', { starred_only: true });
    expect(await screen.findByTestId('enhanced-batch-note')).toHaveTextContent(
      'Queued 2 of 3 starred feature(s); 1 already in flight.'
    );
  });
});

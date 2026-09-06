/**
 * The two icon-only actions on a completed extraction had no visible label.
 *
 * A bare chevron and a bare circular arrow sat next to a green "Label Features"
 * button, and the only way to learn what either did was to hover and wait for a
 * tooltip. One opens the feature browser; the other DISCARDS all NLP analysis
 * and recomputes it. Those carry very different consequences and looked
 * identical in weight.
 *
 * Both now carry text, and each is tinted to the thing it acts on — emerald for
 * features, cyan for NLP, matching the count and the badge already on the card.
 *
 * MUTATION CONTROLS:
 *   * remove the <span> from either button      -> its label test fails
 *   * drop the feature count from the expand label -> count test fails
 *   * swap the rotating chevron back to two icons  -> rotation test fails
 *   * drop aria-expanded                        -> a11y test fails
 */

import { describe, it, expect, vi, afterEach } from 'vitest';
import { screen, fireEvent } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ExtractionJobCard } from './ExtractionJobCard';

vi.mock('../../api/models', () => ({
  triggerNlpAnalysis: vi.fn(),
  cancelNlpAnalysis: vi.fn(),
  resetNlpAnalysis: vi.fn(),
}));

// Expanding fetches the feature list. The mock is here so the test controls
// whether that request SUCCEEDS OR FAILS — not to make the failure go away.
//
// The first version of this file mocked `get` to resolve, permanently. That
// silenced an unhandled-rejection warning, and in doing so hid the reason for
// it: `fetchExtractionFeatures` records its error in store state and then
// re-throws, while every one of its twelve call sites ignores the promise. So
// every failed feature fetch in a real browser produced an unhandled rejection.
// A mock that removes the only symptom of a defect is not test hygiene.
vi.mock('axios', () => ({
  default: {
    get: vi.fn().mockResolvedValue({ data: { features: [], total: 0 } }),
    post: vi.fn().mockResolvedValue({ data: {} }),
    delete: vi.fn().mockResolvedValue({ data: {} }),
  },
}));

import axios from 'axios';

const completed = {
  id: 'extr_1',
  status: 'completed',
  progress: 1,
  features_extracted: 32766,
  total_features: 32766,
  sae_name: 'SAE from granite-4.1-8b (L36-residual)',
  created_at: new Date().toISOString(),
  config: {},
  nlp_status: 'completed',
  statistics: { total_features: 32766, interpretable_count: 22478 },
};

function renderCard(overrides: Record<string, unknown> = {}) {
  return render(
    <ExtractionJobCard
      extraction={{ ...completed, ...overrides } as any}
      onDelete={vi.fn()}
      onCancel={vi.fn()}
    />,
  );
}

describe('ExtractionJobCard action labels', () => {
  afterEach(() => vi.clearAllMocks());

  it('says what the expand button opens, and how much is in it', () => {
    renderCard();

    // The count is the clearest possible statement of what expanding reveals.
    expect(screen.getByText('Browse 32,766 features')).toBeInTheDocument();
  });

  it('falls back to a plain label when the count is unknown', () => {
    renderCard({ statistics: undefined });

    expect(screen.getByText('Browse features')).toBeInTheDocument();
  });

  it('switches to a hide label once open', () => {
    renderCard();

    fireEvent.click(screen.getByText('Browse 32,766 features'));

    expect(screen.getByText('Hide features')).toBeInTheDocument();
  });

  it('reports its open state to assistive tech', () => {
    renderCard();
    const button = screen.getByText('Browse 32,766 features').closest('button')!;

    expect(button).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(button);
    expect(
      screen.getByText('Hide features').closest('button'),
    ).toHaveAttribute('aria-expanded', 'true');
  });

  it('names the NLP action instead of leaving a bare glyph', () => {
    renderCard();

    expect(screen.getByText('Re-run NLP')).toBeInTheDocument();
  });

  it('warns in the NLP tooltip that existing analysis is discarded', () => {
    renderCard();
    const button = screen.getByText('Re-run NLP').closest('button')!;

    // The consequence belongs somewhere the user can find BEFORE clicking, not
    // only in the confirm() that follows.
    expect(button.getAttribute('title')).toMatch(/discard/i);
  });

  it('uses ONE chevron that rotates, not two that swap', () => {
    const { container } = renderCard();
    const button = screen.getByText('Browse 32,766 features').closest('button')!;

    const icon = button.querySelector('svg')!;
    expect(icon.getAttribute('class')).toContain('transition-transform');
    expect(icon.getAttribute('class')).not.toContain('rotate-180');

    fireEvent.click(button);
    const openIcon = screen
      .getByText('Hide features')
      .closest('button')!
      .querySelector('svg')!;
    expect(openIcon.getAttribute('class')).toContain('rotate-180');
    expect(container).toBeTruthy();
  });
});


describe('ExtractionJobCard — a failed feature fetch stays handled', () => {
  afterEach(() => vi.clearAllMocks());

  /** Collect anything Node decides nobody handled, for the duration of `run`. */
  async function unhandledDuring(run: () => void): Promise<unknown[]> {
    const escaped: unknown[] = [];
    const listener = (reason: unknown) => escaped.push(reason);
    process.on('unhandledRejection', listener);
    try {
      run();
      // One turn for the rejection to propagate, one more for Node to conclude
      // that nothing caught it.
      await new Promise((r) => setTimeout(r, 0));
      await new Promise((r) => setTimeout(r, 0));
    } finally {
      process.off('unhandledRejection', listener);
    }
    return escaped;
  }

  it('does not leak an unhandled rejection when the request fails', async () => {
    // The real failure: a flaky network, a restarting backend, a 500.
    (axios.get as any).mockRejectedValueOnce(new Error('Network Error'));
    renderCard();

    const escaped = await unhandledDuring(() => {
      fireEvent.click(screen.getByText('Browse 32,766 features'));
    });

    expect(escaped).toEqual([]);
  });

  it('proves the harness can SEE an unhandled rejection', async () => {
    // Negative control. Without this, the test above passes just as happily
    // against a harness that cannot detect the thing it is asserting about —
    // which is how the original defect survived a green suite.
    const escaped = await unhandledDuring(() => {
       
      Promise.reject(new Error('deliberately unhandled'));
    });

    expect(escaped).toHaveLength(1);
  });

  it('still opens the panel when the fetch fails', async () => {
    (axios.get as any).mockRejectedValueOnce(new Error('Network Error'));
    renderCard();

    fireEvent.click(screen.getByText('Browse 32,766 features'));
    await new Promise((r) => setTimeout(r, 0));

    // The user gets the opened panel and whatever error the store recorded,
    // rather than a silently dead click.
    expect(screen.getByText('Hide features')).toBeInTheDocument();
  });
});

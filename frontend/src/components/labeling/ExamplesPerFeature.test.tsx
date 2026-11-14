/**
 * How many stored examples the judge is sent, and what bounds that choice.
 *
 * THE DEFECT. "Examples Per Feature" was bounded by a hardcoded 10-50 that bore
 * no relation to what the extraction retained. `TOP_K_EXAMPLES_SQL` selects
 * `rank <= :max_examples`, so asking for more than was stored returns fewer
 * rows with NO error and NO notice. Against the current top-k of 25 the control
 * offered 50 and silently delivered 25; against the older top-k-100 extractions
 * it capped at 50 and could not reach the other half of what was on disk. In
 * both directions the number on screen misdescribed what the judge would read.
 *
 * The ceiling is now what the run actually stored (`config.top_k_examples`),
 * and the fallback 50 survives only for callers that cannot say.
 *
 * SECOND DEFECT, fixed here too: `labeling_default_max_examples` was a DEAD
 * SETTING. Settings wrote it and nothing read it, while the batch-size default
 * immediately above it in the same component was wired — two identical-looking
 * defaults, one inert.
 *
 * MUTATION CONTROLS (each applied alone; suite must go red):
 *   E1  drop `storedExamples` from the ExtractionJobCard call site
 *          -> "bounds the input by what the extraction stored" fails
 *   E2  restore `max={50}` on the input
 *          -> the same test fails on the max attribute
 *   E3  drop the clamp from onChange
 *          -> "refuses a number above what is stored" fails
 *   E4  drop the clamp from the settings effect (pass the raw parsed value)
 *          -> "clamps a settings default above the stored count" fails
 *   E5  delete the settings effect entirely
 *          -> "honours the Settings default" fails
 *   E6  drop `max_examples` from the request body
 *          -> "sends the chosen count" fails
 *   E7  make the help text constant
 *          -> "states the stored count" fails
 *   E8  fall back to `storedExamples` instead of 50 when it is absent
 *          -> "keeps the historical ceiling when retention is unknown" fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

import { StartLabelingButton } from './StartLabelingButton';
import * as labelingAPI from '../../api/labeling';
import { useSettingsStore } from '../../stores/settingsStore';

vi.mock('../../api/labeling', async (orig) => ({
  ...(await orig<typeof labelingAPI>()),
  startLabeling: vi.fn(),
}));

/** Put rows in the settings store without going near the network. */
function seedSettings(rows: { key: string; value: string }[]) {
  useSettingsStore.setState({
    settings: rows as never,
    fetchAll: vi.fn().mockResolvedValue(undefined) as never,
  });
}

async function openForm(props: Record<string, unknown> = {}) {
  render(<StartLabelingButton extractionId="extr_1" {...props} />);
  fireEvent.click(await screen.findByText(/Label Features/i));
  return (await screen.findByLabelText(/Examples Per Feature/i)) as HTMLInputElement;
}

describe('Examples per feature is bounded by what was stored', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seedSettings([]);
  });

  it('bounds the input by what the extraction stored', async () => {
    const input = await openForm({ storedExamples: 25 });

    expect(input).toHaveAttribute('max', '25');
    expect(input).toHaveAttribute('min', '10');
  });

  it('keeps the historical ceiling when retention is unknown', async () => {
    // An absent value must not NARROW the control below what it always allowed.
    const input = await openForm();

    expect(input).toHaveAttribute('max', '50');
  });

  it('allows more than the old hardcoded 50 when more was stored', async () => {
    // The top-k-100 extractions are the case the old literal locked out.
    const input = await openForm({ storedExamples: 100 });

    expect(input).toHaveAttribute('max', '100');
  });

  it('refuses a number above what is stored', async () => {
    const input = await openForm({ storedExamples: 25 });

    fireEvent.change(input, { target: { value: '40' } });

    // Clamped, not accepted-then-silently-truncated by the server.
    expect(input.value).toBe('25');
  });

  it('opens below its own default when the extraction stored less', async () => {
    // The component's default is 25; a top-k-20 run must not present 25.
    const input = await openForm({ storedExamples: 20 });

    expect(input.value).toBe('20');
  });

  it('states the stored count so the number is not a guess', async () => {
    await openForm({ storedExamples: 25 });

    expect(
      screen.getByText(/This extraction stored 25 per feature/i),
    ).toBeInTheDocument();
  });

  it('offers a one-click way to send everything stored', async () => {
    const input = await openForm({ storedExamples: 25 });
    fireEvent.change(input, { target: { value: '10' } });

    fireEvent.click(screen.getByText(/Use all 25/i));

    expect(input.value).toBe('25');
  });
});

describe('The Settings default is no longer inert', () => {
  beforeEach(() => vi.clearAllMocks());

  it('honours the Settings default', async () => {
    seedSettings([{ key: 'labeling_default_max_examples', value: '18' }]);

    const input = await openForm({ storedExamples: 25 });

    await waitFor(() => expect(input.value).toBe('18'));
  });

  it('clamps a settings default above the stored count', async () => {
    // A global default of 50 must not override a run that retained 25.
    seedSettings([{ key: 'labeling_default_max_examples', value: '50' }]);

    const input = await openForm({ storedExamples: 25 });

    await waitFor(() => expect(input.value).toBe('25'));
  });
});

describe('The card actually supplies the stored count', () => {
  // REACHABILITY. Every test above hands `storedExamples` in directly, so all
  // of them stay green against a production call site that never passes it —
  // the capability would be untestable-by-those-tests and unreachable by users
  // at the same time. This one drives the real card, so deleting the prop from
  // ExtractionJobCard turns it red.
  beforeEach(() => {
    vi.clearAllMocks();
    seedSettings([]);
  });

  it('passes config.top_k_examples through from the extraction card', async () => {
    const { ExtractionJobCard } = await import('../features/ExtractionJobCard');

    render(
      <ExtractionJobCard
        extraction={
          {
            id: 'extr_1',
            status: 'completed',
            progress: 1,
            features_extracted: 16382,
            sae_name: 'SAE from train_b14d263e (L12-residual)',
            created_at: new Date().toISOString(),
            nlp_status: 'completed',
            // The number under test: not 25, not 50, so neither a default nor
            // the old literal can produce a passing result by coincidence.
            config: { top_k_examples: 40 },
            statistics: { total_features: 16382 },
          } as never
        }
        onDelete={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByText(/Label Features/i));

    const input = (await screen.findByLabelText(
      /Examples Per Feature/i,
    )) as HTMLInputElement;
    expect(input).toHaveAttribute('max', '40');
  });
});

describe('The chosen count reaches the request', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    seedSettings([]);
  });

  it('sends the chosen count', async () => {
    const input = await openForm({ storedExamples: 25 });
    fireEvent.change(input, { target: { value: '25' } });

    fireEvent.click(screen.getByText(/^Start Labeling$/i));

    await waitFor(() => expect(labelingAPI.startLabeling).toHaveBeenCalledTimes(1));
    const [body] = (labelingAPI.startLabeling as ReturnType<typeof vi.fn>).mock
      .calls[0];
    // Assert the PAYLOAD, not merely that the call happened: a request sending
    // the wrong count passes a was-called assertion.
    expect(body.max_examples).toBe(25);
  });
});

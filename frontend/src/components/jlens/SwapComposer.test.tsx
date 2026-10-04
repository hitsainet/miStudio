/**
 * The swap composer: pick a value from the readout, type any replacement.
 *
 * The two hazards this component exists to remove, both paid for in real runs:
 *
 *  1. THE PARTNER WAS INVISIBLE. It came from pin order, so re-pinning in a
 *     different order ran a different experiment under an identical click, and
 *     `interventions.json` recorded a `target_token` nobody saw.
 *  2. THE LAYERS WERE GUESSED. Three intervention runs came back null before
 *     the first real result, every one because the perturbed layers carried
 *     none of the concept — a matched random control did just as well there.
 */

import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, it, expect, vi, beforeEach } from 'vitest';

import { SwapComposer, seedRange } from './SwapComposer';

vi.mock('../../api/jlens', () => ({
  jlensApi: { checkTokens: vi.fn().mockResolvedValue([]) },
}));
import { jlensApi } from '../../api/jlens';

const base = (over: Partial<React.ComponentProps<typeof SwapComposer>> = {}) => ({
  modelId: 'm_1',
  from: ' Cisco',
  fromLayers: [30, 31, 32],
  lensLayers: [28, 29, 30, 31, 32, 33],
  budget: 10,
  busy: false,
  onCancel: vi.fn(),
  onSubmit: vi.fn(),
  ...over,
});

beforeEach(() => {
  vi.clearAllMocks();
  (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([]);
});

describe('seedRange', () => {
  it('seeds from the layers the token ACTUALLY reached', () => {
    // MUTATION CONTROL: seed from `lensLayers` and this fails with [28, 33].
    expect(seedRange([30, 31, 32], [28, 29, 30, 31, 32, 33])).toEqual([30, 32]);
  });

  it('never offers layers the lens does not cover', () => {
    /**
     * A partial fit is the normal case now — gemma-4-12b-it is fitted at L42-45
     * of 48. Seeding from the row alone would offer layers the artifact cannot
     * speak for, and the swap would be filed against it as though it could.
     */
    expect(seedRange([2, 43, 44, 99], [42, 43, 44, 45])).toEqual([43, 44]);
  });

  it('falls back to the row when the two do not intersect at all', () => {
    // Better an honest out-of-coverage range the reader can see and correct
    // than a silent empty selection that reads as "nothing to do".
    expect(seedRange([5, 6], [42, 43])).toEqual([5, 6]);
  });

  it('is null when there is nothing to seed from', () => {
    expect(seedRange([], [])).toBeNull();
  });
});

describe('SwapComposer', () => {
  it('shows the clicked row as the FROM side, with the raw token recoverable', () => {
    render(<SwapComposer {...base()} />);
    const chip = screen.getByTestId('swap-from');
    // Rendered like every other token in the panel: a leading space is `·`.
    expect(chip.textContent).toBe('·Cisco');
    // ...and the exact string is still available, because ' Cisco' and 'Cisco'
    // are different directions and must not look identical.
    expect(chip.getAttribute('title')).toBe(JSON.stringify(' Cisco'));
  });

  it('cannot run until a DIFFERENT, non-empty partner is typed', async () => {
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SwapComposer {...base({ onSubmit })} />);

    expect(screen.getByTestId('swap-run')).toBeDisabled();

    await user.type(screen.getByTestId('swap-target'), ' Cisco');
    expect(screen.getByTestId('swap-run')).toBeDisabled();
    expect(screen.getByText(/exchanges two DIFFERENT coordinates/i)).toBeTruthy();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('submits the typed token VERBATIM, leading space and all', async () => {
    /**
     * MUTATION CONTROL: `.trim()` the value anywhere on the way out and this
     * fails. ' Juniper' and 'Juniper' are different rows of W_U.
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SwapComposer {...base({ onSubmit })} />);

    await user.type(screen.getByTestId('swap-target'), ' Juniper');
    await user.click(screen.getByTestId('swap-run'));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toBe(' Juniper');
  });

  it('accepts a token that never appeared in the readout', async () => {
    /**
     * The server never restricted this and neither may the UI — those are
     * precisely the interesting swap targets ("does ' Rome' arrive if I put
     * ' Paris' where it was?").
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SwapComposer {...base({ onSubmit })} />);
    await user.type(screen.getByTestId('swap-target'), ' Wombat');
    await user.click(screen.getByTestId('swap-run'));
    expect(onSubmit.mock.calls[0][0]).toBe(' Wombat');
  });

  it('submits the SEEDED layers, not the whole lens axis', async () => {
    /**
     * MUTATION CONTROL: submit `lensLayers` and this fails with six layers.
     * This is the null-run lesson encoded: hook where the concept is.
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(<SwapComposer {...base({ onSubmit })} />);
    await user.type(screen.getByTestId('swap-target'), ' Juniper');
    await user.click(screen.getByTestId('swap-run'));
    expect(onSubmit.mock.calls[0][1]).toEqual([30, 31, 32]);
  });

  it('ACCEPTS a multi-piece replacement and says it is weaker evidence', async () => {
    /**
     * This used to block. It excluded most of the interesting vocabulary —
     * ' Juniper' is [' Jun', 'iper'] for several tokenizers this project runs,
     * and there is nothing incoherent about the concept those pieces encode.
     *
     * What the reader must not lose is WHAT CHANGED about the measurement: the
     * direction becomes the mean of the pieces' rows, and only the first piece
     * is scored, because a next-token distribution can answer for exactly one
     * position. ' Jun' also begins ' June' and ' Junior', so a hit is weaker
     * than a single-token target — and this run is filed as rung-2 evidence.
     *
     * MUTATION CONTROL: re-add `!verdict` to `ready` and the enabled assertion
     * fails; drop the warning paragraph and the disclosure assertion fails.
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        token: ' rugby union',
        ids: [1, 2, 3],
        n_tokens: 3,
        usable: false,
        detail: '3 tokens — a direction needs exactly one',
      },
    ]);
    render(<SwapComposer {...base({ onSubmit })} />);

    await user.type(screen.getByTestId('swap-target'), ' rugby union');
    await user.tab(); // the check runs on blur
    await waitFor(() =>
      expect(screen.getByText(/a direction needs exactly one/)).toBeTruthy(),
    );
    // DISCLOSED, not blocked: the reader is told what the measurement becomes.
    expect(screen.getByText(/only the first is scored/i)).toBeTruthy();
    expect(screen.getByTestId('swap-run')).toBeEnabled();

    await user.click(screen.getByTestId('swap-run'));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0]).toBe(' rugby union');
  });

  it('SEEDS within the budget, so the default cannot oversteer', () => {
    /**
     * A ranked-row click passes every layer the token reached, and a common
     * token reaches all of them — one click used to hook the whole stack at
     * strength 1, the oversteer BR-017 v0.2 warns about for small models.
     *
     * MUTATION CONTROL: drop the budget argument from the seed and this fails
     * with three layers.
     */
    render(<SwapComposer {...base({ budget: 2 })} />);
    const summary = screen.getByTestId('swap-layer-summary').textContent ?? '';
    expect(summary).toMatch(/L31/);
    expect(summary).toMatch(/L32/);
    // The shallowest hit is dropped: shallow hits are mostly the junk bands.
    expect(summary).not.toMatch(/L30/);
    expect(summary).not.toMatch(/Over the/);
  });

  it('warns once the reader WIDENS past the budget', () => {
    /**
     * Not a refusal — the server records `over_layer_budget` and runs it — but
     * swaps oversteer easily and a silent overrun is how one click hooked a
     * whole stack. The seed is safe; widening is the reader's call, and it is
     * told what it costs.
     *
     * MUTATION CONTROL: drop the budget comparison and this fails.
     */
    render(<SwapComposer {...base({ budget: 2 })} />);
    const [first] = screen.getAllByRole('spinbutton');
    // fireEvent.change, not user.type: the input is CONTROLLED and clamped, so
    // typing appends digit by digit and each intermediate value is clamped —
    // '2' then '8' lands on 288 -> 33, not 28.
    fireEvent.change(first, { target: { value: '28' } });
    expect(screen.getByTestId('swap-layer-summary').textContent).toMatch(
      /Over the 2-layer budget/,
    );
  });

  it('never hooks a layer the token did not reach', async () => {
    /**
     * Filtering the whole lens AXIS by the range hooked every layer BETWEEN the
     * hits too: a token at L5 and L20 hooked all sixteen, fourteen carrying
     * none of the concept. That is the shape that produced three null runs — a
     * matched random control does just as well where there is nothing to
     * perturb.
     *
     * MUTATION CONTROL: pool from `lensLayers` and this fails with 16 layers.
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SwapComposer
        {...base({
          fromLayers: [5, 20],
          lensLayers: Array.from({ length: 24 }, (_, i) => i),
          budget: 10,
          onSubmit,
        })}
      />,
    );
    await user.type(screen.getByTestId('swap-target'), ' Juniper');
    await user.click(screen.getByTestId('swap-run'));
    expect(onSubmit.mock.calls[0][1]).toEqual([5, 20]);
  });

  it('states the layer span honestly when it is not contiguous', () => {
    render(
      <SwapComposer
        {...base({ fromLayers: [42, 43, 45], lensLayers: [42, 43, 44, 45] })}
      />,
    );
    // describeSpan renders the real set rather than implying a solid run.
    expect(screen.getByTestId('swap-layer-summary').textContent).toMatch(/L42/);
  });

  it('tells the reader to include the leading space', () => {
    // The reference UI puts this in its label; this repo only implied it, and
    // the unspaced form is the wrong token nearly every time.
    render(<SwapComposer {...base()} />);
    expect(screen.getByText(/Include the leading space/i)).toBeTruthy();
  });

  it('BLOCKS a target that encodes to nothing — it has no row at all', async () => {
    /**
     * `usable` is false for TWO different things and only one is a warning.
     * Dropping the whole rejection check to allow multi-piece targets also let
     * n_tokens === 0 through: a string with no unembedding row, which takes a
     * 202 and a slot on the single-GPU queue before the worker refuses it. That
     * is the exact waste `/jlens/token-check` exists to prevent.
     *
     * MUTATION CONTROL: drop `!encodesToNothing` from `ready` and this fails.
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        token: '\u200b',
        ids: [],
        n_tokens: 0,
        usable: false,
        detail: 'Encodes to nothing; there is no direction to act along.',
      },
    ]);
    render(<SwapComposer {...base({ onSubmit })} />);
    await user.type(screen.getByTestId('swap-target'), '\u200b');
    await user.tab();
    await waitFor(() =>
      expect(screen.getByText(/encodes to no tokens at all/i)).toBeTruthy(),
    );
    expect(screen.getByTestId('swap-run')).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('BLOCKS a token whose layers the lens never covered', async () => {
    /**
     * `hitLayers` falls back to the raw hits so the mismatch is visible rather
     * than an empty selection — but submitting it credits an artifact for
     * layers it was never fitted at, and the server refuses with a 400 AFTER
     * the composer has closed and dropped the typed target.
     *
     * MUTATION CONTROL: drop `!outOfCoverage` from `ready` and this fails.
     */
    const user = userEvent.setup();
    const onSubmit = vi.fn();
    render(
      <SwapComposer
        {...base({ fromLayers: [5, 6], lensLayers: [42, 43, 44, 45], onSubmit })}
      />,
    );
    await user.type(screen.getByTestId('swap-target'), ' Juniper');
    expect(screen.getByText(/not fitted for/i)).toBeTruthy();
    expect(screen.getByTestId('swap-run')).toBeDisabled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it('cancels without running anything', async () => {
    const user = userEvent.setup();
    const onCancel = vi.fn();
    const onSubmit = vi.fn();
    render(<SwapComposer {...base({ onCancel, onSubmit })} />);
    await user.click(screen.getByRole('button', { name: /^Cancel$/ }));
    expect(onCancel).toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });
});

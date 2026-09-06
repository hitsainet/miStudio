/**
 * Choosing which layers to read.
 *
 * The range is ABSOLUTE layer numbers, never indices into an axis: the axis
 * differs per lens type and per artifact, so an index means a different layer
 * depending on which lens is being read, and a range that silently moves when
 * the mode changes is worse than no range.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * stop clamping to [min, max]     -> "clamps to what the model has" fails
 *   * stop ordering crossed ends      -> "orders a crossed range" fails
 *   * drop the re-read caveat         -> "says the range is a REQUEST" fails
 */
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

import { LayerRangePicker } from './LayerRangePicker';

function setup(over: Record<string, unknown> = {}) {
  const onChange = vi.fn();
  render(
    <LayerRangePicker
      min={0}
      max={25}
      value={null}
      onChange={onChange}
      {...over}
    />,
  );
  return { onChange };
}

describe('LayerRangePicker', () => {
  it('shows the model’s own span rather than a constant', () => {
    setup({ min: 3, max: 14 });
    expect(screen.getByText('of 3–14')).toBeInTheDocument();
  });

  it('CLAMPS to what the model actually has', async () => {
    /**
     * An out-of-range bound asks the server for a layer the model does not
     * have. It is refused there, which is correct and a poor way to find out.
     *
     * MUTATION CONTROL: pass the raw value through and this fails.
     */
    const { onChange } = setup({ min: 0, max: 25, value: [0, 25] });
    // ONE change event, not keystrokes: the inputs are controlled by `value`,
    // so typing digit by digit recomputes from the SAME prop each time and the
    // final call depends on typing order rather than on the clamp.
    fireEvent.change(screen.getByLabelText('Last layer'), {
      target: { value: '99' },
    });
    expect(onChange).toHaveBeenLastCalledWith([0, 25]);
  });

  it('ORDERS a crossed range instead of selecting nothing', async () => {
    /**
     * Ends that have crossed select no layers at all, and a ranked list over no
     * layers reads as "the model surfaced nothing".
     *
     * MUTATION CONTROL: emit [a, b] unordered and this fails.
     */
    const { onChange } = setup({ value: [10, 20] });
    fireEvent.change(screen.getByLabelText('First layer'), {
      target: { value: '24' },
    });
    expect(onChange).toHaveBeenLastCalledWith([20, 24]);
  });

  it('offers a way BACK to every layer once narrowed', async () => {
    const { onChange } = setup({ value: [4, 8] });
    await userEvent.click(screen.getByRole('button', { name: /All layers/ }));
    expect(onChange).toHaveBeenCalledWith(null);
  });

  it('does not offer "All layers" when nothing is narrowed', () => {
    setup({ value: null });
    expect(screen.queryByRole('button', { name: /All layers/ })).toBeNull();
  });

  it('does not offer Re-read while a readout is IN FLIGHT', async () => {
    /**
     * Each click is a GPU-bound job. The request-sequence guard discards stale
     * responses but does not stop the server doing the work, so repeated clicks
     * during a minute-long model load queue several concurrent readouts.
     *
     * MUTATION CONTROL: drop `disabled={busy}` and this fails.
     */
    const onApply = vi.fn();
    render(
      <LayerRangePicker min={0} max={9} value={[1, 4]} onChange={vi.fn()} onApply={onApply} busy />,
    );
    const btn = screen.getByRole('button', { name: /Reading/ });
    expect(btn).toBeDisabled();
    await userEvent.click(btn);
    expect(onApply).not.toHaveBeenCalled();
  });

  it('says WHICH views the narrowing affects', () => {
    /**
     * "Narrowing filters what is shown" was true only of the ranked columns —
     * the grid, rail and trajectory keep rendering the full axis, so the panel
     * shows a 6-layer and a 26-layer view of one readout at the same time.
     *
     * MUTATION CONTROL: revert to the unqualified wording and this fails.
     */
    setup();
    expect(screen.getByText(/ranked lists only/i)).toBeInTheDocument();
  });

  it('says the range is a REQUEST parameter, not just a filter', () => {
    /**
     * The numbers on screen came from whatever range was READ. Narrowing
     * without re-reading ranks over data captured under the old range, and a
     * reader who is not told that will take the counts as measured over the
     * span shown.
     *
     * MUTATION CONTROL: drop the caveat and this fails.
     */
    setup();
    expect(
      screen.getByText(/Re-read to capture only these/i),
    ).toBeInTheDocument();
  });

  it('re-reads on demand rather than on every keystroke', async () => {
    /** A readout is a GPU-bound job; firing one per digit typed is not a filter. */
    const onApply = vi.fn();
    const onChange = vi.fn();
    render(
      <LayerRangePicker
        min={0}
        max={25}
        value={[2, 6]}
        onChange={onChange}
        onApply={onApply}
      />,
    );
    fireEvent.change(screen.getByLabelText('First layer'), {
      target: { value: '5' },
    });
    expect(onApply).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('button', { name: /Re-read/ }));
    expect(onApply).toHaveBeenCalledTimes(1);
  });
});

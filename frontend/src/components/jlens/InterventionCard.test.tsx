/**
 * The control is not optional (BR-018).
 *
 * MUTATION CONTROLS:
 *   * omit k / control_seed from the request -> "always sends" fails
 *   * enable the card with no pinned tokens  -> "needs a direction" fails
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';
import { InterventionCard } from './InterventionCard';

vi.mock('../../api/jlens', () => ({
  jlensApi: { intervene: vi.fn(), checkTokens: vi.fn() },
}));
vi.mock('../../api/models', () => ({ getTaskStatus: vi.fn() }));

import { jlensApi } from '../../api/jlens';
import { getTaskStatus } from '../../api/models';

beforeEach(() => vi.clearAllMocks());

describe('InterventionCard', () => {
  it('cannot run without a pinned token to act along', () => {
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[]}
        layers={[1]}
        artifactId={null}
      />
    );
    expect(screen.getByRole('button', { name: /intervene/i })).toBeDisabled();
  });

  it('ALWAYS sends a size-matched, reconstructible control', async () => {
    vi.mocked(jlensApi.intervene).mockResolvedValue({
      task_id: 't1',
      model_id: 'm_1',
      queue: 'extraction',
    });
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId="slug"
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /run with control/i }));

    await waitFor(() => expect(jlensApi.intervene).toHaveBeenCalledTimes(1));
    const sent = vi.mocked(jlensApi.intervene).mock.calls[0][0];
    // An intervention without a control is not a weaker finding; it is not a
    // finding. There is deliberately no way to omit these.
    expect(sent.k).toBeGreaterThanOrEqual(1);
    expect(typeof sent.control_seed).toBe('number');
    // And the direction travels as a TOKEN, because the browser has no W_U.
    expect(sent.direction_token).toBe(' Paris');
    expect(sent.layers).toEqual([10, 11]);
  });

  it('EXPLAINS an empty direction list instead of showing a blank select', async () => {
    /**
     * OBSERVED IN THE PRODUCT. With nothing pinned this rendered as a blank
     * dropdown beside three fields that DO accept typing, which reads as a
     * broken control rather than a missing prerequisite — the caption "a pinned
     * token" only means something once you already know what pinning is.
     *
     * MUTATION CONTROL: map over `pinned` unconditionally and this fails.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[]}
        layers={[1]}
        artifactId={null}
      />
    );
    // The toggle is disabled with nothing pinned, so the reason has to be
    // reachable from the toggle itself as well as from inside the form.
    expect(
      screen.getByRole('button', { name: /intervene/i })
    ).toHaveAttribute('title', expect.stringMatching(/Pin a token first/));
  });

  it('shows the empty-list OPTION, not merely a disabled toggle', async () => {
    /**
     * The test above does NOT pin the branch its own comment claims. That
     * `title` sits on the toggle, is pre-existing, and is untouched by the
     * empty-select change — and with `pinned=[]` the toggle is disabled, so
     * `open` stays false and the `<select>` is not in the DOM at all. Reverting
     * the option branch left the suite green under a "MUTATION CONTROL" line
     * saying it would not.
     *
     * Opened with a token pinned, then re-rendered empty, so the select is
     * actually mounted when the assertion runs.
     *
     * MUTATION CONTROL: map over `pinned` unconditionally and this fails.
     */
    const { rerender } = render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[1]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    rerender(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[]}
        layers={[1]}
        artifactId={null}
      />
    );
    // SCOPED TO THE DIRECTION SELECT. The form carries a primitive select
    // too, so an unscoped option query counts its members and passes for the
    // wrong reason.
    const direction = screen.getByRole('combobox', { name: /Direction/i });
    const options = within(direction).getAllByRole('option');
    expect(options).toHaveLength(1);
    expect(options[0].textContent).toMatch(/No pinned tokens/i);
  });

  it('intervenes on the PROMPT ON SCREEN, never an empty one', async () => {
    /**
     * This sent `prompt: ''` — an empty string — so every intervention launched
     * from the card scored a forward pass over nothing while the readout beside
     * it described a real prompt. The result named a layer and a direction and
     * measured neither in the context the reader was looking at. The server
     * would 422 it now (`min_length=1`), which is the only reason it surfaced.
     *
     * The old test could not see it: it never passed a prompt at all, and
     * because test files are excluded from `tsc`, adding the required prop did
     * not make it fail — it simply sent `undefined`.
     *
     * MUTATION CONTROL: revert to `prompt: ''` and this fails.
     */
    vi.mocked(jlensApi.intervene).mockResolvedValue({
      task_id: 't2',
      model_id: 'm_1',
      queue: 'extraction',
    });
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' spider']}
        layers={[4]}
        artifactId="slug"
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /run with control/i }));

    await waitFor(() => expect(jlensApi.intervene).toHaveBeenCalledTimes(1));
    const sent = vi.mocked(jlensApi.intervene).mock.calls[0][0];
    expect(sent.prompt).toBe('the animal that spins webs');
  });

  it('SENDS the extra trial prompts, de-duplicated, with the screen prompt first', async () => {
    /**
     * Below four trials NO outcome separates the intervened and control
     * intervals, and every surface sent exactly one prompt — so "no effect was
     * demonstrated" was the only verdict the product could ever produce, and
     * the panel's remedy pointed at this card, which had no such control.
     * `JLensRequest.prompts` existed and nothing populated it.
     *
     * MUTATION CONTROLS:
     *   * stop sending `prompts`            -> "sends" fails
     *   * drop the de-duplication           -> "de-duplicated" fails
     *   * put the screen prompt last        -> "first" fails
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByTestId('intervention-prompts'), {
      // A BLANK LINE, A DUPLICATE OF THE SCREEN PROMPT, AND A REPEAT. Two
      // identical trials are one observation counted twice, which narrows the
      // Wilson interval on evidence that does not exist.
      target: {
        value:
          'the capital of Italy is\n\nthe animal that spins webs\nthe capital of Italy is\n  the capital of Japan is  ',
      },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));

    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.prompts).toEqual([
      'the animal that spins webs',
      'the capital of Italy is',
      'the capital of Japan is',
    ]);
  });

  it('does NOT send a prompts list when there is only the one on screen', async () => {
    /**
     * Otherwise every request carries a one-element list, and the worker's
     * "prompt alone is accepted and reported as n=1" path is never exercised.
     *
     * MUTATION CONTROL: always send `prompts` and this fails.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    expect(
      (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0].prompts
    ).toBeUndefined();
  });

  it('WARNS that too few trials cannot separate, BEFORE the GPU job', async () => {
    /**
     * Learning this from the result costs a GPU job on a single-GPU queue,
     * behind a possible 45-minute fit.
     *
     * MUTATION CONTROL: drop the `< MIN_TRIALS` branch and this fails.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    const count = screen.getByTestId('intervention-trial-count');
    expect(count).toHaveTextContent('1 trial');
    expect(count).toHaveTextContent(/not attainable below 4/);

    // AND THE WARNING CLEARS once there are enough — or it would be permanent
    // decoration and the assertion above would pass against a static string.
    fireEvent.change(screen.getByTestId('intervention-prompts'), {
      target: { value: 'a\nb\nc' },
    });
    expect(screen.getByTestId('intervention-trial-count')).toHaveTextContent(
      '4 trials'
    );
    expect(
      screen.getByTestId('intervention-trial-count')
    ).not.toHaveTextContent(/not attainable/);
  });

  it('reports NOT ATTAINABLE rather than "no effect" on a one-trial run', async () => {
    /**
     * The panel grew this three-state verdict and the card did not, so the card
     * kept printing the exact sentence the change exists to remove — and it is
     * the only surface from which a projective_ablation can be run at all.
     *
     * MUTATION CONTROL: drop the `separation_attainable === false` branch and
     * this fails, falling through to the OVERLAP wording.
     */
    (getTaskStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 't1',
      state: 'SUCCESS',
      result: {
        n_trials: 1,
        separation_attainable: false,
        min_trials_for_separation: 4,
        separated_from_control: false,
        excess_top1_over_control: 1,
        baseline_top1: { hits: 0, n: 1, rate: 0, ci95_low: 0, ci95_high: 0.79 },
        intervened_top1: { hits: 1, n: 1, rate: 1, ci95_low: 0.21, ci95_high: 1 },
        control_top1: { hits: 0, n: 1, rate: 0, ci95_low: 0, ci95_high: 0.79 },
      },
    });
    render(
      <InterventionCard
        modelId="m_1"
        prompt="the animal that spins webs"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));

    await waitFor(
      () => expect(screen.getByTestId('intervention-result')).toBeInTheDocument(),
      { timeout: 8000 }
    );
    // SCOPED TO THE RESULT. The pre-run trial counter carries the same phrase,
    // so an unscoped query matches two elements and would also pass against a
    // result block that said nothing at all.
    const out = screen.getByTestId('intervention-result');
    expect(within(out).getByText(/not attainable below 4/)).toBeInTheDocument();
    expect(within(out).queryByText(/no effect was demonstrated/)).toBeNull();
    // AND THE ARMS ARE STILL SHOWN — the run happened and its numbers are real,
    // they just cannot answer this question.
    expect(screen.getByText('1/1 = 1.000 [0.21, 1.00]')).toBeInTheDocument();
  }, 15000);


  it('can RUN A SWAP, sending the partner it named on screen', async () => {
    /**
     * A swap was runnable and not demonstrable anywhere in the product. The
     * ranked list can launch one — it has a pinned partner to hand — but sends
     * a single prompt, and below four trials no outcome separates, so that path
     * can only ever report "not attainable". This card is the only surface that
     * can supply trials, and it offered neither `coordinate_swap` nor a way to
     * name its partner, so it could not run one at all.
     *
     * MUTATION CONTROLS:
     *   * drop coordinate_swap from PRIMITIVES  -> "can run a swap" fails
     *   * stop sending target_token             -> "sending the partner" fails
     *   * send target_token for additive too    -> "only for a swap" fails
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris', ' Rome']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByRole('combobox', { name: /Primitive/i }), {
      target: { value: 'coordinate_swap' },
    });
    // FOUR TRIALS, or the run is a guaranteed null and the surface is still
    // unable to demonstrate anything — which is the whole point of the fix.
    fireEvent.change(screen.getByTestId('intervention-prompts'), {
      target: {
        value:
          'The capital of Italy is\nThe capital of Japan is\nThe capital of Spain is',
      },
    });
    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));

    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.primitive).toBe('coordinate_swap');
    expect(sent.direction_token).toBe(' Paris');
    expect(sent.target_token).toBe(' Rome');
    expect(sent.target_token).not.toBe(sent.direction_token);
    expect(sent.prompts).toHaveLength(4);
  });

  it('sends NO partner for a primitive that takes one direction', async () => {
    /**
     * `target_token` defaults to `direction_token` on the server, so sending a
     * partner for an additive steer silently changes what gets SCORED — "does
     * Paris arrive" becomes "does Rome arrive" under an unchanged label.
     *
     * MUTATION CONTROL: send `target_token` unconditionally and this fails.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris', ' Rome']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.primitive).toBe('additive');
    expect(sent.target_token).toBeUndefined();
  });

  it('BLOCKS a swap with one pinned token, and says why', async () => {
    /**
     * The server refuses it, but only after a 202 and a slot on the single-GPU
     * queue behind a possible 45-minute fit. Whether two tokens were pinned is
     * knowable here and needs no model.
     *
     * MUTATION CONTROL: drop the `isSwap && !chosenPartner` clause and this
     * fails — the button becomes enabled.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByRole('combobox', { name: /Primitive/i }), {
      target: { value: 'coordinate_swap' },
    });
    const run = screen.getByRole('button', { name: /Run with control/i });
    expect(run).toBeDisabled();
    // The wording names BOTH ways out — pinning is no longer the only one.
    expect(run).toHaveAttribute('title', expect.stringMatching(/second token/));
    expect(run).toHaveAttribute('title', expect.stringMatching(/or type it/));
    // AND THE PARTNER SELECT EXPLAINS ITSELF rather than sitting blank.
    expect(screen.getByTestId('intervention-partner')).toHaveTextContent(
      /Pin a second token/
    );

    // THE SAME FORM RUNS once the prerequisite is met — or "disabled" could be
    // permanent and the assertion above would pass against a dead control.
    fireEvent.change(screen.getByRole('combobox', { name: /Primitive/i }), {
      target: { value: 'additive' },
    });
    expect(screen.getByRole('button', { name: /Run with control/i })).toBeEnabled();
  });

  it('DISABLES strength for a primitive that ignores it', async () => {
    /**
     * The hook passes strength to `apply_additive` alone; an ablation and a
     * swap ignore it, and the server records null. An editable box invites a
     * strength sweep that returns bit-identical results at every value.
     *
     * MUTATION CONTROL: drop the USES_STRENGTH guard and this fails.
     */
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris', ' Rome']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    const strength = screen.getByRole('spinbutton', { name: /Strength/i });
    expect(strength).toBeEnabled();

    fireEvent.change(screen.getByRole('combobox', { name: /Primitive/i }), {
      target: { value: 'coordinate_swap' },
    });
    expect(screen.getByRole('spinbutton', { name: /Strength/i })).toBeDisabled();
    expect(screen.getByText(/ignored by coordinate swap/i)).toBeInTheDocument();
  });


  it('SWAPS WITH A TYPED TOKEN that never appeared in the readout', async () => {
    /**
     * A direction is `W_U[id]`, so any single token has one. The pinned set is
     * only what the readout SURFACED, and a swap target is usually a token that
     * is not in the top-k yet — asking whether it arrives is the experiment.
     * Restricting this form to what was on screen was a limit the server never
     * had.
     *
     * MUTATION CONTROLS:
     *   * ignore `typedPartner` in `chosenPartner` -> "swaps with a typed token" fails
     *   * ignore `typedDirection` in `chosen`      -> "the typed direction wins" fails
     */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      { token: ' Rome', ids: [4874], n_tokens: 1, usable: true, detail: 'One token — usable as a direction.' },
    ]);
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        // ONE pinned token, so the partner CANNOT have come from the pinned set.
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByRole('combobox', { name: /Primitive/i }), {
      target: { value: 'coordinate_swap' },
    });
    fireEvent.change(screen.getByTestId('intervention-partner-typed'), {
      target: { value: ' Rome' },
    });
    fireEvent.blur(screen.getByTestId('intervention-partner-typed'));
    await waitFor(() =>
      expect(screen.getByTestId('token-verdict')).toHaveTextContent(/id 4874/)
    );
    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));

    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.direction_token).toBe(' Paris');
    expect(sent.target_token).toBe(' Rome');
  });

  it('BLOCKS a token the vocabulary says is more than one, before the GPU', async () => {
    /**
     * The worker refuses a multi-token direction correctly — but only after a
     * 202 and a slot on a single-GPU queue that may sit behind a 45-minute fit.
     * The tokenizer answers it with no weights loaded.
     *
     * MUTATION CONTROL: drop the `rejected(chosen)` clause from `blocked` and
     * this fails — the button becomes enabled.
     */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      {
        token: 'Rome',
        ids: [4874, 883],
        n_tokens: 2,
        usable: false,
        detail: "2 tokens. A lens direction is defined for a SINGLE token. Try ' Rome' with a leading space — that is one token.",
      },
    ]);
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByTestId('intervention-direction-typed'), {
      target: { value: 'Rome' },
    });
    fireEvent.blur(screen.getByTestId('intervention-direction-typed'));

    await waitFor(() =>
      expect(screen.getByTestId('token-verdict')).toHaveTextContent(/SINGLE token/)
    );
    // THE HINT REACHES THE USER, not just the verdict. The leading space is the
    // cause almost every time and is invisible in a text box.
    expect(screen.getByTestId('token-verdict')).toHaveTextContent(/leading space/);
    expect(
      screen.getByRole('button', { name: /Run with control/i })
    ).toBeDisabled();
    expect(jlensApi.intervene as ReturnType<typeof vi.fn>).not.toHaveBeenCalled();
  });

  it('does NOT block when the check itself fails', async () => {
    /**
     * The endpoint being unreachable must not strand the form: the worker
     * refuses a bad direction anyway, so this is an early warning and not a
     * gate.
     *
     * MUTATION CONTROL: let the catch set a blocking state and this fails.
     */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('offline')
    );
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByTestId('intervention-direction-typed'), {
      target: { value: ' Rome' },
    });
    fireEvent.blur(screen.getByTestId('intervention-direction-typed'));
    await waitFor(() =>
      expect(jlensApi.checkTokens as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    expect(screen.getByRole('button', { name: /Run with control/i })).toBeEnabled();
  });


  it('sends the typed token VERBATIM, leading space and all', async () => {
    /**
     * The leading space is the character that makes ' Rome' a single token and
     * 'Rome' two. An implementation that trimmed the input sent 'Rome' to the
     * worker — the multi-token form the worker must refuse — immediately after
     * the verdict had approved ' Rome'. The check and the run would have been
     * examining different strings.
     *
     * MUTATION CONTROL: `.trim()` the typed token anywhere on the path and this
     * fails.
     */
    (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mockResolvedValue([
      { token: ' Rome', ids: [4874], n_tokens: 1, usable: true, detail: 'One token.' },
    ]);
    render(
      <InterventionCard
        modelId="m_1"
        prompt="The capital of France is"
        pinned={[' Paris']}
        layers={[10, 11]}
        artifactId={null}
      />
    );
    fireEvent.click(screen.getByRole('button', { name: /intervene/i }));
    fireEvent.change(screen.getByTestId('intervention-direction-typed'), {
      target: { value: ' Rome' },
    });
    fireEvent.blur(screen.getByTestId('intervention-direction-typed'));
    await waitFor(() =>
      expect(jlensApi.checkTokens as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    // CHECKED VERBATIM...
    expect(
      (jlensApi.checkTokens as ReturnType<typeof vi.fn>).mock.calls[0][1]
    ).toEqual([' Rome']);

    fireEvent.click(screen.getByRole('button', { name: /Run with control/i }));
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    // ...AND RUN VERBATIM. The two must be the same string or the verdict
    // describes something other than what was sent.
    expect(
      (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0]
        .direction_token
    ).toBe(' Rome');
  });

});

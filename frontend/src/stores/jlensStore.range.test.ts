/**
 * The layer range is a REQUEST parameter, not a display filter.
 *
 * `check_readout_budget` bounds positions x layers BEFORE capture, so narrowing
 * has to reach the server to be worth anything — reading every layer and then
 * hiding most of them pays the whole cost and calls it a saving.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * drop `layers` from the readout body     -> "sends the range" fails
 *   * send a range when none is set           -> "asks for every layer" fails
 *   * send the two bounds instead of the span -> "expands to every layer in it" fails
 */
import { describe, expect, it, vi, beforeEach } from 'vitest';

vi.mock('../api/jlens', () => ({
  jlensApi: {
    readout: vi.fn(),
    readoutResult: vi.fn(),
    listArtifacts: vi.fn().mockResolvedValue([]),
    intervene: vi.fn(),
  },
}));

import { jlensApi } from '../api/jlens';
import { useJLensStore } from './jlensStore';

function readoutReturns() {
  vi.mocked(jlensApi.readout).mockResolvedValue({
    task_id: 't1',
    model_id: 'm_1',
    status: 'queued',
  } as never);
  vi.mocked(jlensApi.readoutResult).mockResolvedValue({
    task_id: 't1',
    status: 'SUCCESS',
    readout: {
      meta: {
        kind: 'meta',
        model: 'org/m',
        types: ['LOGIT_LENS'],
        layers_by_type: { LOGIT_LENS: [4, 5, 6] },
        top_n: 4,
        prompt_len: 1,
      },
      tokens: [],
    },
  } as never);
}

beforeEach(() => {
  vi.clearAllMocks();
  useJLensStore.getState().reset();
  readoutReturns();
});

describe('the layer range reaches the server', () => {
  it('SENDS the range, expanded to every layer in it', async () => {
    /**
     * The endpoint takes an explicit list and has no notion of a range —
     * inventing one on the wire would be a miStudio-shaped field in a format
     * that is not ours to design (BR-029).
     *
     * MUTATION CONTROL: drop `layers` from the body, or send `[lo, hi]` rather
     * than the span, and this fails.
     */
    useJLensStore.setState({
      modelId: 'm_1',
      prompt: 'hello',
      layerRange: [4, 7],
    });
    await useJLensStore.getState().fetchReadout();

    expect(jlensApi.readout).toHaveBeenCalledTimes(1);
    const sent = vi.mocked(jlensApi.readout).mock.calls[0][0];
    expect(sent.layers).toEqual([4, 5, 6, 7]);
  });

  it('asks for EVERY layer when nothing is narrowed', async () => {
    /**
     * Absent, not an invented full range: `layers: null` is the endpoint's own
     * "all of them", and sending a computed span would pin the request to
     * whatever the client believed the model had.
     *
     * MUTATION CONTROL: always send a range and this fails.
     */
    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: null });
    await useJLensStore.getState().fetchReadout();
    const sent = vi.mocked(jlensApi.readout).mock.calls[0][0];
    expect(sent.layers).toBeUndefined();
  });

  it('sends nothing for a range whose ends have crossed', async () => {
    /** Better to read everything than to ask for an empty selection. */
    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: [9, 2] });
    await useJLensStore.getState().fetchReadout();
    const sent = vi.mocked(jlensApi.readout).mock.calls[0][0];
    expect(sent.layers).toBeUndefined();
  });

  it('survives a reload: the range persists with the rest of the setup', () => {
    useJLensStore.setState({ layerRange: [3, 9] });
    const persisted = JSON.parse(
      localStorage.getItem('miStudio-jlens') ?? '{"state":{}}',
    );
    expect(persisted.state.layerRange).toEqual([3, 9]);
  });
});

describe('the range belongs to the model it was chosen for', () => {
  it('is CLEARED when the model changes', () => {
    /**
     * A persisted range of 20-25 carried onto a 16-layer model makes the server
     * refuse every readout — and with `meta` null the picker is not mounted, so
     * its own "All layers" reset is unreachable and the panel is stuck with no
     * visible way out.
     *
     * MUTATION CONTROL: leave layerRange alone in setModelId and this fails.
     */
    useJLensStore.setState({
      modelId: 'm_big',
      layerRange: [20, 25],
      fullSpan: [0, 25],
    });
    useJLensStore.getState().setModelId('m_small', 'org/small');
    expect(useJLensStore.getState().layerRange).toBeNull();
    expect(useJLensStore.getState().fullSpan).toBeNull();
  });

  it('is KEPT when the same model is re-selected', () => {
    /** Re-picking the model you are already on is not a change. */
    useJLensStore.setState({ modelId: 'm_a', layerRange: [2, 5] });
    useJLensStore.getState().setModelId('m_a', 'org/a');
    expect(useJLensStore.getState().layerRange).toEqual([2, 5]);
  });

  it('learns the FULL span only from an unnarrowed read', async () => {
    /**
     * A narrowed re-read returns only the layers it asked for. Learning the
     * span from it ratchets the picker down: after reading L4-L5 the model
     * appears to offer only those and the clamp refuses anything wider.
     *
     * MUTATION CONTROL: set fullSpan from every response and this fails.
     */
    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: null });
    await useJLensStore.getState().fetchReadout();
    expect(useJLensStore.getState().fullSpan).toEqual([4, 6]);

    // THE NARROWED READ MUST RETURN A NARROWER AXIS, or learning from every
    // response gives the same answer as learning from unnarrowed ones and the
    // mutation survives — the fixture agreeing with both behaviours.
    vi.mocked(jlensApi.readoutResult).mockResolvedValue({
      task_id: 't1',
      status: 'SUCCESS',
      readout: {
        meta: {
          kind: 'meta',
          model: 'org/m',
          types: ['LOGIT_LENS'],
          layers_by_type: { LOGIT_LENS: [5] },
          top_n: 4,
          prompt_len: 1,
        },
        tokens: [],
      },
    } as never);
    useJLensStore.setState({ layerRange: [5, 5] });
    await useJLensStore.getState().fetchReadout();
    expect(useJLensStore.getState().fullSpan).toEqual([4, 6]);
  });
});

describe('what survives a reload', () => {
  /**
   * `fullSpan` is the only record of what the model OFFERS. `layerRange` is
   * what was asked for, and the meta axis of a narrowed read is what came
   * back — neither of them can widen.
   *
   * So persisting the range without the span it bounds is a RATCHET: reload
   * with [5,5] saved, fullSpan gone, and the picker rebuilds its bounds from
   * the narrowed axis. L5-L5 becomes the whole model. Every widening is then
   * clamped straight back to 5, and the only escape is clearing storage —
   * which reads as a bug in the picker, not in what was saved.
   *
   * MUTATION CONTROL: drop `fullSpan: state.fullSpan` from `partialize` and
   * "carries the FULL SPAN" fails. It survived a round without this test.
   */
  const persisted = () => {
    const raw = localStorage.getItem('miStudio-jlens');
    return raw ? JSON.parse(raw).state : null;
  };

  it('carries the FULL SPAN, not only the narrowed range', () => {
    useJLensStore.setState({
      modelId: 'm_1',
      prompt: 'hello',
      fullSpan: [0, 25],
      layerRange: [5, 5],
    });

    const saved = persisted();
    expect(saved).not.toBeNull();
    // THE RANGE ALONE IS NOT ENOUGH. Both assertions matter: the first is the
    // fix, the second proves the fixture would have noticed either going
    // missing rather than passing on an empty object.
    expect(saved.fullSpan).toEqual([0, 25]);
    expect(saved.layerRange).toEqual([5, 5]);
  });

  it('does NOT persist a span that a narrowed read invented', async () => {
    /**
     * Belt and braces on the same defect from the other side: if a narrowed
     * read ever wrote fullSpan, persisting it would make the ratchet
     * permanent instead of merely per-session.
     */
    useJLensStore.setState({ modelId: 'm_1', prompt: 'hello', layerRange: null, fullSpan: null });
    await useJLensStore.getState().fetchReadout();
    expect(persisted().fullSpan).toEqual([4, 6]);
  });
});

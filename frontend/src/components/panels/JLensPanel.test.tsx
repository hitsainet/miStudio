/**
 * J-Lens panel — the four ways this port could look right and be wrong.
 *
 * The reference implementation (`0xcc/brds/JSpacePanel.jsx`) hardcodes three
 * model-specific constants and ships a fixture generator. Every one of them
 * renders a complete, plausible panel after a straight paste:
 *
 *   LAYERS = 21 layers at 0,5,...,100   -> a grid of the wrong height
 *   BAND   = { 40, 90 }                 -> Sonnet-4.5 boundaries on any model
 *   TOP_N  = 8                          -> a mis-scaled heatmap ramp
 *   FIXTURES/buildFixture               -> synthetic readouts indistinguishable
 *                                          from real ones
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * hardcode the grid axis to a constant             -> "layer axis" fails
 *   * enable Jacobian regardless of meta.types         -> "disablement" fails
 *   * default bandReport to { 40, 90 }                 -> "bands" fails
 *   * hardcode topN in rankColor                       -> "top-n ramp" fails
 *   * drop the interpretability caveat                 -> "framing" fails
 *   * clear meta/tokens at the start of fetchReadout   -> "refetch" fails
 */

import fs from 'node:fs';
import path from 'node:path';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, waitFor, act, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { JLensPanel } from './JLensPanel';
import { useJLensStore } from '../../stores/jlensStore';
import { rankColor } from '../jlens/utils';
import type { LensType, ReadoutResponse } from '../../types/jlens';
import { ABSENCE_CAVEAT, READOUT_LIMITS } from '../../config/jspaceClaims';

/** Match a required caveat verbatim rather than by a paraphrase of it. */
function escapeRe(s: string) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

vi.mock('../../stores/modelsStore');
vi.mock('../../api/jlens', () => ({
  jlensApi: {
    readout: vi.fn(),
    readoutResult: vi.fn(),
    listArtifacts: vi.fn().mockResolvedValue([]),
    intervene: vi.fn(),
    previewRepo: vi.fn(),
    acquire: vi.fn(),
    publish: vi.fn(),
  },
}));

// The panel polls a queued intervention to a terminal state; without this the
// poll would reach the real client and the test would hang on a timer.
vi.mock('../../api/models', () => ({ getTaskStatus: vi.fn() }));

/**
 * The readout is a TWO-STEP contract now: POST returns a task id, and the
 * result arrives by polling. Mocking it as a single call would test a shape
 * the server no longer has — the readout was made asynchronous because a
 * synchronous one 502'd at the ingress on a real model.
 */
function mockReadout(response: ReadoutResponse) {
  (jlensApi.readout as ReturnType<typeof vi.fn>).mockResolvedValue({
    task_id: 't1',
    model_id: 'm_lfm2',
    status: 'queued',
  });
  (jlensApi.readoutResult as ReturnType<typeof vi.fn>).mockResolvedValue({
    task_id: 't1',
    status: 'SUCCESS',
    readout: response,
  });
}

// ResponsiveContainer measures its parent, and jsdom reports every element as
// 0x0 — so recharts renders nothing at all and a chart assertion would pass
// vacuously against a broken series. Fixed dimensions make the SVG real.
vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts');
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: React.ReactElement }) => (
      <actual.ResponsiveContainer width={800} height={300}>
        {children}
      </actual.ResponsiveContainer>
    ),
  };
});

import { jlensApi } from '../../api/jlens';
import { getTaskStatus } from '../../api/models';
import { useModelsStore } from '../../stores/modelsStore';

const MODELS = [
  { id: 'm_lfm2', name: 'LFM2.5-1.2B-Instruct', status: 'ready' },
  { id: 'm_gemma', name: 'gemma-2-2b-it', status: 'ready' },
];

/**
 * A readout with an arbitrary layer axis. `layers` is passed explicitly at every
 * call site so no test can accidentally agree with a hardcoded default — the
 * "fixtures agree by construction" trap.
 */
function makeReadout(
  layers: number[],
  types: LensType[] = ['LOGIT_LENS'],
  topN = 4
): ReadoutResponse {
  const words = ['Paris', 'France', 'capital', 'city', 'the', 'a', 'of', 'is'];
  const tokens = ['The', ' capital', ' of', ' France', ' is'].map((tok, pos) => ({
    kind: 'token' as const,
    position: pos,
    token: tok,
    id: 1000 + pos,
    is_generated: false,
    results: types.map((type) => ({
      type,
      top_tokens: layers.map((_, li) =>
        Array.from({ length: topN }, (_, k) => words[(li + k + pos) % words.length])
      ),
      top_probs: layers.map((_, li) =>
        // Later layers are confident; the earliest are diffuse, which is the
        // real shape and what the "expected to be uninterpretable" marking keys on.
        Array.from({ length: topN }, (_, k) =>
          (li / Math.max(layers.length - 1, 1)) * 0.8 * (1 - k / (topN + 1))
        )
      ),
    })),
  }));

  return {
    meta: {
      kind: 'meta',
      model: 'test-model',
      types,
      layers_by_type: Object.fromEntries(types.map((t) => [t, layers])),
      top_n: topN,
      prompt_len: tokens.length,
    },
    tokens,
  };
}

function seed(response: ReadoutResponse) {
  act(() => {
    useJLensStore.setState({
      modelId: 'm_lfm2',
      prompt: 'The capital of France is',
      meta: response.meta,
      tokens: response.tokens,
      provenance: { artifact_id: null },
      lensMode: 'LOGIT_LENS',
      selPos: 0,
      selLayerIdx: 0,
      pinned: [],
      hover: null,
      isLoading: false,
      error: null,
      bandReport: null,
    });
  });
}

/** Grid body rows = one per layer, plus the token footer row. */
function gridRowCount(container: HTMLElement): number {
  return container.querySelectorAll('table tbody tr').length - 1;
}

beforeEach(() => {
  vi.clearAllMocks();
  act(() => useJLensStore.getState().reset());
  // Selector-aware: the panel subscribes with selectors so a models-store tick
  // does not re-render the readout grid.
  const modelsState = { models: MODELS, fetchModels: vi.fn() };
  const hook = useModelsStore as unknown as ReturnType<typeof vi.fn>;
  hook.mockImplementation(
    (selector?: (s: typeof modelsState) => unknown) =>
      selector ? selector(modelsState) : modelsState
  );
  // The panel reads the actions off the store STATICALLY for its mount-once
  // fetch, so the mock needs getState too — keying that effect on the hook's
  // return identity is what makes a one-shot fetch loop.
  hook.getState = () => modelsState;
});

describe('layer axis is model-derived', () => {
  it.each([
    ['16-layer hybrid', Array.from({ length: 16 }, (_, i) => i)],
    ['26-layer dense', Array.from({ length: 26 }, (_, i) => i)],
    ['sparse axis', [0, 3, 7, 11, 15]],
  ])('follows layers_by_type for a %s model', (_label, layers) => {
    const { container } = render(<JLensPanel />);
    seed(makeReadout(layers));
    expect(gridRowCount(container)).toBe(layers.length);
  });

  it('labels rows with ABSOLUTE layer numbers, not row indices', () => {
    // A sparse axis is the only shape that distinguishes "layer" from "index".
    render(<JLensPanel />);
    seed(makeReadout([0, 3, 7, 11, 15]));
    expect(screen.getAllByText('L15').length).toBeGreaterThan(0);
    expect(screen.getAllByText('L11').length).toBeGreaterThan(0);
    expect(screen.queryByText('L4')).toBeNull();
  });
});

describe('positions are looked up by position, not by array index', () => {
  /**
   * Every other fixture here numbers positions 0..n-1, where index and position
   * agree by construction — so an index-based lookup passes all of them. The
   * wire format permits a readout over a SUBSET of positions, and that is the
   * only shape that tells the two apart.
   */
  function slicedReadout() {
    const base = makeReadout([0, 1, 2]);
    base.tokens = base.tokens.slice(0, 3).map((t, i) => ({
      ...t,
      position: i * 2, // 0, 2, 4
    }));
    base.meta.prompt_len = 5;
    return base;
  }

  it('shows the token at the SELECTED POSITION in the by-layer rail', () => {
    render(<JLensPanel />);
    seed(slicedReadout());
    act(() => useJLensStore.setState({ selPos: 4 }));

    // Array index 4 does not exist; position 4 is the third token, " of".
    const header = screen.getByText(/By layer · position 4/);
    expect(header.textContent).toContain('·of');
  });

  it('clamps a vanished position onto a real one rather than to an index', async () => {
    seed(makeReadout([0, 1, 2]));
    act(() => useJLensStore.setState({ selPos: 9 }));

    mockReadout(slicedReadout());
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    // 4 is the last real position; 2 would be the last array INDEX.
    expect(useJLensStore.getState().selPos).toBe(4);
  });
});

describe('before any readout', () => {
  it('offers the logit lens and names what the other two need', () => {
    // The logit lens needs no artifact and works on any loaded model, so
    // reporting all three as unavailable said "nothing works" when the
    // default path always does.
    render(<JLensPanel />);

    expect(screen.getByRole('button', { name: /Logit/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Jacobian/ })).toBeDisabled();
    expect(screen.getByRole('button', { name: /Diff/ })).toBeDisabled();
    expect(screen.getByText(/Needs a validated J-lens artifact/i)).toBeInTheDocument();

    // Scoped to the tab group: the provenance strip legitimately says "no
    // readout yet" — that is a statement about provenance, not about whether
    // a lens is usable, and an unscoped query conflates the two.
    const logit = screen.getByRole('button', { name: /Logit/ });
    expect(logit.parentElement?.textContent).not.toMatch(/No readout yet/i);
  });

  it('offers a way to fit the lens it tells the user to fit', async () => {
    // REACHABILITY, not existence. The panel said "fit one to enable it" while
    // the only routes in were REST and MCP, so the remedy it named could not be
    // performed from the product. Asserting the card renders HERE — inside the
    // panel — is what makes deleting <FitLensCard/> turn this red; a test that
    // rendered the card directly would pass against a card nobody can reach.
    render(<JLensPanel />);

    const open = screen.getByRole('button', { name: /fit a lens/i });
    // Disabled until a model is chosen: a fit is per-model and there is nothing
    // sensible to fit against.
    expect(open).toBeDisabled();

    act(() => useJLensStore.setState({ modelId: 'm_lfm2' }));
    expect(screen.getByRole('button', { name: /fit a lens/i })).toBeEnabled();

    await userEvent.click(screen.getByRole('button', { name: /fit a lens/i }));
    expect(screen.getByLabelText(/one prompt per line/i)).toBeInTheDocument();
    // The corpus is the caller's choice and is recorded in the recipe (BR-007),
    // so the form must ask for it rather than defaulting one server-side.
    expect(screen.getByLabelText(/corpus name/i)).toBeInTheDocument();
  });
});

describe('lens-mode disablement is derived from what the stream carries', () => {
  it('disables Jacobian and Diff with a stated reason on a logit-only readout', () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS']));

    const jacobian = screen.getByRole('button', { name: /Jacobian/ });
    const diff = screen.getByRole('button', { name: /Diff/ });

    expect(jacobian).toBeDisabled();
    expect(diff).toBeDisabled();
    expect(screen.getByText(/No validated J-lens artifact/i)).toBeInTheDocument();
    expect(screen.getByText(/only one is present/i)).toBeInTheDocument();
  });

  it('enables them when the readout actually carries both lenses', () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['JACOBIAN_LENS', 'LOGIT_LENS']));

    expect(screen.getByRole('button', { name: /Jacobian/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Diff/ })).toBeEnabled();
  });

  it('never renders logit data under a Jacobian label', () => {
    // With only LOGIT_LENS transported, clicking Jacobian must not switch mode.
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS']));

    fireEvent.click(screen.getByRole('button', { name: /Jacobian/ }));
    expect(useJLensStore.getState().lensMode).toBe('LOGIT_LENS');
  });
});

describe('bands are earned, never defaulted', () => {
  it('draws no bands and says why when no report exists', () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2]));

    expect(screen.getByTestId('jlens-no-bands')).toHaveTextContent(
      /No band report for this model/i
    );
    expect(screen.queryByText('Workspace')).toBeNull();
    expect(screen.queryByText('Motor')).toBeNull();
  });

  it('has no band constant anywhere in the feature source', () => {
    // BR-002 requires porting the paper's L40/L90 to be impossible by
    // construction, so the guard is over the SOURCE, not the render.
    const roots = [
      path.resolve(__dirname, '../jlens'),
      path.resolve(__dirname, 'JLensPanel.tsx'),
      path.resolve(__dirname, '../../stores/jlensStore.ts'),
      path.resolve(__dirname, '../../types/jlens.ts'),
    ];
    const files = roots.flatMap((p) =>
      fs.statSync(p).isDirectory()
        ? fs.readdirSync(p).map((f) => path.join(p, f))
        : [p]
    );
    for (const file of files) {
      if (file.endsWith('.test.tsx') || file.endsWith('.test.ts')) continue;
      const code = stripComments(fs.readFileSync(file, 'utf8'));
      // Two shapes, because renaming the constant is the obvious way to get
      // past a name-only guard: an explicit boundary field with a number, and
      // any `BAND`-ish binding holding numeric literals.
      expect(code, `${file} defines a band boundary`).not.toMatch(
        /(workspace|motor|sensory)\w*\s*[:=]\s*-?\d/i
      );
      expect(code, `${file} binds a band constant`).not.toMatch(
        /\bband\w*\s*=\s*\{[^}]*\d/i
      );
    }
  });
});

describe('the colour ramp is scaled by the top-n the server sent', () => {
  it('gives the same rank different weight at different top-n', () => {
    // Hardcoding TOP_N=8 makes these identical, and the heatmap mis-scales
    // silently on any readout with a different depth.
    expect(rankColor(3, 4)).not.toBe(rankColor(3, 40));
  });

  it('renders the top-n the readout declares in the hover hint', () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 5));
    expect(screen.getByText(/full top-5 readout/i)).toBeInTheDocument();
  });
});

describe('interpretability framing', () => {
  beforeEach(() => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2]));
  });

  it('carries the evidence rung and what would raise it', () => {
    expect(screen.getByText(/Rung 0 · Readout/)).toBeInTheDocument();
    expect(screen.getByText(/coordinate swap with a matched control/i)).toBeInTheDocument();
  });

  it('states the single-token limitation', () => {
    expect(screen.getByText(new RegExp(escapeRe(READOUT_LIMITS)))).toBeInTheDocument();
  });

  it('states that an uninterpretable readout is not a null result', () => {
    expect(screen.getByText(/is not a null result/i)).toBeInTheDocument();
  });

  it('states that absence of a signal is not evidence of absence', () => {
    // Asserted against the SHARED constant, not a retyped sentence. A test
    // holding its own copy drifts alongside the component and then passes
    // against weakened copy.
    expect(screen.getByText(new RegExp(escapeRe(ABSENCE_CAVEAT)))).toBeInTheDocument();
  });

  it('says explicitly that the logit lens involves no artifact', () => {
    expect(screen.getByTestId('jlens-no-artifact')).toHaveTextContent(
      /no artifact involved/i
    );
  });

  it('marks diffuse readouts rather than presenting them as content', () => {
    expect(screen.getAllByText('diffuse').length).toBeGreaterThan(0);
  });
});

describe('pinning turns the grid into a rank heatmap', () => {
  it('pins from the hover list and shows the pinned token in the rail', async () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2]));

    const cells = document.querySelectorAll('table tbody tr td');
    fireEvent.mouseEnter(cells[1]);

    // Scoped to the hover panel: the prompt strip renders buttons with the same
    // token text, and an unscoped query pins by clicking the wrong control.
    const detail = screen.getByTestId('jlens-hover-detail');
    const hoverTokens = await waitFor(() => {
      const found = Array.from(detail.querySelectorAll('button'));
      expect(found.length).toBeGreaterThan(0);
      return found;
    });
    fireEvent.click(hoverTokens[0]);

    expect(useJLensStore.getState().pinned).toEqual([
      hoverTokens[0].textContent?.replace(/^·/, ' '),
    ]);
  });
});

describe('a refetch never blanks the readout', () => {
  it('keeps meta, tokens and pins across a slow refetch', async () => {
    const first = makeReadout([0, 1, 2, 3]);
    seed(first);
    act(() => useJLensStore.setState({ pinned: ['Paris'] }));

    let resolve!: (r: ReadoutResponse) => void;
    (jlensApi.readout as ReturnType<typeof vi.fn>).mockReturnValue(
      new Promise<ReadoutResponse>((r) => {
        resolve = r;
      })
    );

    const pending = useJLensStore.getState().fetchReadout();

    // Mid-flight: the readout is still on screen and the pins survive. Clearing
    // state here is what unmounted the grid and dropped the user's work.
    expect(useJLensStore.getState().isLoading).toBe(true);
    expect(useJLensStore.getState().tokens.length).toBe(first.tokens.length);
    expect(useJLensStore.getState().pinned).toEqual(['Paris']);

    await act(async () => {
      resolve(makeReadout([0, 1, 2, 3]));
      await pending;
    });

    expect(useJLensStore.getState().pinned).toEqual(['Paris']);
  });

  it('clamps a stale selection into a shorter new readout', async () => {
    seed(makeReadout(Array.from({ length: 26 }, (_, i) => i)));
    act(() => useJLensStore.setState({ selLayerIdx: 25, selPos: 4 }));

    mockReadout(makeReadout(Array.from({ length: 16 }, (_, i) => i)));

    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    expect(useJLensStore.getState().selLayerIdx).toBe(15);
  });
});

describe('a mode the new readout cannot serve is not left selected', () => {
  it('falls back to the logit lens when the Jacobian is no longer carried', async () => {
    seed(makeReadout([0, 1, 2], ['JACOBIAN_LENS', 'LOGIT_LENS']));
    act(() => useJLensStore.setState({ lensMode: 'DIFF' }));

    mockReadout(makeReadout([0, 1, 2], ['LOGIT_LENS']));
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    // Leaving DIFF selected renders an empty grid, which reads as "this lens
    // found nothing" while the disabled tab says "this lens is not present".
    expect(useJLensStore.getState().lensMode).toBe('LOGIT_LENS');
  });

  it('keeps a mode the new readout still carries', async () => {
    seed(makeReadout([0, 1, 2], ['JACOBIAN_LENS', 'LOGIT_LENS']));
    act(() => useJLensStore.setState({ lensMode: 'DIFF' }));

    mockReadout(makeReadout([0, 1, 2], ['JACOBIAN_LENS', 'LOGIT_LENS']));
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    expect(useJLensStore.getState().lensMode).toBe('DIFF');
  });
});

describe('the panel sends the request it appears to send', () => {
  /**
   * Asserts the PAYLOAD and the CALL COUNT, not merely that a call happened —
   * "was called" passes against a call that sends the wrong model, an empty
   * prompt, or a lens type the server cannot serve without an artifact.
   */
  it('submits the model and prompt exactly once, requesting the logit lens', async () => {
    const user = userEvent.setup();
    mockReadout(makeReadout([0, 1, 2]));
    render(<JLensPanel />);

    await user.selectOptions(screen.getByRole('combobox'), 'm_gemma');
    await user.type(screen.getByPlaceholderText(/capital of France/), 'hello');
    await user.click(screen.getByRole('button', { name: /Read out/ }));

    expect(jlensApi.readout).toHaveBeenCalledTimes(1);
    expect(jlensApi.readout).toHaveBeenCalledWith({
      model_id: 'm_gemma',
      prompt: 'hello',
      // Never JACOBIAN_LENS: the server refuses it without an artifact_id, and
      // requesting it anyway surfaces a 422 the user cannot act on.
      types: ['LOGIT_LENS'],
    });
  });

  it('does not fire at all without a model', async () => {
    const user = userEvent.setup();
    render(<JLensPanel />);

    await user.type(screen.getByPlaceholderText(/capital of France/), 'hello');
    expect(screen.getByRole('button', { name: /Read out/ })).toBeDisabled();
    expect(jlensApi.readout).not.toHaveBeenCalled();
  });
});

describe('pinning is reachable without a pointer', () => {
  it('shows the selected cell top-k when nothing is hovered', () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2]));

    const detail = screen.getByTestId('jlens-hover-detail');
    expect(detail.textContent).toMatch(/\(selected\)/);
    // Hover is pointer-only; without this fallback the top-k list — and so
    // pinning, the panel's core interaction — had no keyboard path.
    expect(detail.querySelectorAll('button').length).toBeGreaterThan(0);
  });

  it('pins from the selected-cell list', () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2]));

    const first = screen
      .getByTestId('jlens-hover-detail')
      .querySelector('button') as HTMLButtonElement;
    fireEvent.click(first);

    expect(useJLensStore.getState().pinned.length).toBe(1);
  });
});

describe('the layer clamp follows the lens the panel will actually read', () => {
  /**
   * Two lens types may carry DIFFERENT layer counts — a Jacobian artifact can
   * be fitted over a subset of layers while the logit lens covers all of them.
   * Every other fixture here gives both types the same axis, where clamping
   * against either one gives the same answer: the "agree by construction" trap
   * that let this survive its first mutation control.
   */
  function unevenAxes(): ReadoutResponse {
    const logit = Array.from({ length: 12 }, (_, i) => i);
    const jac = [0, 1, 2];
    const base = makeReadout(logit, ['JACOBIAN_LENS', 'LOGIT_LENS']);
    base.meta.layers_by_type = { JACOBIAN_LENS: jac, LOGIT_LENS: logit };
    base.tokens = base.tokens.map((t) => ({
      ...t,
      results: t.results.map((r) =>
        r.type === 'JACOBIAN_LENS'
          ? {
              ...r,
              top_tokens: r.top_tokens.slice(0, jac.length),
              top_probs: r.top_probs.slice(0, jac.length),
            }
          : r
      ),
    }));
    return base;
  }

  it('clamps into the Jacobian axis when the Jacobian lens is selected', async () => {
    seed(makeReadout([0, 1, 2], ['JACOBIAN_LENS', 'LOGIT_LENS']));
    act(() => useJLensStore.setState({ lensMode: 'JACOBIAN_LENS', selLayerIdx: 11 }));

    mockReadout(unevenAxes());
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    // The Jacobian axis has 3 layers; clamping against the 12-layer logit axis
    // leaves an index that reads past the end of every Jacobian slice row.
    expect(useJLensStore.getState().selLayerIdx).toBe(2);
  });
});

describe('trajectories survive tokens that collide with chart internals', () => {
  /**
   * A chart row is a flat object carrying the x value under `layer`. Keying
   * series by token text puts token data in the SAME namespace, so pinning the
   * token "layer" overwrites the x value with a rank — every point plots at
   * x = its own rank, and the result still looks like a chart.
   *
   * Every other fixture here pins tokens that happen not to collide, which is
   * why this needs its own vocabulary.
   */
  function collidingReadout(): ReadoutResponse {
    const base = makeReadout([0, 10, 20, 30]);
    base.tokens = base.tokens.map((t) => ({
      ...t,
      results: t.results.map((r) => ({
        ...r,
        top_tokens: r.top_tokens.map((row) => ['layer', '.', ...row.slice(2)]),
      })),
    }));
    return base;
  }

  it('keeps the layer axis when a pinned token is named "layer"', () => {
    render(<JLensPanel />);
    seed(collidingReadout());
    act(() => useJLensStore.setState({ pinned: ['layer', '.'] }));

    const curves = document.querySelectorAll('.recharts-line-curve');
    expect(curves.length).toBe(2);
    for (const curve of curves) {
      expect(curve.getAttribute('d')).toMatch(/^M/);
    }

    // The x axis must still be the LAYER axis. With the collision, the ticks
    // become ranks (1..top_n) instead of 0,10,20,30.
    const ticks = Array.from(
      document.querySelectorAll('.recharts-xAxis .recharts-cartesian-axis-tick-value')
    ).map((t) => t.textContent);
    expect(ticks).toContain('30');
  });
});

describe('request hygiene', () => {
  it('drops a slow earlier response in favour of a fast later one', async () => {
    seed(makeReadout([0, 1]));
    const store = useJLensStore.getState();

    let resolveSlow!: (r: ReadoutResponse) => void;
    (jlensApi.readout as ReturnType<typeof vi.fn>)
      .mockReturnValueOnce(
        new Promise<ReadoutResponse>((r) => {
          resolveSlow = r;
        })
      )
      .mockResolvedValueOnce(makeReadout([0, 1, 2, 3, 4, 5, 6]));

    const slow = store.fetchReadout();
    await act(async () => {
      await store.fetchReadout();
    });
    expect(useJLensStore.getState().tokens.length).toBeGreaterThan(0);
    const afterFast = useJLensStore.getState().meta?.layers_by_type.LOGIT_LENS;

    await act(async () => {
      resolveSlow(makeReadout([0, 1]));
      await slow;
    });

    // The stale response must not land: settling on the superseded readout is
    // indistinguishable from the newer request having returned that data.
    expect(useJLensStore.getState().meta?.layers_by_type.LOGIT_LENS).toEqual(afterFast);
  });

  it('refuses a prompt longer than the server accepts, without a round trip', async () => {
    act(() =>
      useJLensStore.setState({ modelId: 'm_lfm2', prompt: 'x'.repeat(8001) })
    );
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    expect(jlensApi.readout).not.toHaveBeenCalled();
    expect(useJLensStore.getState().error).toMatch(/at most 8000/);
  });

  it('drops pins and the readout when the model changes', () => {
    seed(makeReadout([0, 1, 2]));
    act(() => useJLensStore.setState({ pinned: ['Paris'] }));

    act(() => useJLensStore.getState().setModelId('m_gemma'));

    // Pins are token strings from the previous model's vocabulary; carried
    // across they draw empty lines that look like a measured absence.
    expect(useJLensStore.getState().pinned).toEqual([]);
    expect(useJLensStore.getState().meta).toBeNull();
  });

  it('keeps pins when the model is re-selected unchanged', () => {
    seed(makeReadout([0, 1, 2]));
    act(() => useJLensStore.setState({ pinned: ['Paris'] }));

    act(() => useJLensStore.getState().setModelId('m_lfm2'));

    expect(useJLensStore.getState().pinned).toEqual(['Paris']);
  });
});

describe('nothing synthetic ships', () => {
  it('has no fixture generator anywhere under src/', () => {
    const offenders: string[] = [];
    walk(path.resolve(__dirname, '../..'), (file) => {
      if (!/\.(ts|tsx)$/.test(file)) return;
      if (/\.test\.tsx?$/.test(file)) return;
      const code = stripComments(fs.readFileSync(file, 'utf8'));
      if (/\b(FIXTURES|buildFixture|scoreAt)\b/.test(code)) offenders.push(file);
    });
    expect(offenders).toEqual([]);
  });
});

function walk(dir: string, visit: (file: string) => void) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) walk(full, visit);
    else visit(full);
  }
}

/**
 * Strip comments before scanning.
 *
 * Without this, the guards flag the very docblocks that WARN about the
 * constants they forbid — a self-defeating assertion that forces the
 * explanation out of the code. (Feature 022 shipped exactly this bug twice.)
 */
function stripComments(code: string): string {
  return code.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
}


describe('the readout is asynchronous because it measurably had to be', () => {
  /**
   * Bound synchronously, /jlens/readout 502'd at the ingress TWICE on a real
   * model — 64.9s and 54.0s against nginx's 60s ceiling — because a J-space
   * readout needs the whole model resident for its forward pass.
   *
   * Raising the proxy timeout would not bound it: readout cost is
   * O(positions x layers x top_n) ON TOP of the load. So the contract is
   * queue-and-poll, and these pin that contract rather than the timeout.
   */
  it('queues, polls, and only then applies the readout', async () => {
    const response = makeReadout([0, 1, 2]);
    (jlensApi.readout as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 't-async',
      model_id: 'm_lfm2',
      status: 'queued',
    });
    (jlensApi.readoutResult as ReturnType<typeof vi.fn>)
      .mockResolvedValueOnce({ task_id: 't-async', status: 'PROGRESS', stage: 'loading_model' })
      .mockResolvedValueOnce({ task_id: 't-async', status: 'SUCCESS', readout: response });

    act(() =>
      useJLensStore.setState({ modelId: 'm_lfm2', prompt: 'hello', artifacts: [] })
    );
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    expect(jlensApi.readout).toHaveBeenCalledTimes(1);
    expect(jlensApi.readoutResult).toHaveBeenCalledWith('t-async');
    expect(useJLensStore.getState().tokens.length).toBe(response.tokens.length);
    expect(useJLensStore.getState().stage).toBeNull();
  });

  it('reports a FAILED task by its reason, never as an empty readout', async () => {
    (jlensApi.readout as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 't-fail',
      model_id: 'm_lfm2',
      status: 'queued',
    });
    (jlensApi.readoutResult as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 't-fail',
      status: 'FAILURE',
      error: 'google/gemma-2-2b-it is not downloaded locally.',
    });

    act(() =>
      useJLensStore.setState({ modelId: 'm_lfm2', prompt: 'hello', artifacts: [] })
    );
    await act(async () => {
      await useJLensStore.getState().fetchReadout();
    });

    // An empty readout is indistinguishable from a real one with no content —
    // the failure this whole feature is built to avoid.
    expect(useJLensStore.getState().error).toMatch(/not downloaded/);
    expect(useJLensStore.getState().meta).toBeNull();
    expect(useJLensStore.getState().isLoading).toBe(false);
  });
})

describe('a mounted artifact is enough to select the Jacobian lens', () => {
  it('says "fit one" and disables the lens when nothing is mounted', () => {
    render(<JLensPanel />);
    act(() => useJLensStore.setState({ modelId: 'm_lfm2', artifacts: [] }));

    expect(screen.getByText(/Fit one to enable it/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /Jacobian/ })).toBeDisabled();
  });

  it('enables Jacobian and Diff from the artifact alone, before any readout', () => {
    // The tab used to be disabled until a readout carried JACOBIAN_LENS — but
    // `fetchReadout` requests both lens types whenever the model has an
    // artifact, so the tab was gated on the absence of a stream that the click
    // itself produces. Observed on the cluster: a published, validated 21 MB
    // artifact sat in the strip directly below a tab telling the user to fit one.
    //
    // MUTATION CONTROL: drop the `hasArtifact` early return in
    // `modeAvailability`, or drop the prop from the LensModeTabs call, and this
    // fails.
    render(<JLensPanel />);
    act(() =>
      useJLensStore.setState({
        modelId: 'm_lfm2',
        modelRepoId: 'google/gemma-2-2b-it',
        artifacts: [
          {
            slug: 'gemma-2-2b-it',
            directory: '/data/jlens/gemma-2-2b-it',
            lens_file: 'gemma-2-2b-it_jacobian_lens.pt',
            size_bytes: 21_235_717,
            has_config: true,
          },
        ],
      })
    );

    expect(screen.getByRole('button', { name: /Jacobian/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Diff/ })).toBeEnabled();
    expect(screen.queryByText(/Fit one to enable it/i)).not.toBeInTheDocument();
  });

  it('still disables the lens a completed readout did NOT carry', () => {
    // Enabling on artifact presence promises the REQUEST, not the result. Once
    // a stream exists it is the authority again — otherwise a readout that came
    // back logit-only would leave a Jacobian tab selectable and rendering logit
    // data under a Jacobian label (BR-019).
    render(<JLensPanel />);
    act(() =>
      useJLensStore.setState({
        modelId: 'm_lfm2',
        modelRepoId: 'google/gemma-2-2b-it',
        artifacts: [
          {
            slug: 'gemma-2-2b-it',
            directory: '/data/jlens/gemma-2-2b-it',
            lens_file: 'gemma-2-2b-it_jacobian_lens.pt',
            size_bytes: 21_235_717,
            has_config: true,
          },
        ],
      })
    );
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS']));

    expect(screen.getByRole('button', { name: /Jacobian/ })).toBeDisabled();
  });
});

describe('the setup survives a refresh, and Clear forgets it', () => {
  it('persists the readout as well as the setup', () => {
    // Losing a readout on refresh is the wrong trade: it costs a model load
    // and up to a minute of GPU, and re-running it to see what you were
    // already looking at is worse than the risk of a stale grid. That risk is
    // handled by `readoutPrompt`, not avoided by throwing the results away.
    //
    // MUTATION CONTROL: drop meta/tokens from partialize and this fails.
    const persisted = (
      useJLensStore as unknown as {
        persist: { getOptions: () => { partialize: (s: unknown) => object } };
      }
    ).persist.getOptions().partialize(useJLensStore.getState()) as Record<string, unknown>;

    for (const key of ['modelId', 'prompt', 'lensMode', 'pinned', 'meta', 'tokens']) {
      expect(persisted).toHaveProperty(key);
    }
    // The prompt that PRODUCED the grid travels with it, or a restored readout
    // cannot say whether it still describes what is in the prompt box.
    expect(persisted).toHaveProperty('readoutPrompt');
  });

  it('drops an oversized readout rather than losing the whole entry', () => {
    // localStorage throws on overflow and zustand's persist middleware loses
    // the ENTIRE entry when it does — so an 8000-character prompt over 26
    // layers would take the model and prompt down with it.
    //
    // MUTATION CONTROL: remove the size guard and this fails.
    const huge = Array.from({ length: 400 }, (_, p) => ({
      kind: 'token' as const,
      position: p,
      token: 't',
      id: p,
      is_generated: false,
      results: [
        {
          type: 'LOGIT_LENS' as LensType,
          top_tokens: Array.from({ length: 26 }, () =>
            Array.from({ length: 8 }, () => 'x'.repeat(40))
          ),
          top_probs: Array.from({ length: 26 }, () =>
            Array.from({ length: 8 }, () => 0.1)
          ),
        },
      ],
    }));
    act(() =>
      useJLensStore.setState({ modelId: 'm_lfm2', prompt: 'p', tokens: huge })
    );

    const persisted = (
      useJLensStore as unknown as {
        persist: { getOptions: () => { partialize: (s: unknown) => object } };
      }
    ).persist.getOptions().partialize(useJLensStore.getState()) as Record<string, unknown>;

    expect(persisted).not.toHaveProperty('tokens');
    // The setup still survives, which is the point of dropping only the grid.
    expect(persisted.modelId).toBe('m_lfm2');
  });

  it('shows a restored prompt in the input, not an empty box', () => {
    // The input is component state seeded from the store, so persistence alone
    // is not enough — a restored prompt that does not appear in the field is
    // invisible to the user and unre-submittable.
    act(() =>
      useJLensStore.setState({ modelId: 'm_lfm2', prompt: 'The smell of a rose is' })
    );
    render(<JLensPanel />);

    expect(screen.getByDisplayValue('The smell of a rose is')).toBeInTheDocument();
  });

  it('Clear empties the store AND the visible input', async () => {
    act(() =>
      useJLensStore.setState({
        modelId: 'm_lfm2',
        modelRepoId: 'org/m',
        prompt: 'The capital of France is',
        pinned: [' Paris'],
        lensMode: 'LOGIT_LENS',
      })
    );
    render(<JLensPanel />);
    expect(screen.getByDisplayValue('The capital of France is')).toBeInTheDocument();

    await userEvent.click(screen.getByRole('button', { name: /clear/i }));

    expect(useJLensStore.getState().modelId).toBe('');
    expect(useJLensStore.getState().prompt).toBe('');
    expect(useJLensStore.getState().pinned).toEqual([]);
    // MUTATION CONTROL: drop the setPromptDraft('') and this fails — the old
    // text stays in the box, contradicting the cleared state and one click
    // from being re-submitted.
    expect(screen.queryByDisplayValue('The capital of France is')).toBeNull();
  });

  it('Clear is disabled when there is nothing to forget', () => {
    render(<JLensPanel />);
    expect(screen.getByRole('button', { name: /clear/i })).toBeDisabled();
  });
});

describe('the Diff view shows where the lenses part company', () => {
  function bothLenses(jacTop: string[][], logitTop: string[][], layers: number[]) {
    const mk = (type: LensType, top: string[][]) => ({
      type,
      top_tokens: top,
      top_probs: top.map((r) => r.map(() => 0.5)),
    });
    return {
      meta: {
        kind: 'meta' as const,
        model: 'm',
        types: ['JACOBIAN_LENS', 'LOGIT_LENS'] as LensType[],
        layers_by_type: { JACOBIAN_LENS: layers, LOGIT_LENS: layers },
        top_n: 2,
        prompt_len: 1,
      },
      tokens: [
        {
          kind: 'token' as const,
          position: 0,
          token: 'x',
          id: 1,
          is_generated: false,
          results: [mk('JACOBIAN_LENS', jacTop), mk('LOGIT_LENS', logitTop)],
        },
      ],
    };
  }

  it('names the first layer at which the two lenses disagree', () => {
    // MUTATION CONTROL: return the LAST disagreement, or the row index instead
    // of the absolute layer, and this fails.
    render(<JLensPanel />);
    act(() =>
      useJLensStore.setState({
        // TWO disagreements, so "first" and "last" are DIFFERENT layers. The
        // earlier fixture had exactly one, which meant a mutation returning
        // the last disagreement passed — verified: it survived.
        //   L10 agree · L11 differ · L12 differ · L13 differ
        ...bothLenses(
          [['a', 'b'], ['Paris', 'b'], ['Paris', 'b'], ['Paris', 'b']],
          [['a', 'b'], ['a', 'b'], ['a', 'b'], ['b', 'a']],
          [10, 11, 12, 13]
        ),
        modelId: 'm_lfm2',
        lensMode: 'DIFF',
        selPos: 0,
        selLayerIdx: 0,
        pinned: [],
        isLoading: false,
        error: null,
        bandReport: null,
      })
    );

    expect(screen.getByText(/lenses first diverge at L11/)).toBeInTheDocument();
  });

  it('says nothing when the lenses agree everywhere', () => {
    // "they never disagree" and "we could not tell" must not look the same.
    render(<JLensPanel />);
    act(() =>
      useJLensStore.setState({
        ...bothLenses([['a', 'b'], ['a', 'b']], [['a', 'b'], ['a', 'b']], [0, 1]),
        modelId: 'm_lfm2',
        lensMode: 'DIFF',
        selPos: 0,
        selLayerIdx: 0,
        pinned: [],
        isLoading: false,
        error: null,
        bandReport: null,
      })
    );

    expect(screen.queryByText(/lenses first diverge/)).toBeNull();
  });
});

describe('a restored readout says which prompt it describes', () => {
  it('warns when the grid no longer matches the prompt box', async () => {
    // The cost of keeping results across a refresh is that editing the prompt
    // afterwards leaves a grid that LOOKS current. Say so rather than clearing
    // it or letting it pass as fresh.
    //
    // MUTATION CONTROL: drop the readoutPrompt !== promptDraft check -> fails.
    // SEED THE STORE BEFORE MOUNTING, which is the order a refresh produces:
    // persist rehydrates, then the panel mounts and seeds its prompt draft from
    // the restored prompt. Setting state after render leaves the draft empty
    // and tests a state the app never reaches.
    const response = makeReadout([0, 1, 2]);
    act(() =>
      useJLensStore.setState({
        modelId: 'm_lfm2',
        prompt: 'The capital of France is',
        readoutPrompt: 'The capital of France is',
        meta: response.meta,
        tokens: response.tokens,
        restored: true,
        isLoading: false,
        error: null,
      })
    );
    render(<JLensPanel />);

    // Same prompt: it is a restoration, not a mismatch.
    expect(screen.getByText(/restored from your last session/i)).toBeInTheDocument();
    expect(screen.queryByText(/read out again to update/i)).toBeNull();

    // Now edit the prompt — the grid describes something else.
    await userEvent.type(
      screen.getByDisplayValue('The capital of France is'),
      ' not'
    );

    expect(screen.getByText(/read out again to update/i)).toBeInTheDocument();
    expect(screen.queryByText(/restored from your last session/i)).toBeNull();
  });
});

describe('the request block stays while the readout scrolls', () => {
  it('gives the panel its own scroll region below a fixed request block', () => {
    // The grid is tall. Scrolling the whole page took the model selector and
    // the prompt off the top, so changing either meant scrolling back up.
    //
    // MUTATION CONTROLS: drop `min-h-0` from the scroller, or the height
    // calc from the root, and this fails. `min-h-0` is load-bearing — a flex
    // child defaults to min-height:auto and grows instead of scrolling.
    const { container } = render(<JLensPanel />);
    const root = container.firstElementChild as HTMLElement;

    expect(root.className).toContain('flex');
    expect(root.className).toContain('flex-col');
    // Viewport minus the app header, which is h-14 and sticky.
    expect(root.className).toContain('h-[calc(100dvh-3.5rem)]');

    const scroller = container.querySelector('.overflow-y-auto');
    expect(scroller).not.toBeNull();
    expect(scroller!.className).toContain('min-h-0');
    expect(scroller!.className).toContain('flex-1');
  });

  it('keeps the model and prompt controls OUT of the scrolling region', () => {
    const { container } = render(<JLensPanel />);
    const scroller = container.querySelector('.overflow-y-auto')!;

    // If either lands inside the scroller it scrolls away, which is the whole
    // problem this layout exists to fix.
    expect(scroller.querySelector('select')).toBeNull();
    expect(
      scroller.querySelector('input[placeholder="The capital of France is"]')
    ).toBeNull();
  });
});

describe('a diffuse readout stays readable', () => {
  it('marks low confidence by HUE, not by fading toward the background', () => {
    // `dark:text-slate-700` was near-black on a slate cell and invisible on
    // the red DIFF shading — so the cells carrying the most interesting
    // signal (a token the logit lens does not rank at all) were the hardest
    // to read.
    //
    // MUTATION CONTROL: restore a slate/grey class for the diffuse case and
    // this fails.
    render(<JLensPanel />);
    // top-1 probabilities here are well under the diffuse threshold.
    const response = makeReadout([0, 1, 2]);
    for (const tk of response.tokens) {
      for (const slice of tk.results) {
        slice.top_probs = slice.top_probs.map((row) => row.map(() => 0.01));
      }
    }
    seed(response);

    const dimmed = Array.from(
      document.querySelectorAll('td[class*="text-pink"]')
    );
    expect(dimmed.length).toBeGreaterThan(0);
    for (const cell of dimmed) {
      expect(cell.className).not.toMatch(/text-slate-(600|700)\b/);
    }
  });
});

/**
 * The ranked lists and the Swap/Steer path, ASSERTED THROUGH THE PANEL.
 *
 * The components have their own tests, and those tests passed while nothing
 * rendered them: deleting the `<RankedReadouts />` element from the panel left
 * this file at 53/53 green. A component nobody mounts is the reachability gate
 * in the Code Quality Checklist, and it is the same shape as 16 MCP tools that
 * were implemented, tested and never registered.
 *
 * MUTATION CONTROLS:
 *   * delete <RankedReadouts /> from the panel   -> "renders the ranked lists" fails
 *   * send the empty prompt from the panel       -> "intervenes on the readout prompt" fails
 *   * queue a swap with no partner               -> "refuses a swap with no partner" fails
 */
describe('JLensPanel — ranked readouts and interventions', () => {
  it('renders the ranked lists for every lens the readout carries', async () => {
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() =>
      expect(screen.getByTestId('jlens-ranked')).toBeInTheDocument()
    );
    expect(screen.getByTestId('ranked-LOGIT_LENS')).toBeInTheDocument();
  });

  it('intervenes on the READOUT PROMPT and the token\u2019s OWN layers', async () => {
    /**
     * Not an empty string and not the whole axis: the readout on screen
     * describes one prompt, and a result scored on anything else measures a
     * different forward pass than the one being looked at.
     *
     * THE FIXTURE PUTS THE TOKEN AT A STRICT SUBSET OF THE AXIS. An earlier
     * version used a 3-layer axis and asserted `layers.length <= 3`, which the
     * whole axis satisfies — so handing the intervention every layer survived
     * the mutation. Here `dog` appears at L0 and L2 of a four-layer axis, so
     * only its own layers can produce [0, 2].
     *
     * MUTATION CONTROL: pass `axis` instead of the row's layers and this fails.
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-1',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    render(<JLensPanel />);
    seed({
      meta: {
        kind: 'meta',
        model: 'org/m',
        types: ['LOGIT_LENS'],
        layers_by_type: { LOGIT_LENS: [0, 1, 2, 3, 4, 5, 6, 7] },
        top_n: 1,
        prompt_len: 1,
      },
      tokens: [
        {
          kind: 'token',
          position: 0,
          token: 'x',
          id: 1,
          is_generated: false,
          results: [
            {
              type: 'LOGIT_LENS',
              // dog at L0 and L2 only; cat and pet fill the rest.
              //
              // EIGHT LAYERS, so the layer BUDGET (a quarter of the stack, so
              // two here) does not itself reduce the answer and confound what
              // this test measures. Handing over the whole axis would send the
              // deepest two of it, [6, 7] \u2014 still distinguishable from the
              // token's own [0, 2], which is the property being pinned.
              top_tokens: [
                ['dog'],
                ['cat'],
                ['dog'],
                ['pet'],
                ['cat'],
                ['pet'],
                ['cat'],
                ['pet'],
              ],
              top_probs: [
                [0.9],
                [0.9],
                [0.9],
                [0.9],
                [0.9],
                [0.9],
                [0.9],
                [0.9],
              ],
            },
          ],
        },
      ],
    } as unknown as ReadoutResponse);

    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());
    await user.click(screen.getByTitle('Steer along dog'));

    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.prompt).toBe('The capital of France is');
    expect(sent.primitive).toBe('additive');
    expect(sent.layers).toEqual([0, 2]);
  });

  it('REFUSES a swap that has no partner rather than queueing a doomed one', async () => {
    /**
     * With nothing pinned there is no second coordinate, and a swap with one
     * token is an additive steer wearing a swap\u2019s name. Queueing it would
     * report "Swap queued" here and be refused on the GPU seconds later.
     *
     * SCOPED TO THE RANKED COLUMN. An earlier version searched the whole
     * document for /Pin a token first/, which ALSO matches InterventionCard's
     * disabled toggle — so the assertion passed against a ranked list whose
     * Swap buttons were all enabled.
     *
     * MUTATION CONTROL: return no reason from `swapDisabledFor` and this fails.
     */
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());

    const column = screen.getByTestId('ranked-LOGIT_LENS');
    const swaps = within(column).getAllByRole('button', { name: /Swap/ });
    expect(swaps.length).toBeGreaterThan(0);
    for (const b of swaps) expect(b).toBeDisabled();
  });
  it('CAPS how many layers one click hooks, at a quarter of the stack', async () => {
    /**
     * A ranked-list click passes every layer the token appeared at. On
     * gemma-2-2b a common token is in the top-k at ALL 26, so one click hooked
     * the whole stack at strength 1 \u2014 guaranteed oversteering, and exactly
     * what BR-017 v0.2 warns about for small models. `default_swap_layers`
     * derived the budget as a quarter of the stack and had NO production
     * caller until this.
     *
     * The DEEPEST are kept: the shallow hits are mostly the junk bands the
     * non-word filter exists to declutter.
     *
     * MUTATION CONTROL: pass `layers` through unbudgeted and this fails with
     * all eight.
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-5',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    render(<JLensPanel />);
    // 'ubiquitous' at EVERY layer of an eight-layer stack \u2014 the shape that
    // caused the oversteer.
    seed({
      meta: {
        kind: 'meta',
        model: 'org/m',
        types: ['LOGIT_LENS'],
        layers_by_type: { LOGIT_LENS: [0, 1, 2, 3, 4, 5, 6, 7] },
        top_n: 1,
        prompt_len: 1,
      },
      tokens: [
        {
          kind: 'token',
          position: 0,
          token: 'x',
          id: 1,
          is_generated: false,
          results: [
            {
              type: 'LOGIT_LENS',
              top_tokens: Array.from({ length: 8 }, () => ['ubiquitous']),
              top_probs: Array.from({ length: 8 }, () => [0.9]),
            },
          ],
        },
      ],
    } as unknown as ReadoutResponse);

    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());
    await user.click(screen.getByTitle('Steer along ubiquitous'));
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.layers).toEqual([6, 7]);
  });

  /**
   * Set up a panel showing BOTH columns with a real artifact mounted.
   *
   * `listArtifacts` is mocked BEFORE render because the panel loads artifacts
   * on mount and again when the model changes; a `setState` is overwritten by
   * that load a tick later, and then `hasArtifact` is false and BOTH branches
   * return undefined for the same reason. Two drafts of this test died there.
   */
  async function panelWithArtifact() {
    (jlensApi.listArtifacts as ReturnType<typeof vi.fn>).mockResolvedValue([
      { slug: 'm', directory: '/d', lens_file: 'm.pt', size_bytes: 1 },
    ]);
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-6',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['JACOBIAN_LENS', 'LOGIT_LENS'], 4));
    act(() => {
      useJLensStore.setState({ modelRepoId: 'org/m' });
    });
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());
    await waitFor(() =>
      expect(useJLensStore.getState().artifacts).toHaveLength(1)
    );
  }

  async function steerFrom(testId: string) {
    const user = userEvent.setup();
    const column = screen.getByTestId(testId);
    await user.click(within(column).getAllByRole('button', { name: /Steer/ })[0]);
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    return (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
  }

  it('CREDITS the artifact when the click came from the Jacobian column', async () => {
    /**
     * The precondition for the test below. Without it that one is vacuous the
     * moment anything stops sending an artifact_id at all.
     */
    await panelWithArtifact();
    expect((await steerFrom('ranked-JACOBIAN_LENS')).artifact_id).toBe('m');
  });

  it('does NOT credit it when the click came from the logit column', async () => {
    /**
     * `artifact_id` is provenance, not measurement — the perturbation happens
     * in activation space either way. Sending it for a LOGIT-lens token, which
     * needs no artifact at all, filed an `evidence_rung: 2` record under
     * `lens_type: JACOBIAN_LENS` into the artifact's `interventions.json`: the
     * file built to travel to HuggingFace and into a serving runtime,
     * describing a finding the Jacobian played no part in.
     *
     * SEPARATE TEST, ONE CLICK EACH. Both clicks in one body cannot work: the
     * second is refused by the in-flight guard until the first run polls to a
     * terminal state.
     *
     * MUTATION CONTROL: send `artifact_id` unconditionally and this fails.
     */
    await panelWithArtifact();
    expect((await steerFrom('ranked-LOGIT_LENS')).artifact_id).toBeUndefined();
  });

  it('counts the STACK ONCE when two lenses share it', async () => {
    /**
     * The budget fallback used `Object.values(layers_by_type).flat().length`,
     * which CONCATENATES every lens type's axis — so a 4-layer model carrying
     * both a Jacobian and a logit lens counted as 8 and the budget doubled,
     * hooking twice as much of the stack as BR-017's derivation allows.
     *
     * The fallback is reachable: `fullSpan` is only learned from an unnarrowed
     * read, and a reload from storage written before it was persisted has a
     * `layerRange` and no `fullSpan`, with no migration between them.
     *
     * MUTATION CONTROL: use `.flat().length` instead of `new Set(...).size` and
     * this fails with four layers instead of two.
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-7',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    render(<JLensPanel />);
    seed({
      meta: {
        kind: 'meta',
        model: 'org/m',
        types: ['JACOBIAN_LENS', 'LOGIT_LENS'],
        // THE SAME EIGHT LAYERS UNDER BOTH TYPES. Concatenating gives 16 and a
        // budget of 4; the distinct count gives 8 and a budget of 2.
        layers_by_type: {
          JACOBIAN_LENS: [0, 1, 2, 3, 4, 5, 6, 7],
          LOGIT_LENS: [0, 1, 2, 3, 4, 5, 6, 7],
        },
        top_n: 1,
        prompt_len: 1,
      },
      tokens: [
        {
          kind: 'token',
          position: 0,
          token: 'x',
          id: 1,
          is_generated: false,
          results: ['JACOBIAN_LENS', 'LOGIT_LENS'].map((type) => ({
            type,
            top_tokens: Array.from({ length: 8 }, () => ['ubiquitous']),
            top_probs: Array.from({ length: 8 }, () => [0.9]),
          })),
        },
      ],
    } as unknown as ReadoutResponse);
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());

    // NO fullSpan, so the fallback is the path under test. `seed` does not set
    // it and no unnarrowed fetch has run.
    expect(useJLensStore.getState().fullSpan).toBeNull();

    const col = screen.getByTestId('ranked-JACOBIAN_LENS');
    await user.click(within(col).getAllByRole('button', { name: /Steer/ })[0]);
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    expect(
      (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0].layers,
    ).toEqual([6, 7]);
  });

  it('an acquisition does NOT overwrite a completed intervention verdict', async () => {
    /**
     * `interventionNote` is owned by `runIntervention`, which blanks it on
     * every click — and it holds the ONLY copy of a completed rung-2 verdict.
     * Echoing an acquire acknowledgement into it destroyed the product's
     * headline result ("intervened 6/6 vs control 0/6 … the intervals are
     * disjoint"), and clicking Steer afterwards erased the acquire's only
     * acknowledgement in return. The card shows its own note.
     *
     * MUTATION CONTROL: pass an `onQueued` that writes `interventionNote` and
     * this fails.
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-9',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    (getTaskStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-9',
      state: 'SUCCESS',
      result: {
        n_trials: 6,
        separation_attainable: true,
        separated_from_control: true,
        intervened_top1: { hits: 6, n: 6, rate: 1, ci95_low: 0.61, ci95_high: 1 },
        control_top1: { hits: 0, n: 6, rate: 0, ci95_low: 0, ci95_high: 0.39 },
        baseline_top1: { hits: 0, n: 6, rate: 0, ci95_low: 0, ci95_high: 0.39 },
      },
    });
    (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue({
      repo_id: 'org/lenses',
      revision: 'sha123456',
      candidates: [
        {
          path: 'a_jacobian_lens.pt',
          size_bytes: 100,
          has_config: true,
          has_convergence: false,
          fits_envelope: true,
          envelope_detail: null,
        },
      ],
    });
    (jlensApi.acquire as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'acq-9',
    });

    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());

    const column = screen.getByTestId('ranked-LOGIT_LENS');
    await user.click(within(column).getAllByRole('button', { name: /Steer/ })[0]);
    await waitFor(() => expect(screen.getByText(/are disjoint/)).toBeInTheDocument(), {
      timeout: 8000,
    });

    // Now queue an acquisition from the card in the same panel.
    await user.click(screen.getByRole('button', { name: /Browse/ }));
    await user.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/1 candidate/));
    await user.click(screen.getAllByRole('radio')[0]);
    await user.click(screen.getByTestId('jlens-acquire-run'));
    await waitFor(() =>
      expect(jlensApi.acquire as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );

    // THE VERDICT SURVIVES. It is the only place that result exists in the UI.
    expect(screen.getByText(/are disjoint/)).toBeInTheDocument();
  }, 20000);

  it('tells the card the weights ARE PRESENT for a healthy model', async () => {
    /**
     * The positive control. The default panel fixture carries no `file_path`,
     * so `weightsPresent` was already false everywhere and the prop could be
     * stuck at `false` — a permanent, untrue "is not downloaded … this will be
     * refused" banner on a fully downloaded model, with a green suite.
     *
     * MUTATION CONTROL: `return false;` in the predicate and this fails.
     */
    (useModelsStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      (sel: (s: unknown) => unknown) =>
        sel({
          models: [
            { id: 'm_lfm2', status: 'ready', file_path: '/data/models/raw/m_lfm2' },
          ],
          fetchModels: vi.fn(),
        }),
    );
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-acquire')).toBeTruthy());
    await userEvent.setup().click(screen.getByRole('button', { name: /Browse/ }));
    expect(screen.queryByTestId('jlens-acquire-weights-missing')).toBeNull();
  });

  it('tells the card the WEIGHTS ARE MISSING when the row has no file_path', async () => {
    /**
     * The endpoint refuses via `locate_weights`: a `file_path` that is set AND
     * present on disk. Inferring from `status === "ready"` alone reports a model
     * whose files were pruned as available, so the card implies a fetch the
     * endpoint then 409s — and both prerequisite props were computed here and
     * asserted by nothing, so hardcoding either one left the suite green.
     *
     * MUTATION CONTROL: hardcode `weightsPresent={true}` and this fails.
     */
    (useModelsStore as unknown as ReturnType<typeof vi.fn>).mockImplementation(
      (sel: (s: unknown) => unknown) =>
        sel({
          models: [{ id: 'm_lfm2', status: 'ready', file_path: null }],
          fetchModels: vi.fn(),
        }),
    );
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-acquire')).toBeTruthy());
    await userEvent.setup().click(screen.getByRole('button', { name: /Browse/ }));
    expect(
      screen.getByTestId('jlens-acquire-weights-missing')
    ).toBeInTheDocument();
  });

  it('MOUNTS the acquire card \u2014 the only way to reach a published lens', async () => {
    /**
     * REACHABILITY, not existence. `AcquireLensCard.test.tsx` renders the
     * component directly, so deleting it from this panel would leave that file
     * green while no user could download or publish anything — the shape of the
     * 16 MCP tools that shipped fully tested and unregistered.
     *
     * MUTATION CONTROL: remove `<AcquireLensCard />` from the panel and this
     * fails.
     */
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() =>
      expect(screen.getByTestId('jlens-acquire')).toBeInTheDocument()
    );
  });

  it('MOUNTS the layer-range picker \u2014 the only way a range is ever set', async () => {
    /**
     * REACHABILITY, not existence. `LayerRangePicker.test.tsx` renders the
     * component itself and `jlensStore.range.test.ts` calls `setState`
     * directly, so removing the element from this panel left BOTH green while
     * no user could set a range at all. That is the shape of the 16 MCP tools
     * that were fully implemented, fully tested and never registered.
     *
     * MUTATION CONTROL: change the panel\u2019s `all.length ? (` guard to
     * `false && all.length ? (` and this fails.
     */
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() =>
      expect(screen.getByTestId('jlens-layer-range')).toBeInTheDocument()
    );
    // AND ITS BOUNDS COME FROM THE AXIS, so a picker that mounted with the
    // wrong span would not pass either.
    const picker = screen.getByTestId('jlens-layer-range');
    expect(within(picker).getByText(/of 0\u20132/)).toBeInTheDocument();
  });

  it('sends the SWAP PARTNER, and names it on the button', async () => {
    /**
     * The partner is half the experiment: it supplies the second coordinate
     * AND it is the token whose rank gets scored. Nothing exercised the
     * enabled swap path, so stripping `target_token` from the payload left the
     * suite green while every swap would be 422\u2019d by the backend
     * validator added in this same arc.
     *
     * MUTATION CONTROLS:
     *   * `target_token: undefined` on the swap path -> "sends" fails
     *   * drop `swapPartnerFor` from RankedReadouts   -> "names it" fails
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-2',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());

    // Pin a token so a partner exists. The pin chips carry the display form.
    act(() => {
      useJLensStore.setState({ pinned: ['France'] });
    });

    const column = screen.getByTestId('ranked-LOGIT_LENS');
    const swap = within(column)
      .getAllByRole('button', { name: /Swap/ })
      .find((b) => !(b as HTMLButtonElement).disabled);
    expect(swap, 'no enabled Swap button once a token was pinned').toBeTruthy();
    // NAMED ON THE CONTROL, not merely sent. Re-pinning in a different order
    // runs a different experiment under an identical-looking click.
    expect(swap!.textContent).toMatch(/France/);

    await user.click(swap!);
    await waitFor(() =>
      expect(jlensApi.intervene as ReturnType<typeof vi.fn>).toHaveBeenCalled()
    );
    const sent = (jlensApi.intervene as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.primitive).toBe('coordinate_swap');
    expect(sent.target_token).toBe('France');
    expect(sent.direction_token).not.toBe('France');
    // THE REST OF THE PAYLOAD TOO. "was called" passes against a call sending
    // the wrong arguments.
    expect(sent.k).toBe(4);
    expect(sent.strength).toBe(1);
  });

  it('reports OVERLAP as overlap and DISJOINT as disjoint', async () => {
    /**
     * `pollIntervention` had no coverage at all: `getTaskStatus` was mocked
     * only to stop the poller hanging, and nothing asserted the note. The
     * verdict could be inverted \u2014 telling the reader "the intervals are
     * disjoint" for a run whose intervals overlap \u2014 with the suite green.
     * That is a false causal claim, which is the one thing this string exists
     * to prevent.
     *
     * MUTATION CONTROL: negate the `sep` branch and this fails.
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-3',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    // FOUR TRIALS, so separation is arithmetically attainable and the branch
    // under test is reachable. At n<4 no outcome separates and the panel takes
    // the not-attainable branch instead \u2014 which is what both UI paths
    // actually did, and why this needed saying.
    (getTaskStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-3',
      state: 'SUCCESS',
      result: {
        n_trials: 4,
        separation_attainable: true,
        separated_from_control: false,
        intervened_top1: { hits: 2, n: 4, rate: 0.5, ci95_low: 0.15, ci95_high: 0.85 },
        control_top1: { hits: 1, n: 4, rate: 0.25, ci95_low: 0.05, ci95_high: 0.7 },
        baseline_top1: { hits: 0, n: 4, rate: 0, ci95_low: 0, ci95_high: 0.49 },
      },
    });

    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());
    const column = screen.getByTestId('ranked-LOGIT_LENS');
    await user.click(within(column).getAllByRole('button', { name: /Steer/ })[0]);

    await waitFor(
      () => expect(screen.getByText(/OVERLAP/)).toBeInTheDocument(),
      { timeout: 8000 }
    );
    expect(screen.queryByText(/are disjoint/)).toBeNull();
  }, 15000);

  it('says a SINGLE TRIAL cannot separate, rather than reporting no effect', async () => {
    /**
     * Below four trials a PERFECT intervened arm against a PERFECT null
     * control still overlaps \u2014 verified numerically in
     * `MIN_TRIALS_FOR_SEPARATION`. Both UI paths send one prompt, so "no
     * effect was demonstrated" was the only verdict either could ever produce,
     * and it reported a fact about the sample size as a finding about the
     * direction.
     *
     * MUTATION CONTROL: drop the `separation_attainable === false` branch and
     * this fails, falling through to the OVERLAP wording.
     */
    const user = userEvent.setup();
    (jlensApi.intervene as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-4',
      model_id: 'm_lfm2',
      queue: 'extraction',
    });
    (getTaskStatus as ReturnType<typeof vi.fn>).mockResolvedValue({
      task_id: 'iv-4',
      state: 'SUCCESS',
      result: {
        n_trials: 1,
        separation_attainable: false,
        min_trials_for_separation: 4,
        separated_from_control: false,
        intervened_top1: { hits: 1, n: 1, rate: 1, ci95_low: 0.2065, ci95_high: 1 },
        control_top1: { hits: 0, n: 1, rate: 0, ci95_low: 0, ci95_high: 0.7935 },
        baseline_top1: { hits: 0, n: 1, rate: 0, ci95_low: 0, ci95_high: 0.7935 },
      },
    });

    render(<JLensPanel />);
    seed(makeReadout([0, 1, 2], ['LOGIT_LENS'], 4));
    await waitFor(() => expect(screen.getByTestId('jlens-ranked')).toBeTruthy());
    const column = screen.getByTestId('ranked-LOGIT_LENS');
    await user.click(within(column).getAllByRole('button', { name: /Steer/ })[0]);

    await waitFor(
      () => expect(screen.getByText(/not attainable/)).toBeInTheDocument(),
      { timeout: 8000 }
    );
    // AND IT MUST NOT ALSO SAY "no effect was demonstrated" \u2014 the whole
    // point is that the two readings are opposite.
    expect(screen.queryByText(/no effect was demonstrated/)).toBeNull();
  }, 15000);
});

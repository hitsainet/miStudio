/**
 * Downloading a published lens, and publishing ours.
 *
 * The two prerequisites this card exists to make visible BEFORE a fetch:
 * a lens is unusable without its model's weights (validating one means reading
 * out through it), and a file with no `config.yaml` beside it cannot have its
 * weight identity checked — the pairing then rests on the operator's assertion,
 * and the artifact records that it does.
 *
 * MUTATION CONTROLS (each must turn this file red):
 *   * send the branch instead of the resolved sha -> "pins the RESOLVED commit"
 *   * drop the weights-missing warning            -> "warns BEFORE the fetch"
 *   * render every candidate as identity-checkable -> "marks a config-less file"
 *   * enable publish with no artifact             -> "publish is refused"
 *   * drop the what-does-not-travel note          -> "says the local verdict"
 *   * preview on the acquire button               -> "looks BEFORE it fetches"
 */

import { describe, expect, it, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor, within } from '@testing-library/react';

vi.mock('../../api/jlens', () => ({
  jlensApi: { previewRepo: vi.fn(), acquire: vi.fn(), publish: vi.fn() },
}));

import { jlensApi } from '../../api/jlens';
import { AcquireLensCard } from './AcquireLensCard';

const PREVIEW = {
  repo_id: 'org/lenses',
  // NOT `main`. The card must send this back, or the acquisition names a
  // moving target.
  revision: 'abc123def4567890',
  candidates: [
    {
      path: 'gemma/jlens/wikitext/gemma_jacobian_lens.pt',
      size_bytes: 265_429_252,
      has_config: true,
      has_convergence: true,
      fits_envelope: true,
      envelope_detail: 'within a full fit',
    },
    {
      path: 'loose_lens.pt',
      size_bytes: 1024,
      has_config: false,
      has_convergence: false,
      fits_envelope: true,
      envelope_detail: null,
    },
  ],
};

function mount(over: Partial<React.ComponentProps<typeof AcquireLensCard>> = {}) {
  return render(
    <AcquireLensCard
      modelId="m_1"
      modelRepoId="google/gemma-2-2b-it"
      weightsPresent
      hasArtifact
      {...over}
    />,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue(PREVIEW);
  (jlensApi.acquire as ReturnType<typeof vi.fn>).mockResolvedValue({
    task_id: 'acq-12345678',
  });
  (jlensApi.publish as ReturnType<typeof vi.fn>).mockResolvedValue({
    task_id: 'pub-12345678',
  });
});

const open = () => fireEvent.click(screen.getByRole('button', { name: /Browse/ }));

describe('acquiring', () => {
  it('looks BEFORE it fetches', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() =>
      expect(jlensApi.previewRepo as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    // AND NOTHING WAS DOWNLOADED. A mistyped path must cost a request, not a
    // multi-gigabyte fetch and a slot on the single-GPU queue.
    expect(jlensApi.acquire as ReturnType<typeof vi.fn>).not.toHaveBeenCalled();
  });

  it('pins the RESOLVED commit, not the branch', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));

    await waitFor(() =>
      expect(jlensApi.acquire as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    const sent = (jlensApi.acquire as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.revision).toBe('abc123def4567890');
    expect(sent.path_in_repo).toBe('gemma/jlens/wikitext/gemma_jacobian_lens.pt');
    expect(sent.repo_id).toBe('org/lenses');
  });

  it('warns BEFORE the fetch when the weights are missing', () => {
    mount({ weightsPresent: false });
    open();
    const warning = screen.getByTestId('jlens-acquire-weights-missing');
    expect(warning).toHaveTextContent(/not downloaded/);
    expect(warning).toHaveTextContent(/google\/gemma-2-2b-it/);
  });

  it('does NOT warn when the weights are present', () => {
    mount();
    open();
    expect(screen.queryByTestId('jlens-acquire-weights-missing')).toBeNull();
  });

  it('marks a config-less file as identity-UNVERIFIED', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));

    const rows = screen.getAllByRole('listitem');
    // The one WITH a config reads differently from the one without — a single
    // shared label would tell the operator nothing.
    // THE BADGE STATES WHAT WAS OBSERVED, not a verdict it cannot predict:
    // `has_config` is file presence, while the outcome depends on whether that
    // config NAMES a model — and one naming another model is a hard refusal.
    expect(within(rows[0]).getByText('declares a config')).toBeInTheDocument();
    expect(within(rows[1]).getByText('no config')).toBeInTheDocument();
  });

  it('explains what UNVERIFIED costs, once one is chosen', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[1]);
    expect(screen.getByText(/rests on your assertion/)).toBeInTheDocument();
  });

  it('cannot acquire until a file is chosen', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();
    fireEvent.click(screen.getAllByRole('radio')[0]);
    expect(screen.getByTestId('jlens-acquire-run')).toBeEnabled();
  });
});

describe('publishing', () => {
  const toPublish = () =>
    // SCOPED TO THE MODE TOGGLE. The publish ACTION button shares the
    // accessible name, and only the toggle carries aria-pressed.
    fireEvent.click(
      screen
        .getAllByRole('button')
        .filter((b) => b.hasAttribute('aria-pressed'))[1],
    );

  it('is REFUSED when there is no published artifact', () => {
    mount({ hasArtifact: false });
    open();
    toPublish();
    // A TARGET REPO IS SUPPLIED FIRST. Without it the button is disabled for a
    // DIFFERENT reason, and the assertion below passes whether or not the
    // artifact check exists — a fixture agreeing by construction, which is
    // exactly how the mutation survived the first version of this test.
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    expect(screen.getByTestId('jlens-publish-no-artifact')).toHaveTextContent(
      /staged artifact is not published/,
    );
    expect(screen.getByTestId('jlens-publish-run')).toBeDisabled();
  });

  it('IS enabled once both the repo and an artifact are present', () => {
    // The positive control. Without it, "disabled" could be permanent.
    mount({ hasArtifact: true });
    open();
    toPublish();
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    expect(screen.getByTestId('jlens-publish-run')).toBeEnabled();
  });

  it('describes ACCURATELY what the upload carries', () => {
    mount();
    open();
    toPublish();
    // THE NOTE MUST MATCH WHAT IS ACTUALLY UPLOADED. The earlier version of
    // this test asserted "the local validation verdict does not travel" — which
    // was FALSE: `model_card` writes every check's name, status and detail into
    // the README, and the README is uploaded. Only `validation.json` is
    // withheld. A test pinning a false claim is worse than no test, because
    // correcting the copy turned the suite red.
    const note = screen.getByTestId('jlens-publish-note');
    expect(note).toHaveTextContent(/every check and its status/i);
    expect(note).toHaveTextContent(/deferred/i);
    expect(note).toHaveTextContent(/never been run/);
    // PINNED ON THE VERB, not on the vocabulary. Round 2 replaced a test that
    // asserted a FALSE sentence with one asserting four substrings none of
    // which touched the claim — so "validation.json is UPLOADED too" passed it.
    // The one factual assertion in the sentence is that the file is withheld.
    expect(note).toHaveTextContent(/validation\.json[^.]*withheld/i);
    expect(note).not.toHaveTextContent(/validation\.json[^.]*uploaded/i);
    expect(note).not.toHaveTextContent(/verdict does not travel/i);
  });

  it('sends the corpus segment it was given', async () => {
    mount();
    open();
    toPublish();
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    fireEvent.change(screen.getByTestId('jlens-publish-dataset'), {
      target: { value: 'wikitext' },
    });
    fireEvent.click(screen.getByTestId('jlens-publish-run'));
    await waitFor(() =>
      expect(jlensApi.publish as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    const sent = (jlensApi.publish as ReturnType<typeof vi.fn>).mock.calls[0][0];
    expect(sent.target_repo).toBe('you/lenses');
    expect(sent.dataset).toBe('wikitext');
    expect(sent.model_id).toBe('m_1');
  });

  it('cannot publish without a target repo', () => {
    mount();
    open();
    toPublish();
    expect(screen.getByTestId('jlens-publish-run')).toBeDisabled();
  });
});

describe('the token field', () => {
  it('is masked by default and can be revealed', () => {
    mount();
    open();
    const field = screen.getByTestId('jlens-acquire-token');
    expect(field).toHaveAttribute('type', 'password');
    fireEvent.click(screen.getByRole('button', { name: /Show token/ }));
    expect(screen.getByTestId('jlens-acquire-token')).toHaveAttribute(
      'type',
      'text',
    );
  });

  it('is not sent when left empty, so the configured one is used', async () => {
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() =>
      expect(jlensApi.previewRepo as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    const sent = (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mock
      .calls[0][0];
    expect(sent.access_token).toBeUndefined();
  });
});

describe('review round 1 findings', () => {
  it('DISCARDS a preview when the model changes', async () => {
    /**
     * Every `fits_envelope` verdict in the list was computed server-side for
     * ONE model's dimensions. A list left on screen after the model changes
     * shows badges computed for other weights — and the selection would send a
     * lens for one model against another, which the endpoint cannot catch and
     * the worker only discovers after downloading the whole file.
     *
     * MUTATION CONTROL: drop the `useEffect` on `modelId` and this fails.
     */
    const { rerender } = mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);

    rerender(
      <AcquireLensCard
        modelId="m_OTHER"
        modelRepoId="org/other"
        weightsPresent
        hasArtifact
      />,
    );
    // The card stays OPEN across the rerender — only the preview is discarded.
    expect(screen.queryByText(/2 candidates/)).toBeNull();
    expect(screen.queryAllByRole('radio')).toHaveLength(0);
  });

  it('REFUSES to acquire with no model chosen', () => {
    /**
     * The store initialises `modelId: ''`, so a fresh session renders the
     * prerequisite warning naming no model at all, and the button was enabled —
     * POSTing `model_id: ""` for a 404 the user reads as a mystery.
     *
     * MUTATION CONTROL: drop `!modelId` from the disabled expression -> fails.
     */
    mount({ modelId: '' });
    open();
    expect(screen.getByTestId('jlens-acquire-no-model')).toBeInTheDocument();
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();
  });

  it('the no-model guard holds even once a file IS selected', async () => {
    /**
     * The test above never previews, so `!selected` disables the button on its
     * own and `!modelId` can be deleted freely — its own docstring claimed a
     * control that did not work. Selecting a candidate first is what isolates
     * the clause.
     *
     * MUTATION CONTROL: delete `!modelId ||` from the disabled expression.
     */
    mount({ modelId: '' });
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();
  });

  it('does NOT render a warning naming no model', () => {
    /**
     * On a fresh session `modelId` is '' and this rendered "**  ** is not
     * downloaded" — round 1 fixed the button and left the misleading string
     * beside it. The helper's `weightsPresent` default is `true`, so the
     * earlier fixture could not see it either.
     *
     * MUTATION CONTROL: drop `Boolean(modelId) &&` from the warning.
     */
    mount({ modelId: '', weightsPresent: false });
    open();
    expect(screen.queryByTestId('jlens-acquire-weights-missing')).toBeNull();
    expect(screen.getByTestId('jlens-acquire-no-model')).toBeInTheDocument();
  });

  it('does not let a queued job be queued AGAIN', async () => {
    /**
     * `busy` releases at the 202, not at the job's terminal state, and nothing
     * else about the form changed — so a second click re-downloaded the same
     * multi-gigabyte file, which the worker then refused on the staging guard
     * after paying the bandwidth twice.
     *
     * MUTATION CONTROL: drop `Boolean(queued)` from the disabled expression.
     */
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));
    await waitFor(() =>
      expect(jlensApi.acquire as ReturnType<typeof vi.fn>).toHaveBeenCalledTimes(1),
    );
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();

    // AND STARTING ANOTHER IS A DECISION, not a double-click.
    fireEvent.click(screen.getByRole('button', { name: /start another/ }));
    expect(screen.getByTestId('jlens-acquire-run')).toBeEnabled();
  });

  it('keeps the READ and WRITE tokens apart', async () => {
    /**
     * One shared field silently reused a read-scope token as the publish
     * credential — masked, so the only signal was a label. The endpoint's
     * pre-flight only tests that A token exists, so it 202s and fails inside
     * the worker after taking a slot on the single-GPU queue.
     *
     * MUTATION CONTROL: share one `token` state and this fails.
     */
    mount();
    open();
    fireEvent.change(screen.getByTestId('jlens-acquire-token'), {
      target: { value: 'hf_READ_ONLY' },
    });
    fireEvent.click(
      screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed'))[1],
    );
    expect(screen.getByTestId('jlens-acquire-token')).toHaveValue('');
  });

  it('mirrors the server constraint on the corpus segment', () => {
    /**
     * It is a path segment, and the obvious value to type is the corpus's own
     * id — `wikitext/wikitext-103` — whose slash 422s against a regex the form
     * gave no hint about.
     *
     * MUTATION CONTROL: drop DATASET_PATTERN from the disabled expression.
     */
    mount();
    open();
    fireEvent.click(
      screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed'))[1],
    );
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    fireEvent.change(screen.getByTestId('jlens-publish-dataset'), {
      target: { value: 'wikitext/wikitext-103' },
    });
    expect(screen.getByTestId('jlens-publish-dataset-invalid')).toBeInTheDocument();
    expect(screen.getByTestId('jlens-publish-run')).toBeDisabled();

    fireEvent.change(screen.getByTestId('jlens-publish-dataset'), {
      target: { value: 'wikitext-103' },
    });
    expect(screen.queryByTestId('jlens-publish-dataset-invalid')).toBeNull();
    expect(screen.getByTestId('jlens-publish-run')).toBeEnabled();
  });

  it('names the OTHER publish gate it cannot check', () => {
    /**
     * `hasArtifact` is slug presence only. The endpoint also refuses an
     * artifact whose stored verdict no longer matches its current weights, and
     * the listing deliberately carries no validity field — so the card cannot
     * check it and must not imply a present artifact is sufficient.
     *
     * MUTATION CONTROL: delete the note and this fails.
     */
    mount({ hasArtifact: true });
    open();
    fireEvent.click(
      screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed'))[1],
    );
    expect(screen.getByText(/validation verdict matching/)).toBeInTheDocument();
  });

  it('clears a stale SUCCESS note when the next request FAILS', async () => {
    /**
     * A red error rendered directly beneath a still-green "queued" line, so a
     * refused request read on screen as a queued one.
     *
     * THE FIRST VERSION OF THIS TEST WAS VACUOUS: it only clicked "Look
     * inside", so nothing ever set a note and the assertion held with or
     * without the fix. A real one has to queue a job and then fail the next
     * request.
     *
     * MUTATION CONTROL: drop `setNote(null)` from the request starts.
     */
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));
    await waitFor(() => expect(screen.getByText(/queued as/)).toBeInTheDocument());

    // Now start another and have it refused.
    fireEvent.click(screen.getByRole('button', { name: /start another/ }));
    (jlensApi.acquire as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('507 Insufficient Storage'),
    );
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));
    await waitFor(() =>
      expect(screen.getByText(/507 Insufficient Storage/)).toBeInTheDocument(),
    );
    expect(screen.queryByText(/queued as/)).toBeNull();
  });
});

describe('review round 2 findings', () => {
  it('sends the WRITE token on publish, not the read one', async () => {
    /**
     * Round 1 separated the fields; nothing asserted the wire. Swapping
     * `writeToken` for `readToken` in the request left the suite green, and the
     * defect returns verbatim — the endpoint's pre-flight only tests that A
     * token resolves, so it 202s and dies in the worker after taking a slot.
     *
     * MUTATION CONTROL: send `readToken` from `runPublish` and this fails.
     */
    mount();
    open();
    fireEvent.change(screen.getByTestId('jlens-acquire-token'), {
      target: { value: 'hf_READ' },
    });
    fireEvent.click(
      screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed'))[1],
    );
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    fireEvent.change(screen.getByTestId('jlens-acquire-token'), {
      target: { value: 'hf_WRITE' },
    });
    fireEvent.click(screen.getByTestId('jlens-publish-run'));
    await waitFor(() =>
      expect(jlensApi.publish as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    expect(
      (jlensApi.publish as ReturnType<typeof vi.fn>).mock.calls[0][0].access_token,
    ).toBe('hf_WRITE');
  });

  it('sends the READ token on acquire', async () => {
    mount();
    open();
    fireEvent.change(screen.getByTestId('jlens-acquire-token'), {
      target: { value: 'hf_READ' },
    });
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));
    await waitFor(() =>
      expect(jlensApi.acquire as ReturnType<typeof vi.fn>).toHaveBeenCalled(),
    );
    expect(
      (jlensApi.acquire as ReturnType<typeof vi.fn>).mock.calls[0][0].access_token,
    ).toBe('hf_READ');
  });

  it('REFUSES a candidate the preview already ruled out', async () => {
    /**
     * `fits_envelope: false` rendered a red badge and changed nothing else, so
     * the one verdict the preview exists to produce did not prevent the
     * multi-gigabyte fetch and queue slot it exists to prevent. Both fixtures
     * used `fits_envelope: true`, so nothing exercised it.
     *
     * MUTATION CONTROL: drop the `fits_envelope === false` clause -> fails.
     */
    (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...PREVIEW,
      candidates: [
        {
          ...PREVIEW.candidates[0],
          fits_envelope: false,
          envelope_detail: '9,000,000,000 bytes exceeds 400,000,000',
        },
      ],
    });
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/1 candidate/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    expect(screen.getByTestId('jlens-acquire-too-large')).toHaveTextContent(
      /exceeds 400,000,000/,
    );
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();
  });

  it('survives a candidate with an UNKNOWN size', async () => {
    /**
     * The Hub does not always report one. `formatBytes`'s null branch had no
     * fixture, so it could be deleted freely.
     */
    (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...PREVIEW,
      candidates: [{ ...PREVIEW.candidates[0], size_bytes: null }],
    });
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/1 candidate/));
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('SHOWS a rejection rather than going quiet', async () => {
    /**
     * The most likely thing a user hits had zero coverage: dropping
     * `setError(...)` from a catch left the suite green while a refusal
     * rendered as nothing at all — the spinner stops and the card is silent.
     * The 409 detail is the only explanation that exists.
     *
     * MUTATION CONTROL: drop `setError` from `runPreview` and this fails.
     */
    (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('Could not read org/nope: 404'),
    );
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/404/),
    );
  });

  it('says so when a repo holds NO candidates', async () => {
    (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...PREVIEW,
      candidates: [],
    });
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/No .* files/),
    );
  });

  it('holds the PUBLISH button after a publish, and offers a reset', async () => {
    /**
     * The queued note lived inside the acquire branch, so after a publish the
     * button greyed out with no stated reason and the only recovery was to
     * switch modes.
     *
     * MUTATION CONTROL: drop `Boolean(queued)` from the publish disabled expr.
     */
    mount();
    open();
    fireEvent.click(
      screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed'))[1],
    );
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    fireEvent.click(screen.getByTestId('jlens-publish-run'));
    await waitFor(() =>
      expect(jlensApi.publish as ReturnType<typeof vi.fn>).toHaveBeenCalledTimes(1),
    );
    expect(screen.getByTestId('jlens-publish-run')).toBeDisabled();
    expect(screen.getByTestId('jlens-queued-note')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: /start another/ }));
    expect(screen.getByTestId('jlens-publish-run')).toBeEnabled();
  });

  it('explains BOTH identity outcomes, not only the config-less one', async () => {
    /**
     * The server sorts config-bearing files first, so the candidate most likely
     * to be picked was the one whose real outcome was least described — and a
     * config naming OTHER weights is a hard refusal after the bytes are spent.
     */
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    expect(screen.getByTestId('jlens-identity-note')).toHaveTextContent(/refused/);
    fireEvent.click(screen.getAllByRole('radio')[1]);
    expect(screen.getByTestId('jlens-identity-note')).toHaveTextContent(
      /rests on your assertion/,
    );
  });

  it('announces the active mode to assistive tech', () => {
    mount();
    open();
    const toggles = screen
      .getAllByRole('button')
      .filter((b) => b.hasAttribute('aria-pressed'));
    expect(toggles.map((b) => b.textContent)).toEqual(['Download', 'Publish']);
    expect(toggles[0]).toHaveAttribute('aria-pressed', 'true');
    expect(toggles[1]).toHaveAttribute('aria-pressed', 'false');
    fireEvent.click(toggles[1]);
    const after = screen
      .getAllByRole('button')
      .filter((b) => b.hasAttribute('aria-pressed'));
    expect(after[1]).toHaveAttribute('aria-pressed', 'true');
  });
});

describe('review round 3 — the fixes themselves', () => {
  it('KEEPS each token across a mode round-trip', async () => {
    /**
     * Round 1 pinned the split and not the retention: nothing asserted a token
     * survives switching away and back, so wiping both on every switch was
     * green — and a user who pastes a write token, flips to Download to
     * re-check the list, and flips back loses it silently.
     *
     * MUTATION CONTROL: clear both tokens in the mode toggle's onClick.
     */
    mount();
    open();
    fireEvent.change(screen.getByTestId('jlens-acquire-token'), {
      target: { value: 'hf_READ' },
    });
    const toggles = () =>
      screen.getAllByRole('button').filter((b) => b.hasAttribute('aria-pressed'));
    fireEvent.click(toggles()[1]);
    fireEvent.change(screen.getByTestId('jlens-acquire-token'), {
      target: { value: 'hf_WRITE' },
    });
    fireEvent.click(toggles()[0]);
    expect(screen.getByTestId('jlens-acquire-token')).toHaveValue('hf_READ');
    fireEvent.click(toggles()[1]);
    expect(screen.getByTestId('jlens-acquire-token')).toHaveValue('hf_WRITE');
  });

  it('clears a stale note on the PUBLISH branch too', async () => {
    /**
     * `setNote(null)` was applied to three call sites and pinned at one, so a
     * refused publish still rendered its red error beneath a green
     * "Publishing — queued as…" — round 1's defect verbatim, on the branch it
     * was not tested on.
     *
     * MUTATION CONTROL: delete `setNote(null)` from `runPublish`.
     */
    mount();
    open();
    const toggles = screen
      .getAllByRole('button')
      .filter((b) => b.hasAttribute('aria-pressed'));
    fireEvent.click(toggles[1]);
    fireEvent.change(screen.getByTestId('jlens-acquire-repo'), {
      target: { value: 'you/lenses' },
    });
    fireEvent.click(screen.getByTestId('jlens-publish-run'));
    await waitFor(() => expect(screen.getByText(/queued as/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /start another/ }));
    (jlensApi.publish as ReturnType<typeof vi.fn>).mockRejectedValue(
      new Error('401 no write access'),
    );
    fireEvent.click(screen.getByTestId('jlens-publish-run'));
    await waitFor(() =>
      expect(screen.getByRole('alert')).toHaveTextContent(/401/),
    );
    expect(screen.queryByText(/queued as/)).toBeNull();
  });

  it('RE-ARMS nothing when the model changes back', async () => {
    /**
     * The effect clears `queued`, so model A → B → A returned a form that would
     * queue a SECOND download of the same multi-gigabyte file — the
     * bandwidth-twice defect round 1 fixed, three clicks away. Each sub-clause
     * of the effect was also individually unpinned.
     *
     * MUTATION CONTROL: remove `setQueued(null)`, `setNote(null)` or
     * `setError(null)` from the effect body.
     */
    const { rerender } = mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/2 candidates/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    fireEvent.click(screen.getByTestId('jlens-acquire-run'));
    await waitFor(() =>
      expect(jlensApi.acquire as ReturnType<typeof vi.fn>).toHaveBeenCalledTimes(1),
    );
    expect(screen.getByTestId('jlens-queued-note')).toBeInTheDocument();

    const swap = (id: string) =>
      rerender(
        <AcquireLensCard
          modelId={id}
          modelRepoId="org/x"
          weightsPresent
          hasArtifact
        />,
      );
    swap('m_OTHER');
    swap('m_1');

    // The queued note is gone AND so is the selection, so nothing can be
    // re-queued without previewing again.
    expect(screen.queryByTestId('jlens-queued-note')).toBeNull();
    expect(screen.queryByText(/queued as/)).toBeNull();
    expect(screen.queryAllByRole('radio')).toHaveLength(0);
    expect(screen.getByTestId('jlens-acquire-run')).toBeDisabled();
  });

  it('says when the size check DID NOT RUN', async () => {
    /**
     * `fits_envelope` is null for three reasons, not one — no model named, no
     * size reported, or the model row lacking the dimensions to derive a bound.
     * A hard gate on `=== false` permits all three silently, which is the case
     * the preview exists for.
     *
     * MUTATION CONTROL: drop the `fits_envelope === null` note.
     */
    (jlensApi.previewRepo as ReturnType<typeof vi.fn>).mockResolvedValue({
      ...PREVIEW,
      candidates: [
        { ...PREVIEW.candidates[0], fits_envelope: null, envelope_detail: null },
      ],
    });
    mount();
    open();
    fireEvent.click(screen.getByRole('button', { name: /Look inside/ }));
    await waitFor(() => screen.getByText(/1 candidate/));
    fireEvent.click(screen.getAllByRole('radio')[0]);
    expect(
      screen.getByTestId('jlens-acquire-envelope-unknown'),
    ).toHaveTextContent(/did not run/);
    // It is a WARNING, not a block — the check may still pass server-side.
    expect(screen.getByTestId('jlens-acquire-run')).toBeEnabled();
  });

  it('references the dataset helper only when it EXISTS', () => {
    /**
     * The helper renders only on an invalid value and `mistudio` is valid, so
     * an unconditional `aria-describedby` dangled on every open.
     *
     * MUTATION CONTROL: make the attribute unconditional.
     */
    mount();
    open();
    const toggles = screen
      .getAllByRole('button')
      .filter((b) => b.hasAttribute('aria-pressed'));
    fireEvent.click(toggles[1]);
    const field = screen.getByTestId('jlens-publish-dataset');
    expect(field).not.toHaveAttribute('aria-describedby');

    fireEvent.change(field, { target: { value: 'bad/value' } });
    expect(screen.getByTestId('jlens-publish-dataset')).toHaveAttribute(
      'aria-describedby',
      'jlens-dataset-help',
    );
    expect(document.getElementById('jlens-dataset-help')).not.toBeNull();
  });
});


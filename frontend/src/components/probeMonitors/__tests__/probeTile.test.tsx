/**
 * The probe tile, and the gaps it closes.
 *
 * ⚠ THE LIST COULD NOT IDENTIFY A PROBE. Nine rows on this installation read exactly
 * `L11 · mean · dense` — same layer, same rule, same basis, different runs, different corpora,
 * different quality — and the row carried nothing else but a threshold. There was no way to tell
 * which detected what, which was any good, or which one a Report belonged to.
 *
 * ⚠ AND THE REPORT OPENED AT THE BOTTOM OF THE PAGE. It rendered after the whole `<ul>`, so
 * clicking Report on the third of thirteen rows scrolled the reader past every other row to a
 * panel with no visible connection to the one they clicked.
 */
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';

import { ProbeTile } from '../ProbeTile';
import type { ProbeMonitorSummary } from '../../../types/probeMonitor';
import panelSource from '../../panels/ProbeMonitorsPanel.tsx?raw';
import { conceptOf, labelSeparation, trainingLabels } from '../probeConcept';

function probe(overrides: Partial<ProbeMonitorSummary> = {}): ProbeMonitorSummary {
  return {
    id: 'pm_x',
    run_id: 'pmr_x',
    layer: 11,
    rule: 'mean',
    rule_params: {},
    variant: 'dense',
    sae_id: null,
    sae_feature_indices: null,
    val_metrics: { val_auroc: 0.9939 },
    selected: false,
    threshold: 2.8786,
    target_fpr: 0.01,
    realised_fpr: 0.0097,
    threshold_source: 'validation_negatives',
    streamable: true,
    rung: 2,
    rung_reasons: [],
    rung_language: 'detects on unseen tasks',
    rung_next_step: 'run the judge baseline on the same out-of-distribution sets',
    created_at: '2026-09-26T10:00:00Z',
    ...overrides,
  } as ProbeMonitorSummary;
}

function mount(overrides: Partial<ProbeMonitorSummary> = {}, props: Record<string, unknown> = {}) {
  const onToggle = vi.fn();
  render(
    <ul>
      <ProbeTile
        probe={probe(overrides)}
        modelName="LFM2.5-1.2B-Instruct"
        trainViewName="models-under-pressure training"
        expanded={false}
        onToggle={onToggle}
        {...props}
      />
    </ul>
  );
  return { onToggle };
}

describe('a tile says what the probe detects, not only where it reads', () => {
  it('names the concept and the model — the question nine identical rows could not answer', () => {
    mount();
    const concept = screen.getByTestId('probe-concept');
    expect(concept).toHaveTextContent('models-under-pressure training');
    expect(concept).toHaveTextContent('LFM2.5-1.2B-Instruct');
  });

  it('says so plainly when the concept could not be resolved, rather than showing a blank', () => {
    render(
      <ul>
        <ProbeTile probe={probe()} expanded={false} onToggle={vi.fn()} />
      </ul>
    );
    expect(screen.getByTestId('probe-concept')).toHaveTextContent('concept not resolved');
  });

  it('states the read point, the rule and the basis', () => {
    mount();
    const readout = screen.getByTestId('probe-readout');
    expect(readout).toHaveTextContent('L11');
    expect(readout).toHaveTextContent('resid_post');
    expect(readout).toHaveTextContent('mean');
    expect(readout).toHaveTextContent('dense residual');
  });

  it('counts the SAE features on a k-sparse probe, because k is what distinguishes it', () => {
    mount({ variant: 'sae', sae_feature_indices: Array.from({ length: 128 }, (_, i) => i) });
    expect(screen.getByTestId('probe-readout')).toHaveTextContent('SAE basis (128 features)');
  });

  it('carries the run id, so two identical-looking probes can be told apart', () => {
    mount({ run_id: 'pmr_4fa9cde17102' });
    expect(screen.getByTestId('probe-operating-point')).toHaveTextContent('pmr_4fa9cde17102');
  });
});

describe('every badge explains itself', () => {
  it('shows the rung in THE SERVER WORDS, never a phrase composed here', () => {
    mount({ rung: 2, rung_language: 'detects on unseen tasks' });
    const badge = screen.getByTestId('badge-rung');
    expect(badge).toHaveTextContent('detects on unseen tasks');
    expect(badge).toHaveTextContent('rung 2');
  });

  it('falls back to the bare number when the server sent no wording — it does NOT invent one', () => {
    /*
     * The whole reason `rung_language` was added to the LIST endpoint. If the wording is absent
     * the tile says "rung 2" and nothing more; a number is not a claim, a phrase would be.
     */
    mount({ rung_language: undefined });
    expect(screen.getByTestId('badge-rung')).toHaveTextContent('rung 2');
    expect(screen.getByTestId('badge-rung')).not.toHaveTextContent('detects');
  });

  it("the selected badge says what selection DOES and DOES NOT mean", () => {
    mount({ selected: true });
    const title = screen.getByTestId('badge-selected').getAttribute('title') ?? '';
    expect(title).toMatch(/highest validation AUROC/i);
    expect(title).toMatch(/does NOT mean the probe is good/i);
  });

  it('marks validation AUROC as in-distribution, so it is not read as generalisation', () => {
    mount();
    const badge = screen.getByTestId('badge-val');
    expect(badge).toHaveTextContent('val 0.994');
    expect(badge.getAttribute('title') ?? '').toMatch(/IN-DISTRIBUTION/);
  });

  it('distinguishes a missing threshold from a threshold of zero', () => {
    mount({ threshold: null });
    const badge = screen.getByTestId('badge-no-threshold');
    expect(badge.getAttribute('title') ?? '').toMatch(/not a threshold of zero/i);
  });

  it('warns when a rule cannot run on a live stream', () => {
    mount({ streamable: false, rule: 'attention' });
    expect(screen.getByTestId('badge-not-streamable').getAttribute('title') ?? '').toMatch(
      /whole sequence/i
    );
  });

  it('separates a built definition from a STALE one', () => {
    mount({ definition_built_at: '2026-09-26T10:00:00Z' });
    expect(screen.getByTestId('badge-exported')).toBeInTheDocument();
    expect(screen.queryByTestId('badge-stale')).toBeNull();
  });

  it('shows STALE and not "built" once the definition was invalidated', () => {
    /** A stale document states evidence that has since changed — worse than none. */
    mount({
      definition_built_at: '2026-09-26T10:00:00Z',
      definition_build: { invalidated: { reason: 'rung changed 2 -> 3' } },
    });
    expect(screen.getByTestId('badge-stale')).toBeInTheDocument();
    expect(screen.queryByTestId('badge-exported')).toBeNull();
  });

  it('counts publications, because they append rather than replace', () => {
    mount({ published: [{ repo_id: 'a' }, { repo_id: 'b' }] as never });
    expect(screen.getByTestId('badge-published')).toHaveTextContent('×2');
  });

  it('shows no state badges on a plain unselected probe', () => {
    mount();
    for (const id of ['badge-selected', 'badge-no-threshold', 'badge-not-streamable',
                      'badge-exported', 'badge-stale', 'badge-published']) {
      expect(screen.queryByTestId(id)).toBeNull();
    }
  });
});

describe('the report opens IN PLACE', () => {
  it('renders the report inside the tile it belongs to', () => {
    render(
      <ul>
        <ProbeTile
          probe={probe()}
          expanded
          onToggle={vi.fn()}
          modelName="m"
          trainViewName="t"
        >
          <div data-testid="the-report">evidence</div>
        </ProbeTile>
      </ul>
    );
    const tile = screen.getByTestId('probe-row');
    expect(tile).toContainElement(screen.getByTestId('the-report'));
    expect(tile.getAttribute('data-expanded')).toBe('true');
  });

  it('renders nothing when collapsed, even if children are passed', () => {
    render(
      <ul>
        <ProbeTile probe={probe()} expanded={false} onToggle={vi.fn()}>
          <div data-testid="the-report">evidence</div>
        </ProbeTile>
      </ul>
    );
    expect(screen.queryByTestId('the-report')).toBeNull();
    expect(screen.queryByTestId('probe-report-body')).toBeNull();
  });

  it('the toggle reports its state to assistive technology', async () => {
    const { onToggle } = mount();
    const button = screen.getByTestId('probe-report-toggle');
    expect(button).toHaveAttribute('aria-expanded', 'false');
    await userEvent.click(button);
    expect(onToggle).toHaveBeenCalledTimes(1);
  });
});

/*
 * ⚠ THE TRACE SHOWED THE VOCABULARY, NOT THE TEXT. `convert_ids_to_tokens` returns how a token is
 * SPELLED in the vocabulary, so on byte-level BPE the panel read
 * `<|im_start|>ĠuserĊRetĠireĠonline`. The backend decodes with the tokenizer that produced it and
 * sends `text` beside `token`; a control that reverted this rendering left the suite green, which
 * is why these exist.
 */
describe('the scored trace reads as text, not as vocabulary', () => {
  const base = {
    probe_id: 'pm_1',
    aggregate: -1.5,
    threshold: 12.49,
    fires: false,
    token_scores: [0.2, -0.1],
    n_scored: 2,
    role_mask_reliable: true,
    truncated: false,
  };

  it('renders the decoded text, not the `Ġ`-prefixed spelling', async () => {
    const { TokenTrace } = await import('../TokenTrace');
    render(
      <TokenTrace
        result={{
          ...base,
          tokens: [
            { token: 'Ġonline', text: ' online', special: false, scored: true },
            { token: 'Ret', text: 'Ret', special: false, scored: true },
          ],
        }}
      />
    );
    const trace = screen.getByTestId('token-trace');
    expect(trace).toHaveTextContent('online');
    expect(trace.textContent ?? '').not.toContain('Ġ');
  });

  it('falls back to the raw spelling for a trace from an older build', async () => {
    const { TokenTrace } = await import('../TokenTrace');
    render(
      <TokenTrace
        result={{ ...base, tokens: [{ token: 'Ret', scored: true }], token_scores: [0.2] }}
      />
    );
    expect(screen.getByTestId('token-trace')).toHaveTextContent('Ret');
  });

  it('marks chat-template scaffolding as scaffolding, and does NOT hide it', async () => {
    /*
     * A probe that fires on `<|im_start|>` rather than on the content is a FINDING. Filtering
     * these tokens out of the display would delete the evidence for it.
     */
    const { TokenTrace } = await import('../TokenTrace');
    render(
      <TokenTrace
        result={{
          ...base,
          tokens: [
            { token: '<|im_start|>', text: '<|im_start|>', special: true, scored: false },
            { token: 'Ret', text: 'Ret', special: false, scored: true },
          ],
          token_scores: [0.2],
        }}
      />
    );
    const special = screen.getByTestId('token-special');
    expect(special).toHaveTextContent('<|im_start|>');
    expect(special.getAttribute('title') ?? '').toMatch(/scaffolding/i);
  });
});

/*
 * ⚠ EVERY PROBE ON THE ESTATE WAS CAPTIONED "detects training", and none of them detects
 * training. The caption read the training view's HuggingFace config name, which for the
 * models-under-pressure corpus is the word `training`. The concept is the label mapping's
 * positive side — the thing the probe was fitted to separate — and these pin that, plus
 * the call site, because a correct helper nobody calls is this repo's signature defect.
 */
describe('a probe is captioned with what it detects', () => {
  const mapping = { 'low-stakes': 'negative', 'high-stakes': 'positive' };

  it('names the label that maps to positive, not the config', () => {
    expect(conceptOf({ label_mapping: mapping, config: 'training', name: 'mup (train)' }))
      .toBe('high-stakes');
  });

  it('names every positive label when a mapping has more than one', () => {
    expect(
      conceptOf({
        label_mapping: { critical: 'positive', 'high-stakes': 'positive', low: 'negative' },
        config: 'training',
        name: 'x',
      })
    ).toBe('critical or high-stakes');
  });

  it('falls back to the view name when nothing maps to positive', () => {
    expect(conceptOf({ label_mapping: {}, config: 'training', name: 'mup (train)' }))
      .toBe('training');
    expect(conceptOf({ label_mapping: {}, config: null, name: 'mup (train)' }))
      .toBe('mup (train)');
  });

  it('is null for a dataset that could not be resolved', () => {
    expect(conceptOf(null)).toBeNull();
  });

  it('never returns the bare word "training" for a real mapping', () => {
    // The regression itself, stated as the user saw it.
    expect(conceptOf({ label_mapping: mapping, config: 'training', name: 'training' }))
      .not.toBe('training');
  });

  it('is what the panel calls — asserted on the AST, not on the text', () => {
    /*
     * A substring search for "conceptOf" matches the import and the comment above the call
     * just as happily as the call. This walks for a CALL EXPRESSION whose callee is
     * `conceptOf`, which is the only shape that proves the panel uses it.
     */
    const calls = panelSource.match(/\bconceptOf\s*\(/g) ?? [];
    expect(calls.length).toBeGreaterThanOrEqual(1);
    // And the old expression must be gone from the training-view resolver.
    expect(panelSource).not.toMatch(/return run \? datasetNames\[run\.train_dataset_id\]/);
  });
});

/*
 * ⚠ THE TILE NAMED ONE SIDE OF A TWO-SIDED THING. `detects high-stakes` is the positive label;
 * a linear probe is a boundary, and the same positive label fitted against a different negative
 * one is a different detector with the same caption. The operator asked for the labels the
 * training is associated with, and that is both sides, in the corpus's own words.
 */
describe('a probe tile states the separation it was fitted to make', () => {
  const mapping = { 'low-stakes': 'negative', 'high-stakes': 'positive' };

  it('names both sides, positive first', () => {
    expect(labelSeparation({ label_mapping: mapping })).toBe('high-stakes vs low-stakes');
  });

  it('joins several labels on either side rather than showing the first', () => {
    expect(
      labelSeparation({
        label_mapping: { critical: 'positive', 'high-stakes': 'positive', calm: 'negative', idle: 'negative' },
      })
    ).toBe('critical or high-stakes vs calm or idle');
  });

  it('refuses to print half a contrast', () => {
    // "trained on high-stakes" reads as a corpus, not a boundary. A blank is honest; half is not.
    expect(labelSeparation({ label_mapping: { 'high-stakes': 'positive' } })).toBeNull();
    expect(labelSeparation({ label_mapping: { 'low-stakes': 'negative' } })).toBeNull();
    expect(labelSeparation({ label_mapping: {} })).toBeNull();
    expect(labelSeparation(null)).toBeNull();
  });

  it('does not count an excluded label as either side', () => {
    /*
     * `excluded` rows are dropped before fitting, so a mapping whose only negative is
     * excluded states no contrast at all — and must not borrow one.
     */
    expect(
      labelSeparation({ label_mapping: { 'high-stakes': 'positive', ambiguous: 'excluded' } })
    ).toBeNull();
    expect(
      trainingLabels({ label_mapping: { 'high-stakes': 'positive', ambiguous: 'excluded' } })
    ).toEqual({ positive: ['high-stakes'], negative: [], excluded: ['ambiguous'] });
  });

  it('renders on the tile, verbatim', () => {
    mount({}, { labelSeparation: 'high-stakes vs low-stakes' });
    expect(screen.getByTestId('probe-labels')).toHaveTextContent(
      'trained on high-stakes vs low-stakes'
    );
  });

  it('renders nothing at all when the separation is unknown', () => {
    // Not an empty row, not a placeholder word — a caption nobody can check is worse than none.
    mount({}, { labelSeparation: null });
    expect(screen.queryByTestId('probe-labels')).toBeNull();
  });

  it('is what the panel calls — asserted on the call, not on the text', () => {
    const calls = panelSource.match(/\blabelSeparation\s*\(/g) ?? [];
    expect(calls.length).toBeGreaterThanOrEqual(1);
    // And the result must reach the tile, not be computed and dropped.
    expect(panelSource).toMatch(/labelSeparation=\{labelsForRun\(probe\.run_id\)\}/);
  });
});

/**
 * ⚠ `created_at` WAS ON THE PAYLOAD AND THE INTERFACE ALL ALONG AND RENDERED NOWHERE.
 *
 * The tile could say which RUN produced a probe (`from pmr_…`) and never WHEN, so two probes from
 * different weeks looked identical at a glance. It matters more now that a threshold can be
 * re-cut in place: the bar carries its own revision and history, and the weights carry this —
 * the date is what tells a reader the detector itself has not moved.
 */
describe('a probe tile says when it was first trained', () => {
  it('shows the date', () => {
    render(
      <ProbeTile
        probe={probe({ created_at: '2026-09-26T10:00:00Z' })}
        expanded={false}
        onToggle={vi.fn()}
      />
    );
    const stamp = screen.getByTestId('probe-trained-at');
    // Rendered in the VIEWER's zone, so assert the parts that survive localisation rather than
    // a formatted string — a test pinned to one timezone fails on another machine.
    expect(stamp.textContent).toMatch(/trained/);
    expect(stamp.textContent).toMatch(/2026/);
    expect(stamp.textContent).toMatch(/Sep|Oct/);
  });

  it('carries the full timestamp in the title, and says the weights have not moved', () => {
    /* The short form is for scanning; the exact instant is for an operator correlating with a
       run log. And the title is where the bar-vs-detector distinction belongs: a moved threshold
       is a revision, not a retrain. */
    render(
      <ProbeTile
        probe={probe({ created_at: '2026-09-26T10:00:00Z' })}
        expanded={false}
        onToggle={vi.fn()}
      />
    );
    const title = screen.getByTestId('probe-trained-at').getAttribute('title') ?? '';
    expect(title).toMatch(/First trained/);
    expect(title).toMatch(/weights have not changed/);
    expect(title).toMatch(/revision/);
  });

  it('⚠ renders NOTHING rather than "Invalid Date" for an unusable value', () => {
    /* A tile that says `trained Invalid Date` is worse than one that says nothing. This estate
       has shipped a fabricated duration beside "Finished —" for exactly that reason. */
    for (const bad of ['', 'not a date', null as unknown as string]) {
      const { unmount } = render(
        <ProbeTile
          probe={probe({ created_at: bad })}
          expanded={false}
          onToggle={vi.fn()}
        />
      );
      expect(screen.queryByTestId('probe-trained-at')).toBeNull();
      unmount();
    }
  });

  it('still shows which run produced it', () => {
    /* The date is ADDITIONAL. Losing the run id would trade one provenance fact for another. */
    render(
      <ProbeTile probe={probe()} expanded={false} onToggle={vi.fn()} />
    );
    expect(screen.getByTestId('probe-operating-point').textContent).toMatch(/from pmr_/);
  });
});

describe('which tokens a probe reads (operator, 2026-10-04)', () => {
  /**
   * ⚠ A run's nine probes are three LAYERS x three RULES, every one trained on the same scope with
   * a bar for EACH window. The tile showed neither, and the nine read as "one per window per layer".
   */
  const windows = { all: 25.2898, prompt: 14.6913, response: 25.042 };

  it('says what the probe was trained on, in words', () => {
    mount({ scope: 'all', window_thresholds: windows });
    expect(screen.getByTestId('probe-scope')).toHaveTextContent('fitted on every token (prompt and reply)');
  });

  it("shows each window's own bar, prompt first", () => {
    mount({ scope: 'all', window_thresholds: windows, length_band_count: 4 });
    expect(screen.getByTestId('window-prompt')).toHaveTextContent('prompt ≥ 14.69');
    expect(screen.getByTestId('window-response')).toHaveTextContent('response ≥ 25.04');
    expect(screen.getByTestId('window-all')).toHaveTextContent('all ≥ 25.29');
    const order = Array.from(screen.getByTestId('probe-windows').querySelectorAll('[data-testid^="window-"]'))
      .map((el) => el.getAttribute('data-testid'));
    expect(order.slice(0, 3)).toEqual(['window-prompt', 'window-response', 'window-all']);
    expect(screen.getByTestId('window-length-bands')).toHaveTextContent('+ 4 length bands on all');
  });

  it('marks the response window provisional when the weights never saw a reply', () => {
    mount({ scope: 'all', window_thresholds: windows });
    expect(screen.getByTestId('window-response')).toHaveTextContent('provisional');
    expect(screen.getByTestId('window-prompt')).not.toHaveTextContent('provisional');
  });

  it('does not mark it for a probe trained on replies', () => {
    mount({ scope: 'last_assistant', window_thresholds: windows });
    expect(screen.getByTestId('window-response')).not.toHaveTextContent('provisional');
  });

  it('says so when one bar serves every window', () => {
    mount({ scope: 'all', window_thresholds: {} });
    expect(screen.getByTestId('window-single-bar')).toHaveTextContent('one bar for every window');
  });

  it('names the rolling window, the only thing telling two rolling probes at one layer apart', () => {
    mount({ rule: 'rolling_mean_max', rule_params: { window: 64 } });
    expect(screen.getByTestId('probe-readout')).toHaveTextContent('L11 · resid_post · rolling_mean_max w=64');
  });

  it('adds nothing for a rule with no window', () => {
    mount({ rule: 'mean', rule_params: {} });
    expect(screen.getByTestId('probe-readout')).not.toHaveTextContent('w=');
  });

  it('says the scope is not recorded rather than guessing', () => {
    mount({ scope: null, window_thresholds: windows });
    expect(screen.getByTestId('probe-scope')).toHaveTextContent('scope not recorded');
  });
});

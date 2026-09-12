/**
 * Resume labels exactly what a job did not, and nothing it already decided.
 *
 * WHY THIS EXISTS. The outcome of a labeling attempt was recorded nowhere, so a
 * failure — written as a fake label with `label_source` and `labeled_at` both
 * set — was indistinguishable from finished work. 16,824 features across the
 * estate look labelled and are not. There was no way to ask what was left, and
 * the one action that looked like it would help, `retryLabeling`, had no caller,
 * dropped half the configuration and started a FULL extraction-wide job.
 *
 * MUTATION CONTROLS (each must turn a test red):
 *   * render the Resume button while the job is still running -> gating test fails
 *   * render it when `remaining` is 0                         -> nothing-left test fails
 *   * drop the count from the button label                    -> blind-action test fails
 *   * send `resumeLabeling` a config missing the endpoint or
 *     template                                                -> faithful-config test fails
 *   * have the strip render zeros when coverage fails         -> honest-failure test fails
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
import { renderWithProviders as render } from '../../test/renderWithProviders';

import { LabelingJobCard } from './LabelingJobCard';
import * as labelingAPI from '../../api/labeling';
import { LabelingCoverageStrip } from './LabelingCoverageStrip';
import { useLabelingStore } from '../../stores/labelingStore';
import { useLabelingPromptTemplatesStore } from '../../stores/labelingPromptTemplatesStore';

const coverage = {
  extraction_job_id: 'extr_1',
  total: 52_888,
  by_status: { pending: 38_301, failed: 14_556, succeeded: 31 },
  adjudicated: 31,
  remaining: 52_857,
  in_progress: 0,
  unclassified: 0,
  stale: null,
  failures_without_a_recorded_reason: 0,
  caveat: null,
  outstanding: 52_857,
  exhausted: 0,
  failure_reasons: [],
  resume_feature_ids: ['f1', 'f2'],
};

const job = {
  id: 'lbl_1',
  extraction_job_id: 'extr_1',
  status: 'completed',
  progress: 1,
  features_labeled: 27,
  labeling_method: 'openai_compatible',
  openai_model: null,
  openai_compatible_endpoint: 'http://millm.hitsai.local',
  openai_compatible_model: 'gemma-4-31b-GGUF:IQ4_XS',
  local_model: null,
  prompt_template_id: 'lpt_discrimination',
  max_tokens: 300,
  api_timeout: 120,
  filter_special: true,
  filter_single_char: true,
  filter_punctuation: true,
  filter_numbers: true,
  filter_fragments: true,
  filter_stop_words: false,
  save_requests_for_testing: false,
  save_requests_sample_rate: 1,
  export_format: 'both',
  save_poor_quality_labels: false,
  poor_quality_sample_rate: 1,
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
};

function stubStore(overrides: Record<string, unknown> = {}) {
  useLabelingStore.setState({
    fetchCoverage: vi.fn().mockResolvedValue(coverage),
    resumeLabeling: vi.fn().mockResolvedValue({ ...job, id: 'lbl_2' }),
    ...overrides,
  } as any);
  // Read the mocks BACK off the store rather than returning the ones built
  // here: with an override in play those are two different functions, and a
  // test asserting on the wrong one asserts nothing.
  const state = useLabelingStore.getState() as any;
  return { fetchCoverage: state.fetchCoverage, resumeLabeling: state.resumeLabeling };
}

function renderCard(jobOverrides: Record<string, unknown> = {}) {
  return render(
    <LabelingJobCard job={{ ...job, ...jobOverrides } as any} onDelete={vi.fn()} onCancel={vi.fn()} />,
  );
}

describe('Resume on a finished labeling job', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('offers to resume, and says how much work that is', async () => {
    stubStore();
    renderCard();

    // The count IS the safety feature: the action this replaces was a "Retry"
    // that silently restarted a 53,000-feature extraction from step zero.
    expect(await screen.findByText('Resume 2 of 52,857')).toBeInTheDocument();
  });

  it('promises only what one click actually labels', async () => {
    /*
     * `remaining` is the backlog; one click labels at most the ids the coverage
     * response returned, capped at 2000 by the panel route. The first version of
     * this button read "Resume (52,857)" and labelled 2,000 — overstating by 26x
     * the one number that exists to stop a click being a blind bulk action.
     */
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        resume_feature_ids: Array.from({ length: 2000 }, (_, i) => `f${i}`),
      }),
    });
    renderCard();

    expect(await screen.findByText('Resume 2,000 of 52,857')).toBeInTheDocument();
    expect(screen.queryByText('Resume (52,857)')).not.toBeInTheDocument();
  });

  it('drops the "of N" once one click finishes the job', async () => {
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        remaining: 2,
        outstanding: 52_857,
  exhausted: 0,
  failure_reasons: [],
  resume_feature_ids: ['f1', 'f2'],
      }),
    });
    renderCard();

    expect(await screen.findByText('Resume (2)')).toBeInTheDocument();
  });

  it('does not offer to resume a job that is still running', async () => {
    const { fetchCoverage } = stubStore();
    renderCard({ status: 'labeling' });

    // Everything outstanding is work this job has not reached yet; a second job
    // would compete with it for the same features.
    await waitFor(() => expect(fetchCoverage).not.toHaveBeenCalled());
    expect(screen.queryByText(/^Resume/)).not.toBeInTheDocument();
  });

  it('hides Resume the moment a job goes back to running, even with coverage in hand', async () => {
    /*
     * TWO gates guard this button and they fail differently. The effect gate
     * stops coverage being FETCHED for a running job; the render gate stops the
     * button being SHOWN. The "still running" test above only exercises the
     * first — with coverage never fetched, the render gate is never reached, so
     * removing it entirely left all ten tests green (mutation control F1).
     *
     * This loads coverage on a finished job, then re-renders it as running, so
     * the render gate is the only thing that can hide the button.
     */
    stubStore();
    const { rerender } = renderCard({ status: 'completed' });
    expect(await screen.findByText('Resume 2 of 52,857')).toBeInTheDocument();

    rerender(
      <LabelingJobCard
        job={{ ...job, status: 'labeling' } as any}
        onDelete={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.queryByText(/^Resume/)).not.toBeInTheDocument();
  });

  it('offers nothing when nothing is left', async () => {
    const { fetchCoverage } = stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        by_status: { succeeded: 52_888 },
        adjudicated: 52_888,
        remaining: 0,
        resume_feature_ids: [],
      }),
    });
    renderCard();

    await waitFor(() => expect(fetchCoverage).toHaveBeenCalled());
    // A disabled button invites a click and does nothing; absence is clearer.
    expect(screen.queryByText(/^Resume/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Resume all/)).not.toBeInTheDocument();
  });

  it('renders no Resume button when coverage cannot be read', async () => {
    stubStore({ fetchCoverage: vi.fn().mockRejectedValue(new Error('502')) });
    renderCard();

    await waitFor(() => expect(screen.queryByText(/^Resume/)).not.toBeInTheDocument());
  });

  it('resumes with the SAME judge and template, not a default config', async () => {
    const { resumeLabeling } = stubStore();
    renderCard();

    fireEvent.click(await screen.findByText('Resume 2 of 52,857'));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalledTimes(1));
    const [extractionId, config] = resumeLabeling.mock.calls[0];
    expect(extractionId).toBe('extr_1');
    // Asserting the PAYLOAD, not just that it was called: `retryLabeling` was
    // called correctly too and sent the wrong thing. Every one of these was a
    // field it dropped.
    expect(config).toMatchObject({
      extraction_job_id: 'extr_1',
      labeling_method: 'openai_compatible',
      openai_compatible_endpoint: 'http://millm.hitsai.local',
      openai_compatible_model: 'gemma-4-31b-GGUF:IQ4_XS',
      prompt_template_id: 'lpt_discrimination',
      max_tokens: 300,
      api_timeout: 120,
      filter_stop_words: false,
    });
  });

  it('never sends a hardcoded batch size', async () => {
    const { resumeLabeling } = stubStore();
    renderCard();
    fireEvent.click(await screen.findByText('Resume 2 of 52,857'));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    // `retryLabeling` hardcoded `batch_size: 10` over whatever the job used.
    expect(resumeLabeling.mock.calls[0][1]).not.toHaveProperty('batch_size');
  });
});

describe('Coverage strip', () => {
  beforeEach(() => vi.clearAllMocks());

  it('separates adjudicated, failed and never attempted', async () => {
    stubStore();
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    expect(await screen.findByText('31 adjudicated')).toBeInTheDocument();
    expect(screen.getByText('14,556 failed')).toBeInTheDocument();
    expect(screen.getByText('38,301 not attempted')).toBeInTheDocument();
  });

  it('renders nothing rather than zeros when coverage cannot be read', async () => {
    stubStore({ fetchCoverage: vi.fn().mockRejectedValue(new Error('502')) });
    const { container } = render(<LabelingCoverageStrip extractionId="extr_1" />);

    // Zeros would read as "all done" — the same lie the fake labels told.
    await waitFor(() => expect(container).toBeEmptyDOMElement());
  });

  it('says when failures carry no recorded reason', async () => {
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        failures_without_a_recorded_reason: 2_161,
        caveat: 'these failures carry no recorded reason',
      }),
    });
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    expect(
      await screen.findByText(/2,161 failures\s+carry no recorded reason/),
    ).toBeInTheDocument();
  });

  it('reports a status it does not recognise instead of counting it as done', async () => {
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        by_status: { succeeded: 10, quarantined: 4 },
        adjudicated: 10,
        remaining: 0,
        unclassified: 4,
      }),
    });
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    expect(await screen.findByText('4 unrecognised')).toBeInTheDocument();
  });
});


describe('Failure reasons', () => {
  beforeEach(() => vi.clearAllMocks());

  const withReasons = (failure_reasons: unknown[]) =>
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({ ...coverage, failure_reasons }),
    });

  it('is collapsed by default — a diagnostic, not a status', async () => {
    withReasons([
      { reason: 'APIError', count: 14102, sample_feature_ids: ['f1'] },
      { reason: '(no reason recorded)', count: 454, sample_feature_ids: ['f2'] },
    ]);
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    expect(await screen.findByText(/Why 2 failure reasons/)).toBeInTheDocument();
    expect(screen.queryByText('APIError')).not.toBeInTheDocument();
  });

  it('shows the causes and their counts when opened', async () => {
    withReasons([
      { reason: 'APIError', count: 14102, sample_feature_ids: ['feat_a'] },
      { reason: '(no reason recorded)', count: 454, sample_feature_ids: [] },
    ]);
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    fireEvent.click(await screen.findByText(/Why 2 failure reasons/));

    expect(screen.getByText('14,102')).toBeInTheDocument();
    // The placeholder is NAMED, not dropped: on live data it is the top row at
    // 16,970, and it is the most important thing on the report.
    expect(screen.getByText('(no reason recorded)')).toBeInTheDocument();
    expect(screen.getByText('454')).toBeInTheDocument();
  });

  it('offers a real feature so a count can be checked rather than believed', async () => {
    withReasons([{ reason: 'APIError', count: 3, sample_feature_ids: ['feat_abc', 'feat_def'] }]);
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    fireEvent.click(await screen.findByText(/Why 1 failure reason/));
    expect(screen.getByText('e.g. feat_abc')).toBeInTheDocument();
  });

  it('shows nothing when nothing failed', async () => {
    withReasons([]);
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    await screen.findByText('31 adjudicated');
    expect(screen.queryByText(/failure reason/)).not.toBeInTheDocument();
  });

  it('reports its open state to assistive tech', async () => {
    withReasons([{ reason: 'APIError', count: 1, sample_feature_ids: [] }]);
    render(<LabelingCoverageStrip extractionId="extr_1" />);

    const toggle = (await screen.findByText(/Why 1 failure reason/)).closest('button')!;
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');
  });
});


describe('Try a sample first', () => {
  beforeEach(() => vi.clearAllMocks());

  it('offers a sample when there are failures to sample', async () => {
    stubStore();
    renderCard();
    expect(await screen.findByText('Try 20 first')).toBeInTheDocument();
  });

  it('offers no sample when nothing has failed', async () => {
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        by_status: { pending: 38_301 },
        remaining: 38_301,
      }),
    });
    renderCard();

    // Anchored on the batch button specifically: "Resume all" also starts with
    // "Resume", so /^Resume/ is ambiguous once the sweep control exists.
    await screen.findByText(/^Resume \d/);
    // "Do the failures still fail?" is not a question worth asking here.
    expect(screen.queryByText('Try 20 first')).not.toBeInTheDocument();
  });

  it('samples FAILURES only, and only twenty', async () => {
    const { resumeLabeling } = stubStore();
    renderCard();

    fireEvent.click(await screen.findByText('Try 20 first'));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalledTimes(1));
    // Asserting the third argument, not just that it was called: a sample that
    // quietly takes the default batch is a 2,000-feature job wearing the label
    // of a three-minute one.
    expect(resumeLabeling.mock.calls[0][2]).toEqual({ limit: 20, only: 'failed' });
  });

  it('runs the sample through the same path as a full resume', async () => {
    /*
     * A sample exists to PREDICT the full run. Taken down a different code
     * path it predicts that path instead — different judge resolution,
     * different claiming, different everything that could be the actual cause.
     */
    const { resumeLabeling } = stubStore();
    renderCard();

    fireEvent.click(await screen.findByText('Try 20 first'));
    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    const sampleConfig = resumeLabeling.mock.calls[0][1];

    vi.clearAllMocks();
    fireEvent.click(screen.getByText('Resume 2 of 52,857'));
    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());

    expect(resumeLabeling.mock.calls[0][1]).toEqual(sampleConfig);
  });

  it('a full resume narrows nothing', async () => {
    const { resumeLabeling } = stubStore();
    renderCard();

    fireEvent.click(await screen.findByText('Resume 2 of 52,857'));
    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    expect(resumeLabeling.mock.calls[0][2]).toEqual({});
  });
});


describe('Resume all (sweep)', () => {
  beforeEach(() => vi.clearAllMocks());

  const manyBatches = {
    ...coverage,
    remaining: 52_857,
    resume_feature_ids: Array.from({ length: 2000 }, (_, i) => `f${i}`),
  };

  it('never starts a sweep on one click', async () => {
    /*
     * Sweeping L46 is ~59 GPU-hours. A button that starts that from a single
     * click is an accident waiting to happen, not a feature.
     */
    const start = vi.spyOn(labelingAPI, 'startResumeSweep').mockResolvedValue({} as never);
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(manyBatches) });
    renderCard();

    fireEvent.click(await screen.findByText(/Resume all \(27 batches\)/));

    expect(start).not.toHaveBeenCalled();
    expect(screen.getByText(/Continue\?/)).toBeInTheDocument();
  });

  it('states the cost in GPU-hours before asking', async () => {
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(manyBatches) });
    renderCard();

    fireEvent.click(await screen.findByText(/Resume all \(27 batches\)/));

    // 52,857 x 8 s = ~117 hours.
    expect(screen.getByText(/27 batches, about 117 GPU-hours/)).toBeInTheDocument();
  });

  it('sends an explicit ceiling, never an open-ended sweep', async () => {
    // A COMPLETE sweep, because the API always returns one. The first version
    // of this mock returned `{id, status}` only, and the card then crashed on
    // `features_labeled.toLocaleString()` — reported by vitest as an Unhandled
    // Error while the test itself still "passed". A mock that understates the
    // contract tests a response the server never sends.
    const start = vi.spyOn(labelingAPI, 'startResumeSweep').mockResolvedValue({
      id: 'sweep_1',
      extraction_job_id: 'extr_1',
      status: 'running',
      batches_done: 0,
      max_batches: 27,
      batch_size: 2000,
      features_labeled: 0,
      features_failed: 0,
      last_labeling_job_id: null,
      cancel_requested_at: null,
      error_message: null,
      created_at: new Date().toISOString(),
      completed_at: null,
    });
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(manyBatches) });
    renderCard();

    fireEvent.click(await screen.findByText(/Resume all \(27 batches\)/));
    fireEvent.click(screen.getByText('Start sweep'));

    await waitFor(() => expect(start).toHaveBeenCalledTimes(1));
    const [extractionId, maxBatches, config] = start.mock.calls[0];
    expect(extractionId).toBe('extr_1');
    expect(maxBatches).toBe(27);
    // The same judge as the job it continues — asserting the payload, not the call.
    expect(config).toMatchObject({
      openai_compatible_endpoint: 'http://millm.hitsai.local',
      prompt_template_id: 'lpt_discrimination',
    });
  });

  it('offers no sweep when a single batch finishes the job', async () => {
    /*
     * A sweep is machinery for spanning many batches. Offering it for work that
     * one Resume completes is noise, and worse, it implies the plain button
     * would not finish.
     */
    stubStore({
      fetchCoverage: vi.fn().mockResolvedValue({
        ...coverage,
        remaining: 2,
        resume_feature_ids: ['f1', 'f2'],
      }),
    });
    renderCard();

    expect(await screen.findByText('Resume (2)')).toBeInTheDocument();
    expect(screen.queryByText(/Resume all/)).not.toBeInTheDocument();
  });
});


describe('Staleness is reachable', () => {
  beforeEach(() => vi.clearAllMocks());

  const template = {
    id: 'lpt_discrimination',
    name: 'Discrimination',
    prompt_fingerprint: 'a'.repeat(64),
  };

  function withTemplate(t: unknown | null) {
    useLabelingPromptTemplatesStore.setState({
      templates: t ? [t] : [],
      fetchTemplate: vi.fn().mockResolvedValue(t),
    } as never);
  }

  it('asks coverage WHICH judge produced the verdicts', async () => {
    /*
     * WS4 exists because this call had no caller. The fingerprint column, the
     * staleness branch and the query parameters were all built and nothing sent
     * a value — a capability wired to nothing is not shipped, however well it is
     * tested in isolation.
     */
    withTemplate(template);
    const { fetchCoverage } = stubStore();
    renderCard();

    await waitFor(() => expect(fetchCoverage).toHaveBeenCalled());
    expect(fetchCoverage.mock.calls[0][1]).toEqual({
      promptFingerprint: 'a'.repeat(64),
      judgeModel: 'gemma-4-31b-GGUF:IQ4_XS',
    });
  });

  it('resumes with the SAME predicate the count was computed with', async () => {
    /*
     * TWO ELIGIBILITY DEFINITIONS, ONE ON EACH PATH.
     *
     * The card reads coverage WITH the fingerprint and judge, so the number on
     * the button counts verdicts that are stale for this template. The resume
     * action re-fetched WITHOUT them and took only pending + failed. The button
     * therefore offered work it would never do.
     *
     * `eligible_count_query`'s own docstring is about exactly this failure —
     * the UI said "Resume 0 of 39" and clicking answered "Nothing left to
     * label" — and it was fixed INSIDE the predicate. Passing different
     * arguments to that predicate reintroduces it from outside, and after a
     * migration that changes every fingerprint it does so across the whole
     * estate.
     *
     * MUTATION CONTROL: drop promptFingerprint/judgeModel from the
     * resumeLabeling call in LabelingJobCard -> this fails.
     */
    withTemplate(template);
    const { fetchCoverage, resumeLabeling } = stubStore();
    renderCard();

    await waitFor(() => expect(fetchCoverage).toHaveBeenCalled());
    const displayPredicate = fetchCoverage.mock.calls[0][1];

    // `/Resume/` also matches "Resume all (N batches)" (the sweep) and the
    // sample button. The full-resume control is the one whose label states the
    // batch it will take out of the backlog.
    fireEvent.click(screen.getByText(/^Resume \d/));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    const actionOptions = resumeLabeling.mock.calls[0][2] ?? {};

    expect({
      promptFingerprint: actionOptions.promptFingerprint,
      judgeModel: actionOptions.judgeModel,
    }).toEqual(displayPredicate);
  });

  it('starts a sweep with the predicate its batch count was sized from', async () => {
    /*
     * R4. The sweep is quoted from a coverage read that carries the fingerprint
     * and judge; selecting without them makes every batch take a different set
     * than the operator confirmed. At the 100-batch ceiling that is up to ~444
     * GPU-hours booked against work the sweep will not do.
     *
     * R3 fixed this on three surfaces and added zero frontend tests, so
     * deleting the two lines that send it passed all 1400.
     *
     * MUTATION CONTROL: drop prompt_fingerprint/judge_model from the
     * startResumeSweep call in LabelingJobCard -> this fails.
     */
    withTemplate(template);
    const { fetchCoverage } = stubStore();
    const startSweep = vi
      .spyOn(labelingAPI, 'startResumeSweep')
      .mockResolvedValue({ id: 'sweep_1', status: 'running' } as never);
    renderCard();

    await waitFor(() => expect(fetchCoverage).toHaveBeenCalled());
    const displayPredicate = fetchCoverage.mock.calls[0][1];

    fireEvent.click(screen.getByText(/Resume all/));
    // `/^Start/` also matches the card's "Started: <date>" line.
    fireEvent.click(screen.getByText('Start sweep'));

    await waitFor(() => expect(startSweep).toHaveBeenCalled());
    const sentConfig = startSweep.mock.calls[0][2];

    expect({
      promptFingerprint: sentConfig.prompt_fingerprint,
      judgeModel: sentConfig.judge_model,
    }).toEqual(displayPredicate);
    // The config the sweep freezes must also still carry the judge itself, or
    // the predicate names a model the batches will not use.
    expect(sentConfig.openai_compatible_model).toBeDefined();
  });

  it('asks about the judge that will RUN, not the one that ran before', async () => {
    /*
     * R4. `buildConfig()` sends `judgeOverride ?? job.openai_compatible_model`,
     * and the override is set AUTOMATICALLY when the original judge is gone —
     * the case this feature exists for. Naming the original judge in the
     * staleness predicate made it exactly backwards: verdicts from the judge
     * about to run were marked stale and redone, while verdicts from the judge
     * that is gone were treated as fresh and skipped. A resume under an
     * override never converged.
     *
     * MUTATION CONTROL: remove `judgeOverride ||` from coveragePredicate
     * -> this fails.
     */
    withTemplate(template);
    vi.spyOn(labelingAPI, 'getAvailableJudges').mockResolvedValue({
      original_available: false,
      models: ['a-different-judge'],
    } as never);
    const { fetchCoverage } = stubStore();
    renderCard();

    // The override is preselected, then the predicate must follow it.
    await waitFor(() => {
      const last = fetchCoverage.mock.calls.at(-1)?.[1] as
        | { judgeModel?: string }
        | undefined;
      expect(last?.judgeModel).toBe('a-different-judge');
    });
  });

  it('sends BOTH halves or neither', async () => {
    /*
     * A fingerprint without a model marks every same-template/different-model
     * verdict as fresh; a model without a fingerprint cannot see a template
     * edit. Half an answer is worse than none, because it looks like an answer.
     */
    withTemplate(null);
    const { fetchCoverage } = stubStore();
    renderCard();

    await waitFor(() => expect(fetchCoverage).toHaveBeenCalled());
    expect(fetchCoverage.mock.calls[0][1]).toBeUndefined();
  });

  it('sends nothing when the job names no template', async () => {
    withTemplate(template);
    const { fetchCoverage } = stubStore();
    renderCard({ prompt_template_id: null });

    await waitFor(() => expect(fetchCoverage).toHaveBeenCalled());
    expect(fetchCoverage.mock.calls[0][1]).toBeUndefined();
  });
});


describe('Exhausted retries', () => {
  beforeEach(() => vi.clearAllMocks());

  /*
   * REPORTED FROM THE UI. A card read "Resume 0 of 39" and clicking it answered
   * "Nothing left to label in this extraction." The 39 failures had each been
   * attempted three times, so the batch query excluded them while the coverage
   * count still reported them as remaining.
   */
  const spent = {
    ...coverage,
    total: 8164,
    by_status: { failed: 39, succeeded: 8125 },
    adjudicated: 8125,
    outstanding: 39,
    remaining: 0,
    exhausted: 39,
    resume_feature_ids: [],
  };

  it('offers no Resume button when nothing can actually be taken', async () => {
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(spent) });
    renderCard();

    await screen.findByText(/exhausted retries/);
    expect(screen.queryByText(/^Resume/)).not.toBeInTheDocument();
  });

  it('never renders "Resume 0 of N"', async () => {
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(spent) });
    renderCard();

    await screen.findByText(/exhausted retries/);
    expect(screen.queryByText(/Resume 0 of/)).not.toBeInTheDocument();
  });

  it('explains the gap between outstanding and remaining', async () => {
    /*
     * 39 failed with no button is arithmetic an operator cannot reconcile.
     * Saying nothing is what produced the bug report.
     */
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(spent) });
    renderCard();

    expect(await screen.findByText('39 exhausted retries')).toBeInTheDocument();
  });

  it('offers no sample either — the cap excludes those too', async () => {
    stubStore({ fetchCoverage: vi.fn().mockResolvedValue(spent) });
    renderCard();

    await screen.findByText(/exhausted retries/);
    expect(screen.queryByText('Try 20 first')).not.toBeInTheDocument();
  });

  it('says nothing about exhaustion when there is none', async () => {
    stubStore();
    renderCard();

    // Anchored on the batch button; "Resume all" also starts with "Resume".
    await screen.findByText(/^Resume \d/);
    expect(screen.queryByText(/exhausted retries/)).not.toBeInTheDocument();
  });
});


describe('Choosing a different judge', () => {
  beforeEach(() => vi.clearAllMocks());

  /*
   * THE POINT OF RESUME, per the reported incident. A job's judge
   * (`granite-3.3-8b-instruct`) was removed from the server months after the
   * run. Resume reused it faithfully, every feature 404'd, and there was no way
   * through the UI to pick a model that exists.
   */
  const gone = {
    labeling_job_id: 'lbl_1',
    endpoint: 'http://millm.hitsai.local',
    original_model: 'granite-3.3-8b-instruct',
    original_available: false,
    models: ['gemma-4-31b-it-3MPER0RR-abliterated-GGUF:IQ4_XS', 'granite-4.2-8b'],
    reachable: true,
    detail: null,
  };
  // The original judge IS the job's own model — they are the same value in
  // reality, and a fixture where they differ tests a state that cannot occur.
  const present = {
    ...gone,
    original_model: job.openai_compatible_model,
    original_available: true,
    models: [job.openai_compatible_model, 'granite-4.2-8b'],
  };

  function withJudges(j: unknown) {
    return vi.spyOn(labelingAPI, 'getAvailableJudges').mockResolvedValue(j as never);
  }

  it('says the original judge is unavailable', async () => {
    withJudges(gone);
    stubStore();
    renderCard();

    expect(await screen.findByText(/Original judge unavailable/)).toBeInTheDocument();
  });

  it('preselects a judge that exists, so resume is one click away', async () => {
    withJudges(gone);
    stubStore();
    renderCard();

    const select = await screen.findByLabelText('Model to label with');
    expect((select as HTMLSelectElement).value)
      .toBe('gemma-4-31b-it-3MPER0RR-abliterated-GGUF:IQ4_XS');
  });

  it('resumes with the chosen judge, not the dead one', async () => {
    withJudges(gone);
    const { resumeLabeling } = stubStore();
    renderCard();

    await screen.findByLabelText('Model to label with');
    fireEvent.click(screen.getByText(/^Resume \d/));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    expect(resumeLabeling.mock.calls[0][1].openai_compatible_model)
      .toBe('gemma-4-31b-it-3MPER0RR-abliterated-GGUF:IQ4_XS');
  });

  it('lets a different judge be selected explicitly', async () => {
    withJudges(gone);
    const { resumeLabeling } = stubStore();
    renderCard();

    const select = await screen.findByLabelText('Model to label with');
    fireEvent.change(select, { target: { value: 'granite-4.2-8b' } });
    fireEvent.click(screen.getByText(/^Resume \d/));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    expect(resumeLabeling.mock.calls[0][1].openai_compatible_model).toBe('granite-4.2-8b');
  });

  it('keeps the original when it is still served', async () => {
    /* Both halves of one run stay comparable unless someone says otherwise. */
    withJudges(present);
    const { resumeLabeling } = stubStore();
    renderCard();

    await screen.findByText('Judge');
    fireEvent.click(screen.getByText(/^Resume \d/));

    await waitFor(() => expect(resumeLabeling).toHaveBeenCalled());
    expect(resumeLabeling.mock.calls[0][1].openai_compatible_model)
      .toBe(job.openai_compatible_model);
  });

  it('shows no picker when the endpoint cannot be reached', async () => {
    withJudges({ ...gone, reachable: false, models: [] });
    stubStore();
    renderCard();

    await screen.findByText(/^Resume \d/);
    expect(screen.queryByLabelText('Model to label with')).not.toBeInTheDocument();
  });
});

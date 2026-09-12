/**
 * LabelingJobCard Component
 *
 * Displays an individual labeling job with status, progress, and actions.
 */

import React, { useEffect, useMemo, useState } from 'react';
import { Tag, Loader, CheckCircle, XCircle, Trash2, Clock, Ban, PlayCircle, FlaskConical, FastForward, Square } from 'lucide-react';
import type { LabelingJob } from '../../types/labeling';
import { LabelingStatus } from '../../types/labeling';
import { format } from 'date-fns';
import { COMPONENTS } from '../../config/brand';
import { useLabelingPromptTemplatesStore } from '../../stores/labelingPromptTemplatesStore';
import { LabelingResultsWindow } from './LabelingResultsWindow';
import { useLabelingStore } from '../../stores/labelingStore';
import type { LabelingCoverage, LabelingResumeSweep, AvailableJudges } from '../../types/labeling';
import * as labelingAPI from '../../api/labeling';

/**
 * How many failures a sample retries. Twenty at the measured ~8 s/feature is
 * under three minutes — small enough to spend without a decision, large
 * enough that 0/20 and 18/20 mean genuinely different things.
 */
const SAMPLE_SIZE = 20;

interface LabelingJobCardProps {
  job: LabelingJob;
  onCancel?: () => void;
  onDelete?: () => void;
}

export const LabelingJobCard: React.FC<LabelingJobCardProps> = ({
  job,
  onCancel,
  onDelete,
}) => {
  const [templateName, setTemplateName] = useState<string>('Default Template');
  const [coverage, setCoverage] = useState<LabelingCoverage | null>(null);
  const [isResuming, setIsResuming] = useState(false);
  const [sweep, setSweep] = useState<LabelingResumeSweep | null>(null);
  const [sweepPrompt, setSweepPrompt] = useState(false);
  /*
   * WHICH JUDGE A RESUME USES.
   *
   * Defaulting to the job's own model is right when comparing halves of one
   * run, and useless when that model is GONE — which is exactly when someone
   * needs to resume. A judge removed between the original run and the resume
   * left every remaining feature unlabelable, with no route through the UI.
   *
   * `null` means "whatever the job used". A value overrides it, and
   * `label_model` + `label_prompt_fingerprint` are recorded per feature, so an
   * extraction labelled by two judges stays honest about which produced what.
   */
  const [judges, setJudges] = useState<AvailableJudges | null>(null);
  const [judgeOverride, setJudgeOverride] = useState<string | null>(null);
  const fetchCoverage = useLabelingStore((s) => s.fetchCoverage);
  const resumeLabeling = useLabelingStore((s) => s.resumeLabeling);
  const { templates, fetchTemplate } = useLabelingPromptTemplatesStore();

  /*
   * WS4: ask coverage which verdicts came from a DIFFERENT judge.
   *
   * Both halves are required. A fingerprint alone marks every
   * same-template/different-model verdict as fresh, and a model alone cannot
   * see a template edit.
   *
   * ONE DEFINITION, used by BOTH the count and the click. It lived inside the
   * effect, so the action could not see it and resumed with a different
   * predicate than the button had counted with — the button then offered a
   * number it would never deliver.
   */
  const coveragePredicate = useMemo(() => {
    const template = templates.find((t) => t.id === job.prompt_template_id);
    /*
     * THE JUDGE THAT WILL RUN, not the one that ran last time.
     *
     * `buildConfig()` sends `judgeOverride ?? job.openai_compatible_model`, and
     * the override is set AUTOMATICALLY whenever the original judge is no
     * longer served — which is the case this whole feature exists for. Naming
     * the ORIGINAL judge here made staleness exactly backwards: verdicts from
     * the judge that is about to run were marked stale and redone, while
     * verdicts from the judge that is gone were treated as fresh and skipped.
     *
     * A resume under an override then never converged: each pass re-did the
     * previous pass's work and left the real backlog untouched.
     */
    const judgeModel =
      judgeOverride ||
      job.openai_compatible_model ||
      job.openai_model ||
      job.local_model ||
      undefined;
    return template?.prompt_fingerprint && judgeModel
      ? { promptFingerprint: template.prompt_fingerprint, judgeModel }
      : undefined;
  }, [
    templates,
    job.prompt_template_id,
    judgeOverride,
    job.openai_compatible_model,
    job.openai_model,
    job.local_model,
  ]);
  const [, setTick] = useState(0); // Force re-render for elapsed time updates

  const isActive = job.status === LabelingStatus.QUEUED || job.status === LabelingStatus.LABELING;

  // Timer to update elapsed time every second while job is active
  useEffect(() => {
    if (!isActive) return;

    const timer = setInterval(() => {
      setTick(t => t + 1); // Force re-render to update elapsed time
    }, 1000);

    return () => clearInterval(timer);
  }, [isActive]);
  const isCompleted = job.status === LabelingStatus.COMPLETED;
  const isFailed = job.status === LabelingStatus.FAILED;
  const isCancelled = job.status === LabelingStatus.CANCELLED;
  const isFinished = isCompleted || isFailed || isCancelled;

  // Only once the job has stopped: while it runs, everything outstanding is
  // simply work it has not reached yet, and offering to resume it would start a
  // second job competing for the same features.
  useEffect(() => {
    if (!isFinished) return;
    let cancelled = false;
    labelingAPI
      .getAvailableJudges(job.id)
      .then((j) => {
        if (cancelled) return;
        setJudges(j);
        // Preselect a working judge when the original is gone, so the operator
        // is one click from a resume rather than stuck at a dead button.
        if (!j.original_available && j.models.length > 0) {
          setJudgeOverride((current) => current ?? j.models[0]);
        }
      })
      .catch(() => undefined);

    fetchCoverage(job.extraction_job_id, coveragePredicate)
      .then((result) => {
        if (!cancelled) setCoverage(result);
      })
      // A coverage read that fails must not render a Resume button: its label
      // carries a count, and a button offering an unknown amount of work is
      // exactly the blind bulk action this replaces.
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [isFinished, job.extraction_job_id, job.status, job.id, coveragePredicate, fetchCoverage]);

  /*
   * THE LABEL MUST DESCRIBE THE CLICK, NOT THE BACKLOG.
   *
   * `remaining` is the whole outstanding set — 52,857 on the L46 extraction —
   * but one click labels at most `resume_feature_ids.length`, capped at 2000 by
   * the panel route (Celery's 10 h soft limit at the measured ~8 s/feature). A
   * button reading "Resume (52,857)" that labels 2,000 overstates by 26x, and
   * the count was the entire reason the button carries one: it exists so a
   * click is never a blind bulk action.
   *
   * Batch size is read from the ids themselves rather than a hardcoded 2000, so
   * the frontend cannot drift from the backend's cap.
   */
  const batchSize = coverage?.resume_feature_ids.length ?? 0;
  const batchesRemaining =
    coverage && batchSize > 0 ? Math.ceil(coverage.remaining / batchSize) : 0;
  const resumeLabel =
    coverage && batchSize < coverage.remaining
      ? `Resume ${batchSize.toLocaleString()} of ${coverage.remaining.toLocaleString()}`
      : `Resume (${(coverage?.remaining ?? 0).toLocaleString()})`;

  const buildConfig = () => ({
    extraction_job_id: job.extraction_job_id,
    labeling_method: job.labeling_method,
    openai_model: job.openai_model ?? undefined,
    openai_compatible_endpoint: job.openai_compatible_endpoint ?? undefined,
    openai_compatible_model: judgeOverride ?? job.openai_compatible_model ?? undefined,
    local_model: job.local_model ?? undefined,
    prompt_template_id: job.prompt_template_id ?? undefined,
    max_tokens: job.max_tokens,
    api_timeout: job.api_timeout ?? undefined,
    filter_special: job.filter_special,
    filter_single_char: job.filter_single_char,
    filter_punctuation: job.filter_punctuation,
    filter_numbers: job.filter_numbers,
    filter_fragments: job.filter_fragments,
    filter_stop_words: job.filter_stop_words,
  });

  /*
   * NEVER ONE CLICK. Sweeping L46 is about 59 GPU-hours, and the operator must
   * see that number and choose the ceiling before anything starts. The count is
   * derived from the coverage we already hold, not from a guess.
   */
  const handleStartSweep = async () => {
    if (!coverage) return;
    setIsResuming(true);
    try {
      const started = await labelingAPI.startResumeSweep(
        job.extraction_job_id,
        batchesRemaining,
        {
          ...buildConfig(),
          // `batchesRemaining` was computed from a coverage read carrying these.
          // A sweep that selects without them runs a different set than the one
          // the operator was quoted — and the quote can be 20x too high.
          prompt_fingerprint: coveragePredicate?.promptFingerprint,
          judge_model: coveragePredicate?.judgeModel,
        },
      );
      setSweep(started);
      setSweepPrompt(false);
    } catch {
      // The store surfaces the message; the button simply re-enables.
    } finally {
      setIsResuming(false);
    }
  };

  const handleStopSweep = async () => {
    if (!sweep) return;
    // Cooperative: the batch in flight finishes and nothing more is enqueued.
    setSweep(await labelingAPI.cancelResumeSweep(sweep.id));
  };

  const handleResume = async (
    options: { limit?: number; only?: 'failed' | 'pending' } = {},
  ) => {
    if (!coverage || coverage.remaining === 0) return;
    setIsResuming(true);
    try {
      // ONE config builder, shared with the sweep. There were two, and a
      // judge override added to one would have silently not applied to the
      // other — the same drift that made `retryLabeling` drop the endpoint,
      // the template and every filter flag while looking like it worked.
      /*
       * THE ACTION MUST USE THE PREDICATE THE COUNT WAS COMPUTED WITH.
       *
       * `coverage` above was fetched with the template's fingerprint and the
       * judge, so its `remaining` — the number on this button — includes
       * verdicts that are stale for this template. Resuming without them took a
       * strictly smaller set, so the button could offer a number it would never
       * deliver. That is the "Resume 0 of 39" / "Nothing left to label" defect,
       * reintroduced from outside the predicate that was fixed to prevent it.
       */
      await resumeLabeling(job.extraction_job_id, buildConfig(), {
        ...options,
        promptFingerprint: coveragePredicate?.promptFingerprint,
        judgeModel: coveragePredicate?.judgeModel,
      });
    } catch {
      // resumeLabeling already records the message on the store.
    } finally {
      setIsResuming(false);
    }
  };

  // Fetch template name if template ID is present
  useEffect(() => {
    if (job.prompt_template_id) {
      // First check if we already have it in the store
      const existingTemplate = templates.find(t => t.id === job.prompt_template_id);
      if (existingTemplate) {
        setTemplateName(existingTemplate.name);
      } else {
        // Fetch from API if not in store
        fetchTemplate(job.prompt_template_id)
          .then(() => {
            const template = templates.find(t => t.id === job.prompt_template_id);
            if (template) {
              setTemplateName(template.name);
            }
          })
          .catch(() => {
            setTemplateName('Unknown Template');
          });
      }
    }
  }, [job.prompt_template_id, templates]);

  // Calculate progress percentage
  const totalFeatures = job.total_features || 0;
  const progress = totalFeatures > 0
    ? (job.features_labeled / totalFeatures) * 100
    : 0;

  // Get elapsed time - compute from total seconds to avoid date-fns calendar decomposition issues
  const getElapsedTime = () => {
    const start = new Date(job.created_at);
    const end = job.completed_at ? new Date(job.completed_at) : new Date();

    let totalSeconds = Math.floor((end.getTime() - start.getTime()) / 1000);
    if (totalSeconds < 0) totalSeconds = 0;

    const days = Math.floor(totalSeconds / 86400);
    const hours = Math.floor((totalSeconds % 86400) / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;

    const parts = [];
    if (days) parts.push(`${days}d`);
    if (hours) parts.push(`${hours}h`);
    if (minutes) parts.push(`${minutes}m`);
    if (seconds) parts.push(`${seconds}s`);

    return parts.length > 0 ? parts.join(' ') : '0s';
  };

  // Get estimated time remaining based on current rate
  const getEstimatedTimeRemaining = (): string | null => {
    // Need at least some progress to estimate
    if (job.features_labeled <= 0 || totalFeatures <= 0) {
      return null;
    }

    const start = new Date(job.created_at);
    const now = new Date();
    const elapsedSeconds = (now.getTime() - start.getTime()) / 1000;

    // Need at least 30 seconds of data for a reasonable estimate
    if (elapsedSeconds < 30) {
      return null;
    }

    const rate = job.features_labeled / elapsedSeconds; // features per second
    const remaining = totalFeatures - job.features_labeled;
    const etaSeconds = remaining / rate;

    // Convert to hours, minutes, seconds
    const hours = Math.floor(etaSeconds / 3600);
    const minutes = Math.floor((etaSeconds % 3600) / 60);

    const parts = [];
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0 || hours > 0) parts.push(`${minutes}m`);

    return parts.length > 0 ? `~${parts.join(' ')}` : '<1m';
  };

  // Status icon and color
  const getStatusDisplay = () => {
    switch (job.status) {
      case LabelingStatus.QUEUED:
        return {
          icon: <Clock className="w-5 h-5" />,
          color: 'text-yellow-400',
          bg: 'bg-yellow-500/10',
          border: 'border-yellow-500/30',
          label: 'Queued',
        };
      case LabelingStatus.LABELING:
        return {
          icon: <Loader className="w-5 h-5 animate-spin" />,
          color: 'text-blue-400',
          bg: 'bg-blue-500/10',
          border: 'border-blue-500/30',
          label: 'Labeling',
        };
      case LabelingStatus.COMPLETED:
        return {
          icon: <CheckCircle className="w-5 h-5" />,
          color: 'text-emerald-400',
          bg: 'bg-emerald-500/10',
          border: 'border-emerald-500/30',
          label: 'Completed',
        };
      case LabelingStatus.FAILED:
        return {
          icon: <XCircle className="w-5 h-5" />,
          color: 'text-red-400',
          bg: 'bg-red-500/10',
          border: 'border-red-500/30',
          label: 'Failed',
        };
      case LabelingStatus.CANCELLED:
        return {
          icon: <Ban className="w-5 h-5" />,
          color: 'text-slate-600 dark:text-slate-400',
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/30',
          label: 'Cancelled',
        };
      default:
        return {
          icon: <Tag className="w-5 h-5" />,
          color: 'text-slate-600 dark:text-slate-400',
          bg: 'bg-slate-500/10',
          border: 'border-slate-500/30',
          label: 'Unknown',
        };
    }
  };

  const statusDisplay = getStatusDisplay();

  return (
    <div className={`${COMPONENTS.card.base} px-4 py-3 border ${statusDisplay.border}`}>
      {/* Compact Header: Status, Title, Info, and Actions - Single Row */}
      <div className={`flex items-center justify-between ${isActive ? 'mb-2' : 'mb-1'}`}>
        {/* Left: Status Icon + Title + Key Info */}
        <div className="flex items-center gap-3 flex-1">
          <div className={`p-1.5 rounded-lg ${statusDisplay.bg} ${statusDisplay.color}`}>
            {statusDisplay.icon}
          </div>
          <div className="flex items-center gap-4 flex-1">
            <span className={`text-sm font-semibold ${COMPONENTS.text.primary}`}>
              {job.model_name || 'Unknown Model'}
            </span>
            {job.layer_index != null && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300">
                L{job.layer_index}
              </span>
            )}
            {job.hook_type && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300">
                {job.hook_type}
              </span>
            )}
            {job.sae_name && (
              <span className="text-xs px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
                {job.sae_name}
              </span>
            )}
            <span className={`text-xs px-2 py-0.5 rounded ${statusDisplay.bg} ${statusDisplay.color}`}>
              {statusDisplay.label}
            </span>
            <span className={`text-xs ${COMPONENTS.text.secondary}`}>
              {job.labeling_method === 'openai'
                ? `OpenAI (${job.openai_model || 'gpt-4o-mini'})`
                : job.labeling_method === 'openai_compatible'
                ? job.openai_compatible_model || 'Ollama'
                : `Local (${job.local_model || 'meta-llama/Llama-3.2-1B'})`}
            </span>
            <span className={`text-xs ${COMPONENTS.text.secondary}`}>
              Started: {format(new Date(job.created_at), 'MMM d, h:mm a')}
            </span>
            {isActive && (
              <>
                <span className="text-xs text-blue-400 font-medium">
                  Elapsed: {getElapsedTime()}
                </span>
                <span className={`text-xs ${COMPONENTS.text.secondary}`}>
                  {job.features_labeled.toLocaleString()} / {totalFeatures.toLocaleString()}
                </span>
                <span className="text-xs text-emerald-400 font-medium">
                  {progress.toFixed(1)}%
                </span>
                {getEstimatedTimeRemaining() && (
                  <span className="text-xs text-amber-400 font-medium">
                    ETA: {getEstimatedTimeRemaining()}
                  </span>
                )}
              </>
            )}
            {isCompleted && job.completed_at && (
              <span className="text-xs text-emerald-400 font-medium">
                Completed in {getElapsedTime()}
              </span>
            )}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {isActive && onCancel && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to cancel this labeling job?')) {
                  onCancel();
                }
              }}
              className={`p-1.5 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Cancel labeling"
            >
              <XCircle className="w-4 h-4" />
            </button>
          )}
          {/*
            * THE JUDGE A RESUME WILL USE.
            *
            * Shown whenever there is work left and the endpoint can be
            * enumerated. Highlighted when the original judge is GONE — the case
            * this exists for, where a plain resume cannot succeed no matter how
            * many times it is clicked.
            */}
          {isFinished && coverage && coverage.remaining > 0
            && judges && judges.reachable && judges.models.length > 0 && (
            <label
              className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs border
                ${judges.original_available
                  ? 'border-slate-300/40 dark:border-slate-600/40 text-slate-600 dark:text-slate-400'
                  : 'border-amber-500/40 bg-amber-500/5 text-amber-700 dark:text-amber-400'}`}
              title={
                judges.original_available
                  ? 'The model this resume will label with. Defaults to the one this '
                    + 'job used, so both halves are comparable.'
                  : `This job used "${judges.original_model}", which ${judges.endpoint} `
                    + 'no longer serves — every feature would fail. Pick a model that is '
                    + 'available. Each feature records which judge labelled it.'
              }
            >
              <span className="whitespace-nowrap">
                {judges.original_available ? 'Judge' : 'Original judge unavailable —'}
              </span>
              <select
                aria-label="Model to label with"
                value={judgeOverride ?? judges.original_model ?? ''}
                onChange={(e) => setJudgeOverride(e.target.value)}
                className="bg-transparent border-none focus:outline-none max-w-[16rem] truncate"
              >
                {judges.original_available && judges.original_model && (
                  <option value={judges.original_model}>
                    {judges.original_model} (original)
                  </option>
                )}
                {judges.models
                  .filter((m) => !(judges.original_available && m === judges.original_model))
                  .map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
              </select>
            </label>
          )}

          {/*
            * WHY THERE IS NO RESUME BUTTON.
            *
            * `outstanding` counts features that still lack a verdict;
            * `remaining` counts what a resume would actually take. They differ
            * when failures have used up their retries, and a card showing
            * "39 failed" with no button is arithmetic an operator cannot
            * reconcile in silence. Reported live as "Resume 0 of 39", which
            * then answered "Nothing left to label".
            */}
          {isFinished && coverage && coverage.remaining === 0
            && coverage.exhausted > 0 && (
            <span
              className="text-xs text-amber-700 dark:text-amber-400"
              title={
                `${coverage.exhausted.toLocaleString()} features still have no label, `
                + 'but each has already been attempted the maximum number of times. '
                + 'Resume will not retry them. Look at the failure reasons above — '
                + 'if the cause has since been fixed, they can be retried by raising '
                + 'the attempt limit.'
              }
            >
              {coverage.exhausted.toLocaleString()} exhausted retries
            </span>
          )}

          {/*
            * RESUME ALL — and never in one click.
            *
            * Sweeping L46 is ~27 batches, about 59 GPU-hours. A confirmation
            * that states the cost in those terms is the difference between a
            * decision and an accident, so the button opens a prompt rather than
            * starting work.
            *
            * A running sweep replaces both with its own progress and a Stop.
            */}
          {isFinished && sweep && sweep.status === 'running' && (
            <div className="flex items-center gap-2">
              <span
                className="text-xs text-sky-700 dark:text-sky-400 tabular-nums"
                /*
                 * `?? 0` is not defensive noise. This card renders inside a
                 * list; an exception here unmounts the WHOLE labeling panel, so
                 * one absent counter would cost the operator every other job on
                 * the page. A missing number should read as zero, not as a
                 * blank screen.
                 */
                title={
                  `${(sweep.features_labeled ?? 0).toLocaleString()} labeled, `
                  + `${(sweep.features_failed ?? 0).toLocaleString()} failed so far. `
                  + 'Counted from what each batch actually wrote.'
                }
              >
                Sweeping — batch {(sweep.batches_done ?? 0) + 1} of {sweep.max_batches ?? '?'}
              </span>
              <button
                type="button"
                onClick={handleStopSweep}
                disabled={sweep.cancel_requested_at !== null}
                aria-label="Stop the sweep after the current batch"
                className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                  border transition-all duration-200 ${
                  sweep.cancel_requested_at
                    ? 'border-transparent bg-slate-100 dark:bg-slate-800 text-slate-500 cursor-not-allowed'
                    : 'border-rose-500/25 bg-rose-500/5 text-rose-700 dark:text-rose-400 hover:bg-rose-500/15'
                }`}
                title="The batch in flight finishes; nothing further is queued."
              >
                <Square className="w-3.5 h-3.5" />
                <span>{sweep.cancel_requested_at ? 'Stopping\u2026' : 'Stop'}</span>
              </button>
            </div>
          )}

          {isFinished && !sweep && coverage && batchesRemaining > 1 && sweepPrompt && (
            <div className="flex items-center gap-2 px-2.5 py-1.5 rounded-lg
                            border border-sky-500/30 bg-sky-500/5">
              <span className="text-xs text-sky-800 dark:text-sky-300">
                {batchesRemaining} batches, about{' '}
                {Math.round((coverage.remaining * 8) / 3600)} GPU-hours. Continue?
              </span>
              <button
                type="button"
                onClick={handleStartSweep}
                disabled={isResuming}
                className="text-xs font-medium text-sky-700 dark:text-sky-400 hover:underline"
              >
                Start sweep
              </button>
              <button
                type="button"
                onClick={() => setSweepPrompt(false)}
                className="text-xs text-slate-600 dark:text-slate-400 hover:underline"
              >
                Cancel
              </button>
            </div>
          )}

          {isFinished && !sweep && !sweepPrompt && coverage && batchesRemaining > 1 && (
            <button
              type="button"
              onClick={() => setSweepPrompt(true)}
              disabled={isResuming}
              aria-label={`Resume all ${coverage.remaining} remaining features across ${batchesRemaining} batches`}
              className={`group flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                border border-sky-500/25 bg-sky-500/5 text-sky-700 dark:text-sky-400
                hover:border-sky-500/50 hover:bg-sky-500/15 transition-all duration-200`}
              title={
                `Runs ${batchesRemaining} batches of ${batchSize.toLocaleString()} back to back, `
                + 'stopping when the extraction is done or the ceiling is reached. '
                + 'Asks for confirmation first — this is hours of GPU time.'
              }
            >
              <FastForward className="w-3.5 h-3.5" />
              <span>Resume all ({batchesRemaining} batches)</span>
            </button>
          )}

          {/*
            * TRY A SAMPLE FIRST.
            *
            * Retrying L46's failures is ~32 GPU-hours. Twenty of them is about
            * three minutes and answers the only question that matters: do these
            * still fail? The reason breakdown says WHAT went wrong; this says
            * whether it is still going wrong.
            *
            * Drawn from `failed` alone. A mixed sample answers neither question:
            * 15 fresh successes and 5 repeat failures reads as 75%, which is
            * true of nothing.
            */}
          {isFinished && coverage && (coverage.by_status.failed ?? 0) > 0
            && coverage.remaining > 0 && (
            <button
              type="button"
              onClick={() => handleResume({ limit: SAMPLE_SIZE, only: 'failed' })}
              disabled={isResuming}
              aria-label={
                `Retry ${SAMPLE_SIZE} failed features as a sample before `
                + `committing to all ${coverage.by_status.failed ?? 0}`
              }
              className={`group flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                border transition-all duration-200 ${
                isResuming
                  ? 'border-transparent bg-slate-100 dark:bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'border-amber-500/25 bg-amber-500/5 text-amber-700 dark:text-amber-400 '
                    + 'hover:border-amber-500/50 hover:bg-amber-500/15'
              }`}
              title={
                `Retries ${SAMPLE_SIZE} of the `
                + `${(coverage.by_status.failed ?? 0).toLocaleString()} failed features, `
                + 'with this job\u2019s own model and template. Roughly three minutes, '
                + 'against about '
                + `${Math.round(((coverage.by_status.failed ?? 0) * 8) / 3600)} GPU-hours `
                + 'for all of them. Check the coverage strip afterwards to see how many stuck.'
              }
            >
              <FlaskConical className="w-3.5 h-3.5" />
              <span>Try {SAMPLE_SIZE} first</span>
            </button>
          )}
          {/*
            * Resume: label exactly what this job did not.
            *
            * Labelled with the COUNT, so a click is never a blind bulk action —
            * the failure this replaces is a "Retry" that silently restarted a
            * 53,000-feature extraction from step zero.
            *
            * Rendered only when coverage says work remains, so a finished job
            * with nothing outstanding shows no button rather than a disabled one
            * that invites a click and does nothing.
            */}
          {isFinished && coverage && coverage.remaining > 0 && (
            <button
              type="button"
              onClick={() => handleResume()}
              disabled={isResuming}
              aria-label={
                `Resume labeling: labels ${batchSize} features now, `
                + `${coverage.remaining} still need a label`
              }
              className={`group flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium
                border transition-all duration-200 ${
                isResuming
                  ? 'border-transparent bg-slate-100 dark:bg-slate-800 text-slate-500 cursor-not-allowed'
                  : 'border-emerald-500/25 bg-emerald-500/5 text-emerald-700 dark:text-emerald-400 '
                    + 'hover:border-emerald-500/50 hover:bg-emerald-500/15 hover:shadow-sm hover:shadow-emerald-500/20'
              }`}
              title={
                `Labels only the ${coverage.remaining.toLocaleString()} features that still `
                + `need one (${(coverage.by_status.failed ?? 0).toLocaleString()} failed, `
                + `${(coverage.by_status.pending ?? 0).toLocaleString()} never attempted). `
                + `The ${coverage.adjudicated.toLocaleString()} already adjudicated are left `
                + 'untouched, including features the judge honestly found uninterpretable. '
                + 'Runs with this job\u2019s own model and template. '
                + (batchesRemaining > 1
                    ? `One batch of ${batchSize.toLocaleString()} at a time \u2014 about `
                      + `${batchesRemaining} more runs, not one button.`
                    : 'This finishes the extraction.')
              }
            >
              {isResuming ? (
                <Loader className="w-3.5 h-3.5 animate-spin" />
              ) : (
                <PlayCircle className="w-3.5 h-3.5" />
              )}
              <span>
                {isResuming ? 'Resuming\u2026' : resumeLabel}
              </span>
            </button>
          )}
          {(isCompleted || isFailed || isCancelled) && onDelete && (
            <button
              type="button"
              onClick={() => {
                if (window.confirm('Are you sure you want to delete this labeling job?')) {
                  onDelete();
                }
              }}
              className={`p-1.5 rounded-lg ${COMPONENTS.button.ghost}`}
              title="Delete labeling job"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Progress Bar for Active Jobs */}
      {isActive && (
        <div className="mb-3">
          <div className={`h-1.5 ${COMPONENTS.card.base} rounded-full overflow-hidden`}>
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Real-time Results Window - Full Width at Bottom */}
      {isActive && (
        <div>
          <LabelingResultsWindow labelingJobId={job.id} />
        </div>
      )}

      {/* Compact details for completed/failed/cancelled jobs */}
      {!isActive && (
        <div className={`text-xs ${COMPONENTS.text.secondary} space-y-0.5`}>
          <p>
            {templateName}
            {' · '}
            {job.features_labeled.toLocaleString()}/{totalFeatures.toLocaleString()} features
            {isCompleted && job.statistics && (
              <> ({job.statistics.failed_labels.toLocaleString()} failed)</>
            )}
            {' · '}
            {format(new Date(job.created_at), 'MMM d h:mm a')}
            {job.completed_at && (
              <> → {format(new Date(job.completed_at), 'MMM d h:mm a')}</>
            )}
            {job.completed_at && (
              <span className="text-emerald-400 font-medium"> ({getElapsedTime()})</span>
            )}
          </p>
        </div>
      )}

      {/* Error Message */}
      {isFailed && job.error_message && (
        <div className="mt-4 p-3 bg-red-900/20 border border-red-800 rounded-lg">
          <p className="text-sm text-red-200">
            <span className="font-medium">Error:</span> {job.error_message}
          </p>
        </div>
      )}
    </div>
  );
};

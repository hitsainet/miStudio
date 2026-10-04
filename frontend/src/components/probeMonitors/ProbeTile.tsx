/**
 * One trained probe, in the Probes tab.
 *
 * ⚠ WHAT THIS FIXES. The row used to read `L11 · mean · dense` and a threshold, which does not
 * identify a probe. Nine of them said `L11 · mean · dense` on this installation — same layer, same
 * rule, same basis, different runs, different corpora, different quality — and nothing on the row
 * told them apart or said which was any good. A reader could not answer "which of these detects
 * what, and how well" without opening each one in turn.
 *
 * So the tile now carries the four questions someone actually has:
 *
 *   what does it detect   the training view it was fitted on, and the model it reads
 *   where does it read    layer, hook point, and the basis (dense residual, or an SAE's k features)
 *   how good is it        the rung IN THE SERVER'S WORDS, plus validation AUROC
 *   what state is it in   selected, exported, published, streamable, threshold placed
 *
 * ⚠ NO RUNG WORDING IS COMPOSED HERE. `rung_language` arrives from the server — see `RungChip`,
 * and the note on `summary_without_curve`, which was changed to put the phrase on the LIST for
 * exactly this tile. A number-to-phrase map in the frontend is a second vocabulary that drifts
 * from miLLM's, and the drifted one is what a person reads.
 */
import type { ReactNode } from 'react';
import { ChevronDown, ChevronRight } from 'lucide-react';

import type { ProbeMonitorSummary } from '../../types/probeMonitor';

interface ProbeTileProps {
  probe: ProbeMonitorSummary;
  /** Resolved names, so a tile never shows a bare `m_…` / `pmd_…` id when a name exists. */
  modelName?: string | null;
  trainViewName?: string | null;
  /**
   * The separation the probe was fitted to make, as the corpus's own label values —
   * `"high-stakes vs low-stakes"`. `null` when the training view's mapping does not state
   * both sides; see `labelSeparation`, which refuses to print half a contrast.
   */
  labelSeparation?: string | null;
  expanded: boolean;
  onToggle: () => void;
  /** The report, rendered INSIDE this tile when expanded. */
  children?: ReactNode;
}

/** A badge's colour is decoration; its `title` is the explanation. Both are per badge. */
function Badge({
  tone,
  title,
  children,
  testId,
}: {
  tone: string;
  title: string;
  children: ReactNode;
  testId?: string;
}) {
  return (
    <span
      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${tone}`}
      title={title}
      data-testid={testId}
    >
      {children}
    </span>
  );
}

const RUNG_TONE: Record<number, string> = {
  0: 'bg-slate-700/60 text-slate-300',
  1: 'bg-sky-900/50 text-sky-300',
  2: 'bg-emerald-900/50 text-emerald-300',
  3: 'bg-violet-900/50 text-violet-300',
};

/**
 * When a probe was first trained, short and in the VIEWER'S timezone.
 *
 * ⚠ LOCALISED BY THE BROWSER, NOT FORMATTED BY HAND. The row is stored UTC; `toLocaleString`
 * renders it in whatever zone the reader is in, which is the only correct answer for a timestamp
 * a human reads. Hardcoding an offset would be right for one operator and wrong for a log.
 *
 * ⚠ AND IT RETURNS null RATHER THAN "Invalid Date". An unparseable or absent value must render
 * NOTHING: a tile that says `trained Invalid Date` is worse than one that says nothing, and this
 * estate has shipped a fabricated duration beside "Finished —" for exactly that reason.
 */
function trainedOn(value: string | null | undefined): string | null {
  if (!value) return null;
  const when = new Date(value);
  if (Number.isNaN(when.getTime())) return null;
  return when.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

/**
 * What a TRAINING scope reads, in words. Keyed by the internal scope name the server sends.
 *
 * ⚠ WHY THIS LINE EXISTS (operator, 2026-10-04). A run's probes are layers x rules, every one
 * trained on the SAME scope and carrying a bar for EACH window. A tile that showed neither let a
 * reader take nine probes for "one per window per layer" — and decide whether to retrain on that.
 */
export const SCOPE_WORDS: Record<string, string> = {
  all: 'every token (prompt and reply)',
  user: "the person's turns only",
  input: 'everything before the reply',
  assistant: "the model's turns only",
  last_assistant: "the model's final reply",
};

/** Contract windows in the order a reader expects them. */
export const WINDOW_ORDER = ['prompt', 'response', 'all'] as const;

const WINDOW_TITLES: Record<string, string> = {
  prompt: "The person's side of the request: everything before the model's reply (system text and earlier turns included). This bar was cut from calibration negatives read over exactly that span.",
  response: "The model's reply. This bar was cut from reply negatives, but the probe's weights were fitted on prose read as a person's turn and have never seen a model reply — so a verdict here is a ranking, not a measured rate, and miLLM reports it as provisional.",
  all: 'The whole request, prompt and reply together. The same span the probe was trained on when its scope is `all`.',
};

/** Whether the weights were fitted on model replies — only a reply-scoped probe was. */
function trainedOnReplies(scope: string | null | undefined): boolean {
  return scope === 'last_assistant' || scope === 'assistant';
}

export function ProbeTile({
  probe,
  modelName,
  trainViewName,
  labelSeparation,
  expanded,
  onToggle,
  children,
}: ProbeTileProps) {
  const valAuroc =
    typeof probe.val_metrics?.val_auroc === 'number'
      ? (probe.val_metrics.val_auroc as number)
      : null;
  const featureCount = probe.sae_feature_indices?.length ?? null;
  const publications = probe.published?.length ?? 0;
  const invalidated = Boolean(
    (probe.definition_build as { invalidated?: unknown } | null | undefined)?.invalidated
  );

  return (
    <li
      className={`rounded border ${
        expanded ? 'border-emerald-700 bg-slate-900/80' : 'border-slate-700 bg-slate-900/60'
      }`}
      data-testid="probe-row"
      data-expanded={expanded ? 'true' : 'false'}
    >
      <div className="flex items-start justify-between gap-3 p-3">
        <div className="min-w-0 flex-1">
          {/* line 1 — WHAT IT DETECTS, which is the question a list of nine `L11 · mean · dense`
              rows could not answer at all. */}
          <p className="truncate text-sm text-slate-200" data-testid="probe-concept">
            {trainViewName ? (
              <>
                detects <span className="font-medium text-slate-100">{trainViewName}</span>
              </>
            ) : (
              <span className="text-slate-400">concept not resolved</span>
            )}
            {modelName ? <span className="text-slate-400"> in {modelName}</span> : null}
          </p>

          {/* line 1b — WHICH LABELS. `detects high-stakes` names the positive side only, and a
              linear probe is defined by a contrast: the same positive label against a different
              negative one is a different detector. These are the corpus's own label values. */}
          {labelSeparation ? (
            <p
              className="mt-0.5 truncate text-xs text-slate-400"
              title="The training view's label mapping, verbatim: the values on the left were mapped to `positive` and the values on the right to `negative`. The probe is a boundary between these two sets and nothing else — it says nothing about any label it was never shown."
              data-testid="probe-labels"
            >
              trained on{' '}
              <span className="font-mono text-slate-300">{labelSeparation}</span>
            </p>
          ) : null}

          {/* line 2 — WHERE IT READS and HOW IT POOLS */}
          <p className="mt-0.5 font-mono text-xs text-slate-400" data-testid="probe-readout">
            L{probe.layer} · resid_post · {probe.rule}
            {/* The rolling window is part of the rule: two `rolling_mean_max` probes at one layer
                differ ONLY here, and without it they read identically. */}
            {typeof probe.rule_params?.window === 'number' ? ` w=${probe.rule_params.window}` : ''}
            {probe.variant === 'sae'
              ? ` · SAE basis${featureCount !== null ? ` (${featureCount} features)` : ''}`
              : ' · dense residual'}
          </p>

          {/* line 2b — WHICH TOKENS: what it was trained on, and its bar over each window. */}
          <div className="mt-1 text-xs text-slate-400" data-testid="probe-windows">
            <span data-testid="probe-scope">
              fitted on{' '}
              {probe.scope ? (
                <span
                  className="text-slate-300"
                  title={`Training scope \`${probe.scope}\`: the tokens every one of this run's probes was fitted and calibrated on. A window below is a different slice of the SAME probe, not a different probe.`}
                >
                  {SCOPE_WORDS[probe.scope] ?? probe.scope}
                </span>
              ) : (
                <span className="text-amber-300" title="The run that trained this probe is gone, so its scope cannot be read.">
                  scope not recorded
                </span>
              )}
            </span>
            <div className="mt-1 flex flex-wrap items-center gap-1.5">
              {Object.keys(probe.window_thresholds ?? {}).length === 0 ? (
                <Badge
                  tone="bg-slate-800 text-slate-300"
                  title="No per-window bar was placed (the run had no calibration set), so every window — prompt, response and all — is judged against this one threshold."
                  testId="window-single-bar"
                >
                  one bar for every window
                </Badge>
              ) : (
                WINDOW_ORDER.filter((w) => w in (probe.window_thresholds ?? {})).map((window) => {
                  const bar = probe.window_thresholds?.[window];
                  const untrained = window === 'response' && !trainedOnReplies(probe.scope);
                  return (
                    <Badge
                      key={window}
                      tone={untrained ? 'bg-amber-900/40 text-amber-200' : 'bg-slate-800 text-slate-200'}
                      title={WINDOW_TITLES[window] ?? window}
                      testId={`window-${window}`}
                    >
                      {window} {typeof bar === 'number' ? `≥ ${bar.toFixed(2)}` : 'no bar'}
                      {untrained ? ' · provisional' : ''}
                    </Badge>
                  );
                })
              )}
              {probe.length_band_count ? (
                <span
                  className="text-[10px] text-slate-500"
                  title="The bar over the probe's OWN scope also varies with how many tokens were scored. Only that window uses the bands; every other window is judged against its own bar."
                  data-testid="window-length-bands"
                >
                  + {probe.length_band_count} length bands on {probe.scope === 'all' ? 'all' : 'its own scope'}
                </span>
              ) : null}
            </div>
          </div>

          {/* line 3 — badges. Every one carries its meaning in `title`. */}
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            <Badge
              tone={RUNG_TONE[probe.rung] ?? RUNG_TONE[0]}
              title={
                probe.rung_next_step
                  ? `Evidence rung ${probe.rung}. Next: ${probe.rung_next_step}`
                  : `Evidence rung ${probe.rung}`
              }
              testId="badge-rung"
            >
              {/* The phrase is the SERVER'S. */}
              {probe.rung_language || `rung ${probe.rung}`}
              <span className="ml-1 opacity-70">rung {probe.rung}</span>
            </Badge>

            {probe.selected ? (
              <Badge
                tone="bg-emerald-900/50 text-emerald-300"
                title="The run trained a probe at every layer and rule you asked for, and this is the one it chose — the highest validation AUROC of that sweep. It is the probe the run's own artifacts refer to. It does NOT mean the probe is good, only that it beat its siblings."
                testId="badge-selected"
              >
                selected
              </Badge>
            ) : null}

            {valAuroc !== null ? (
              <Badge
                tone="bg-slate-800 text-slate-300"
                title="Validation AUROC — measured IN-DISTRIBUTION, on a held-out split of the training view. It is not evidence the probe generalises; the report's per-set table is."
                testId="badge-val"
              >
                val {valAuroc.toFixed(3)}
              </Badge>
            ) : null}

            {probe.threshold === null ? (
              <Badge
                tone="bg-amber-900/50 text-amber-300"
                title="No threshold was placed at all — this is not a threshold of zero. The probe produces scores but cannot produce a verdict."
                testId="badge-no-threshold"
              >
                no threshold
              </Badge>
            ) : null}

            {!probe.streamable ? (
              <Badge
                tone="bg-amber-900/40 text-amber-200"
                title="This rule needs the whole sequence before it can score, so it cannot run token-by-token on a live stream. `mean` and `last` can; `attention` and `softmax` cannot."
                testId="badge-not-streamable"
              >
                not streamable
              </Badge>
            ) : null}

            {probe.definition_built_at && !invalidated ? (
              <Badge
                tone="bg-sky-900/50 text-sky-300"
                title="A portable definition has been built for this probe and is current."
                testId="badge-exported"
              >
                definition built
              </Badge>
            ) : null}

            {invalidated ? (
              <Badge
                tone="bg-amber-900/50 text-amber-300"
                title="A definition was built and then invalidated, because something it states changed — a new evaluation, a new judge run, or a recalibrated threshold. Rebuild before handing it to anyone."
                testId="badge-stale"
              >
                definition stale
              </Badge>
            ) : null}

            {publications > 0 ? (
              <Badge
                tone="bg-violet-900/50 text-violet-300"
                title={`Published to HuggingFace ${publications} time(s). Publications are appended, never replaced.`}
                testId="badge-published"
              >
                published{publications > 1 ? ` ×${publications}` : ''}
              </Badge>
            ) : null}
          </div>

          {/* line 4 — the operating point, unchanged in substance */}
          <p className="mt-1 text-xs text-slate-500" data-testid="probe-operating-point">
            {probe.threshold === null ? (
              <span className="text-amber-300">no threshold placed</span>
            ) : (
              <>
                threshold {probe.threshold.toFixed(4)} · spends{' '}
                {probe.realised_fpr !== null
                  ? `${(probe.realised_fpr * 100).toFixed(1)}%`
                  : '—'}
                {probe.threshold_source ? ` (${probe.threshold_source})` : ''}
              </>
            )}
            <span className="ml-2 opacity-70">from {probe.run_id}</span>
            {/* ⚠ WHEN THIS PROBE WAS FIRST TRAINED, beside the run that trained it.
                `created_at` has been on the payload and on this interface all along and was
                rendered nowhere — so the tile could say which run produced a probe and never
                when, and two probes from different weeks were indistinguishable at a glance.
                It matters more now that a THRESHOLD can move: the bar carries its own revision
                and history, and the weights carry this. */}
            {trainedOn(probe.created_at) ? (
              <span
                className="ml-2 opacity-70"
                data-testid="probe-trained-at"
                title={`First trained ${new Date(probe.created_at).toString()}. The weights have not changed since; a moved threshold is recorded separately as a revision.`}
              >
                · trained {trainedOn(probe.created_at)}
              </span>
            ) : null}
          </p>
        </div>

        <button
          type="button"
          onClick={onToggle}
          aria-expanded={expanded}
          className="flex shrink-0 items-center gap-1 rounded border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
          data-testid="probe-report-toggle"
        >
          {expanded ? (
            <ChevronDown className="h-3 w-3" aria-hidden="true" />
          ) : (
            <ChevronRight className="h-3 w-3" aria-hidden="true" />
          )}
          Report
        </button>
      </div>

      {/*
        ⚠ THE REPORT RENDERS HERE, INSIDE THE TILE. It used to render after the whole `<ul>`, so
        opening the report of the third probe in a list of thirteen scrolled the reader to the
        bottom of the page and showed a panel with no visible connection to the row they clicked.
        With nine rows reading `L11 · mean · dense`, there was no way to tell which one it belonged
        to.
      */}
      {expanded && children ? (
        <div className="border-t border-slate-700/70 px-3 pb-3 pt-3" data-testid="probe-report-body">
          {children}
        </div>
      ) : null}
    </li>
  );
}

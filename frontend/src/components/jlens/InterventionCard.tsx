/**
 * Run an intervention on a pinned token, with its matched control.
 *
 * WHY IT LIVES HERE. The intervention endpoint takes a d_model direction, which
 * no browser can produce — that requirement is precisely why this capability
 * shipped with no UI at all. The server now resolves a direction from a SINGLE
 * TOKEN's unembedding row, and a token is the one thing this panel always has
 * on screen. So the affordance is "intervene along this pinned token".
 *
 * THE CONTROL IS NOT OPTIONAL (BR-018). There is no checkbox to skip it: `k`
 * and `control_seed` are always sent, and the result reports all three arms —
 * baseline, intervened and control — so a reader can see the control actually
 * ran rather than taking it on trust. The SEPARATION is the finding; the
 * intervened rate alone is not one.
 *
 * RUNG 2, AND THIS DOCSTRING ONCE SAID OTHERWISE. It described a lens-space
 * displacement, which is what this measured before the causal rewrite —
 * `intervened_outcome`, `control_outcome` and `excess_over_control` are all
 * from that shape and none of them exist any more. What runs now perturbs
 * inside the model's own forward pass, lets it continue, and scores the target
 * token's RANK in the model's real output.
 *
 * MANY PROMPTS, NOT ONE. Below four trials NO outcome separates the intervened
 * and control intervals — a perfect intervened arm against a perfect null
 * control still overlaps — so a one-prompt run can only ever report "no effect
 * demonstrated", which reads as a fact about the direction and is a fact about
 * the sample size. This card is the only surface that can supply more, which
 * is why it has a trial-prompt list.
 */

import { useRef, useState } from 'react';
import { Loader2, Zap } from 'lucide-react';
import { jlensApi } from '../../api/jlens';
import type { JLensTokenCheck } from '../../types/jlens';
import { getTaskStatus } from '../../api/models';

const POLL_MS = 4000;

/**
 * Fewest trials at which disjoint Wilson intervals are arithmetically possible.
 *
 * MIRRORS `MIN_TRIALS_FOR_SEPARATION` in `services/jlens_causal.py`, where it is
 * derived rather than chosen. Duplicated here only to warn BEFORE a GPU job is
 * queued; the authority is the server, which reports
 * `min_trials_for_separation` with every result and is what the verdict above
 * renders.
 */
const MIN_TRIALS = 4;

/**
 * The primitives runnable from here (BR-017). Recorded with every result.
 *
 * `coordinate_swap` BELONGS HERE and was missing, which made a swap runnable
 * but not demonstrable anywhere in the product: the ranked list can launch one
 * because it has a pinned partner to hand, but it sends a single prompt — and
 * below four trials no outcome separates, so that path can only ever report
 * "not attainable". This card is the only surface that can supply trials, and
 * it offered neither the swap nor a way to name its partner.
 *
 * `dynamic_topk_ablation` is deliberately absent: the server refuses it, and
 * offering a control that always 422s is worse than not offering it.
 */
export const PRIMITIVES = [
  { id: 'additive', label: 'Additive', hint: 'Steer along the direction' },
  {
    id: 'projective_ablation',
    label: 'Projective ablation',
    hint: "Remove the activation's component along it",
  },
  {
    id: 'coordinate_swap',
    label: 'Coordinate swap',
    hint: "Exchange this token's coordinate with another pinned token's",
  },
] as const;

/** Primitives that consume `strength`. The others ignore it entirely. */
const USES_STRENGTH = new Set(['additive']);

/**
 * What the model's vocabulary says about a hand-typed token.
 *
 * SHOWN EVEN WHEN IT PASSES. A silent success is indistinguishable from a check
 * that never ran, and the whole reason this control exists is that "is this one
 * token" is not answerable by looking at the string.
 */
function TokenVerdict({
  check,
  busy,
}: {
  check?: JLensTokenCheck;
  busy: boolean;
}) {
  if (busy && !check) {
    return (
      <span className="text-[10px] text-slate-500 dark:text-slate-500">
        checking the model's vocabulary…
      </span>
    );
  }
  if (!check) return null;
  return (
    <span
      data-testid="token-verdict"
      className={`text-[10px] ${
        check.usable
          ? 'text-emerald-700 dark:text-emerald-400'
          : 'text-amber-700 dark:text-amber-400'
      }`}
    >
      {check.usable ? `id ${check.ids[0]} — ` : ''}
      {check.detail}
    </span>
  );
}

interface InterventionCardProps {
  modelId: string;
  /** The prompt the readout on screen describes. */
  prompt: string;
  /** Tokens the user pinned — the directions available to act along. */
  pinned: string[];
  /** Layers the current readout covers, so a request cannot name an absent one. */
  layers: number[];
  artifactId: string | null;
}

/**
 * One arm's hit rate with its Wilson 95% interval.
 *
 * A RATE AND ITS INTERVAL, never a bare number. With twenty trials a ten-point
 * gap is noise, and the interval is what says so.
 */
interface Rates {
  hits: number;
  n: number;
  rate: number;
  ci95_low: number;
  ci95_high: number;
}

/**
 * What the task returns — `CausalReport.summary()`, verbatim.
 *
 * THIS SHAPE IS THE RUNG-2 ONE. The card was written against the rung-1 result
 * (`intervened_outcome` / `control_outcome` / `excess_over_control`, a lens-space
 * displacement) and never updated when the measurement became a real forward-pass
 * intervention. Every key it read had been gone since the rewrite, so the success
 * path called `.toFixed` on `undefined` and took the panel down — on success, and
 * only on success, which is why nothing noticed.
 */
interface Outcome {
  target_token: string;
  primitive: string;
  layers: number[];
  strength: number | null;
  n_trials: number;
  baseline_top1: Rates;
  intervened_top1: Rates;
  control_top1: Rates;
  baseline_top5: Rates;
  intervened_top5: Rates;
  control_top5: Rates;
  excess_top1_over_control: number;
  excess_top5_over_control: number;
  /** TRUE means the intervened and control intervals are DISJOINT. */
  separated_from_control: boolean;
  /**
   * FALSE when no outcome at this trial count COULD have separated them.
   *
   * A different question from `separated_from_control`, and the answers read
   * oppositely: one is "no effect was demonstrated", the other is "nothing
   * could have been demonstrated". The panel grew this branch and this card
   * did not, so the card kept printing the sentence the change exists to
   * remove — and it is the only surface from which a projective_ablation can
   * be run at all.
   */
  separation_attainable?: boolean;
  min_trials_for_separation?: number;
}

/** `12/24 = 0.500 [0.31, 0.69]` — the count, the rate and the interval. */
function fmtRates(r: Rates | undefined): string {
  if (!r) return 'n/a';
  return (
    `${r.hits}/${r.n} = ${r.rate.toFixed(3)} ` +
    `[${r.ci95_low.toFixed(2)}, ${r.ci95_high.toFixed(2)}]`
  );
}

export function InterventionCard({
  prompt,
  modelId,
  pinned,
  layers,
  artifactId,
}: InterventionCardProps) {
  const [open, setOpen] = useState(false);
  const [token, setToken] = useState('');
  const [primitive, setPrimitive] = useState<string>('additive');
  const [strength, setStrength] = useState(1);
  /**
   * Extra trial prompts, one per line.
   *
   * THE ONLY WAY TO REACH A SEPARABLE RESULT. Below four trials no outcome
   * separates the intervened and control intervals, and every surface sent a
   * single prompt — so "no effect was demonstrated" was the only verdict the
   * product could produce, and the panel's remedy pointed here at a card that
   * had no such control.
   */
  const [extraPrompts, setExtraPrompts] = useState('');
  /**
   * The token a swap exchanges with — the SECOND coordinate.
   *
   * It is also the token whose RANK is scored, which is why it is a visible
   * control rather than a silent pick: "swap A with B and see whether B
   * arrives" is a different experiment from "swap A with C", and the ranked
   * list chose from pin order without ever showing which.
   */
  const [partner, setPartner] = useState('');
  /**
   * Tokens typed by hand rather than picked from the pinned set.
   *
   * A direction is `W_U[id]`, so ANY single token has one — including tokens
   * the readout never surfaced, which are exactly the interesting swap targets:
   * "does ' Rome' arrive if I put ' Paris' where it was?" is a question about a
   * token that is, by hypothesis, NOT in the top-k yet. Restricting this form
   * to what is on screen was a limit the server never had.
   */
  const [typedDirection, setTypedDirection] = useState('');
  const [typedPartner, setTypedPartner] = useState('');
  /** Verdicts from the model's own vocabulary, keyed by the string checked. */
  const [tokenChecks, setTokenChecks] = useState<Record<string, JLensTokenCheck>>({});
  const [checking, setChecking] = useState(false);
  const [k, setK] = useState(4);
  const [seed, setSeed] = useState(20260802);
  const [state, setState] = useState<'idle' | 'running'>('idle');
  const [result, setResult] = useState<Outcome | null>(null);
  const [error, setError] = useState<string | null>(null);
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // NOT TRIMMED. The leading space is the character that makes ' Rome' a
  // single token and 'Rome' two — trimming it away silently converts a valid
  // direction into one the worker must refuse, and does it to the exact input
  // the verdict just approved. Emptiness is tested separately.
  const chosen = typedDirection || token || pinned[0] || '';
  const isSwap = primitive === 'coordinate_swap';
  // THE FIRST PINNED TOKEN THAT IS NOT THE DIRECTION. A swap with one token is
  // an additive steer wearing a swap's name, which the server refuses — so the
  // default must differ, and when nothing can differ the run is blocked here
  // rather than 422'd after a round trip.
  const chosenPartner =
    typedPartner ||
    (partner && partner !== chosen
      ? partner
      : pinned.find((t) => t !== chosen) || '');

  /**
   * Check a hand-typed token against the model's vocabulary, on blur.
   *
   * ON BLUR, NOT ON EVERY KEYSTROKE: mid-word text is almost always multi-token
   * and an error that appears while you are still typing trains you to ignore
   * it. And the verdict is CACHED by string, so re-checking the same token is
   * free.
   */
  const checkToken = async (raw: string) => {
    // KEYED AND SENT VERBATIM, whitespace included, because that is what will
    // be tokenised. `.trim()` here would check a different string than the one
    // the run uses.
    const t = raw;
    if (!t.trim() || !modelId || tokenChecks[t]) return;
    setChecking(true);
    try {
      const [verdict] = await jlensApi.checkTokens(modelId, [t]);
      if (verdict) setTokenChecks((prev) => ({ ...prev, [t]: verdict }));
    } catch {
      // A FAILED CHECK MUST NOT BLOCK THE RUN. The worker refuses a
      // multi-token direction anyway; this is an early warning, not a gate
      // that can strand the form when the endpoint is unreachable.
    } finally {
      setChecking(false);
    }
  };

  /** A verdict that says NO. Undefined when unchecked or fine. */
  const rejected = (t: string) => {
    const v = tokenChecks[t];
    return v && !v.usable ? v : undefined;
  };

  /**
   * The prompts this run will score, `prompt` first.
   *
   * DE-DUPLICATED AND TRIMMED HERE, not on the server: two identical trials are
   * one observation counted twice, which narrows the Wilson interval on
   * evidence that does not exist.
   */
  const trialPrompts = [
    prompt,
    ...extraPrompts.split('\n').map((p) => p.trim()),
  ].filter((p, i, all) => p.length > 0 && all.indexOf(p) === i);
  /**
   * Why the run is blocked, or undefined when it is not.
   *
   * A REASON, not a bare disabled button. A swap needs two distinct
   * coordinates; with one token pinned the control is dead and nothing on
   * screen says why, which reads as a broken form rather than a missing
   * prerequisite — the same defect the empty direction list had.
   */
  const blocked =
    state === 'running'
      ? 'A run is already in flight.'
      : !modelId
        ? 'Pick a model first.'
        : !chosen
          ? 'Pin a token — it supplies the direction to act along.'
          : layers.length === 0
            ? 'The readout covers no layers.'
            : isSwap && !chosenPartner
              ? 'A swap EXCHANGES two coordinates, so it needs a second token. Pin one, or type it.'
              : rejected(chosen)
                ? `Direction: ${rejected(chosen)!.detail}`
                : isSwap && rejected(chosenPartner)
                  ? `Exchange with: ${rejected(chosenPartner)!.detail}`
                  : undefined;
  const canRun = !blocked;

  const poll = (taskId: string) => {
    timer.current = setTimeout(async () => {
      try {
        const status = await getTaskStatus(taskId);
        if (status.state === 'SUCCESS') {
          setResult(status.result as Outcome);
          setState('idle');
          return;
        }
        if (status.state === 'FAILURE') {
          setError(status.error ?? 'The intervention failed with no reason given.');
          setState('idle');
          return;
        }
        poll(taskId);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Lost track of the run.');
        setState('idle');
      }
    }, POLL_MS);
  };

  const run = async () => {
    if (!canRun) return;
    setError(null);
    setResult(null);
    setState('running');
    try {
      const accepted = await jlensApi.intervene({
        model_id: modelId,
        // THE PROMPT ON SCREEN. This was `''` — an empty string — so every
        // intervention launched from this card scored a forward pass over
        // nothing, while the readout beside it described a real prompt. The
        // result named a layer and a direction and measured neither in the
        // context the reader was looking at.
        prompt,
        // ONE TRIAL EACH. The paper reports a FRACTION of trials — 50 two-hop
        // prompts, 192 swap trials — never one number from one prompt.
        prompts: trialPrompts.length > 1 ? trialPrompts : undefined,
        primitive,
        layers,
        direction_token: chosen,
        // THE PARTNER, for a swap only. Sending it for an additive steer would
        // silently change what gets SCORED — target_token defaults to
        // direction_token, and overriding it turns "does Paris arrive" into
        // "does Rome arrive" under an unchanged label.
        target_token: isSwap ? chosenPartner : undefined,
        // IGNORED BY THE OTHERS, and the server records null for them. Sent as
        // given so the request says what was asked for.
        strength,
        // Always sent. An intervention without a size-matched control is not a
        // weaker finding — it is not a finding.
        k,
        control_seed: seed,
        artifact_id: artifactId,
      });
      poll(accepted.task_id);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not queue the run.');
      setState('idle');
    }
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Intervention
        </span>
        <span className="text-[10px] text-slate-500 dark:text-slate-500">
          rung 2 · real intervention, scored against a matched control
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          disabled={!pinned.length}
          title={
            pinned.length
              ? undefined
              : 'Pin a token first — it supplies the direction to act along.'
          }
          className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          <Zap className="h-3 w-3" />
          {open ? 'Close' : 'Intervene…'}
        </button>
      </div>

      {open && (
        <div className="mt-3 space-y-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Direction — a pinned token
              </span>
              <select
                value={chosen}
                onChange={(e) => setToken(e.target.value)}
                disabled={pinned.length === 0}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 disabled:opacity-60 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              >
                {/* AN EMPTY LIST MUST EXPLAIN ITSELF. With nothing pinned this
                    rendered as a blank select beside three fields that DO
                    accept typing, which reads as a broken control rather than
                    as a missing prerequisite — and the caption "a pinned token"
                    only makes sense once you already know what pinning is. */}
                {pinned.length === 0 ? (
                  <option value="">
                    No pinned tokens — click one in the readout below to pin it
                  </option>
                ) : (
                  pinned.map((p) => (
                    <option key={p} value={p}>
                      {p}
                    </option>
                  ))
                )}
              </select>
              {/* OR TYPE ONE. The pinned set is what the readout surfaced; the
                  vocabulary is much larger, and a swap target is usually a
                  token that is NOT in the top-k yet — that is the point of
                  asking whether it arrives. */}
              <input
                type="text"
                value={typedDirection}
                onChange={(e) => setTypedDirection(e.target.value)}
                onBlur={(e) => void checkToken(e.target.value)}
                placeholder="…or type any token, e.g. ' Paris'"
                data-testid="intervention-direction-typed"
                className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-[11px] text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
              <TokenVerdict check={tokenChecks[chosen]} busy={checking} />
            </label>

            {/* THE SECOND COORDINATE, NAMED. Only for a swap — the other
                primitives take one direction, and an always-visible partner
                would imply they use it. It is also the token whose RANK is
                scored, which the caption says outright: a reader who thinks it
                is merely "the other one" will misread every result. */}
            {isSwap && (
              <label className="flex flex-col gap-1">
                <span className="text-xs text-slate-600 dark:text-slate-400">
                  Exchange with — and this is what gets scored
                </span>
                <select
                  value={chosenPartner}
                  onChange={(e) => setPartner(e.target.value)}
                  disabled={pinned.filter((t) => t !== chosen).length === 0}
                  data-testid="intervention-partner"
                  className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 disabled:opacity-60 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                >
                  {pinned.filter((t) => t !== chosen).length === 0 ? (
                    <option value="">
                      Pin a second token — a swap needs two coordinates
                    </option>
                  ) : (
                    pinned
                      .filter((t) => t !== chosen)
                      .map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))
                  )}
                </select>
                <input
                  type="text"
                  value={typedPartner}
                  onChange={(e) => setTypedPartner(e.target.value)}
                  onBlur={(e) => void checkToken(e.target.value)}
                  placeholder="…or type any token, e.g. ' Rome'"
                  data-testid="intervention-partner-typed"
                  className="rounded border border-slate-300 bg-white px-2 py-1 font-mono text-[11px] text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                />
                <TokenVerdict check={tokenChecks[chosenPartner]} busy={checking} />
              </label>
            )}
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Primitive
              </span>
              <select
                value={primitive}
                onChange={(e) => setPrimitive(e.target.value)}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              >
                {PRIMITIVES.map((p) => (
                  <option key={p.id} value={p.id} title={p.hint}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="grid grid-cols-3 gap-3">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Strength
                {/* SAID, NOT IMPLIED. An ablation and a swap take no strength —
                    the hook passes it to `apply_additive` alone and the server
                    records null for the others. An editable box beside a
                    primitive that ignores it invites a strength sweep that
                    returns bit-identical results at every value. */}
                {!USES_STRENGTH.has(primitive) && (
                  <span className="ml-1 text-[10px] font-normal text-amber-700 dark:text-amber-400">
                    — ignored by {primitive.replace('_', ' ')}
                  </span>
                )}
              </span>
              <input
                type="number"
                step="0.1"
                value={strength}
                disabled={!USES_STRENGTH.has(primitive)}
                onChange={(e) => setStrength(Number(e.target.value))}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Control size k
              </span>
              <input
                type="number"
                min={1}
                value={k}
                onChange={(e) => setK(Math.max(1, Number(e.target.value)))}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Control seed
              </span>
              <input
                type="number"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-500">
            The control runs every time and cannot be turned off. &ldquo;A random
            direction&rdquo; is not a control; &ldquo;k random directions from
            seed s&rdquo; is, and one nobody can reconstruct is not one either.
          </p>

          <label className="flex flex-col gap-1">
            <span className="text-xs text-slate-600 dark:text-slate-400">
              More trial prompts — one per line
            </span>
            <textarea
              value={extraPrompts}
              onChange={(e) => setExtraPrompts(e.target.value)}
              rows={4}
              placeholder={'The capital of Italy is\nThe capital of Japan is'}
              className="rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              data-testid="intervention-prompts"
            />
          </label>
          {/* THE COUNT AND WHAT IT BUYS, before the run rather than after it.
              Below four trials NO outcome separates the intervened and control
              intervals — a perfect intervened arm against a perfect null
              control still overlaps — so a run at this size can only report
              "no effect demonstrated", which reads as a fact about the
              direction and is a fact about the sample. Saying so afterwards
              costs a GPU job to learn. */}
          <p
            className={`text-[10px] ${
              trialPrompts.length < MIN_TRIALS
                ? 'text-amber-700 dark:text-amber-400'
                : 'text-slate-500 dark:text-slate-500'
            }`}
            data-testid="intervention-trial-count"
          >
            {trialPrompts.length} trial{trialPrompts.length === 1 ? '' : 's'}
            {trialPrompts.length < MIN_TRIALS
              ? ` — separation is not attainable below ${MIN_TRIALS}. This will run, and its verdict will describe the sample rather than the direction.`
              : ' — enough for the intervals to separate if there is an effect.'}
          </p>

          <button
            type="button"
            onClick={run}
            disabled={!canRun}
            title={blocked}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            {state === 'running' ? (
              <span className="flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" /> Running with control…
              </span>
            ) : (
              'Run with control'
            )}
          </button>

          {error && (
            <p className="text-[11px] text-red-600 dark:text-red-400" role="alert">
              {error}
            </p>
          )}

          {result && (
            <div
              className="rounded border border-slate-200 p-2 dark:border-slate-700"
              data-testid="intervention-result"
            >
              {/* THE VERDICT FIRST, in the terms the measurement supports:
                  disjoint intervals or not. Overlap is "not demonstrated
                  here", never "demonstrated absent" — the asymmetry is the
                  whole reason the control exists. */}
              <p
                className={`text-[11px] font-medium ${
                  result.separated_from_control
                    ? 'text-emerald-700 dark:text-emerald-400'
                    : 'text-amber-700 dark:text-amber-400'
                }`}
              >
                {result.separation_attainable === false
                  ? `Only ${result.n_trials} trial${
                      result.n_trials === 1 ? '' : 's'
                    } — separation is not attainable below ${
                      result.min_trials_for_separation ?? 4
                    }. This says nothing about the direction yet; add prompts below.`
                  : result.separated_from_control
                    ? 'The intervals are DISJOINT — an effect over the matched control was demonstrated.'
                    : 'The intervals OVERLAP — no effect was demonstrated here, which is not the same as none existing.'}
              </p>

              {/* ALL THREE ARMS. The baseline is not decoration: an
                  intervention that "achieves" top-1 on prompts where the model
                  already answered that way has moved nothing, and without the
                  baseline any prompt set can manufacture a result. */}
              <dl className="mt-1.5 grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5 font-mono text-[10px]">
                <dt className="text-slate-500 dark:text-slate-400">baseline</dt>
                <dd className="text-slate-700 dark:text-slate-200">
                  {fmtRates(result.baseline_top1)}
                </dd>
                <dt className="text-slate-500 dark:text-slate-400">intervened</dt>
                <dd className="text-slate-700 dark:text-slate-200">
                  {fmtRates(result.intervened_top1)}
                </dd>
                <dt className="text-slate-500 dark:text-slate-400">control</dt>
                <dd className="text-slate-700 dark:text-slate-200">
                  {fmtRates(result.control_top1)}
                </dd>
              </dl>

              <p className="mt-1 font-mono text-[10px] text-slate-500 dark:text-slate-500">
                excess top-1 over control{' '}
                {result.excess_top1_over_control.toFixed(4)} · {result.n_trials}{' '}
                trial{result.n_trials === 1 ? '' : 's'}
              </p>

            </div>
          )}
        </div>
      )}
    </section>
  );
}

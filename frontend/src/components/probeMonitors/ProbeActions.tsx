/**
 * The two things you can do WITH a probe, on the report where its evidence is.
 *
 * ⚠ BOTH WERE UNREACHABLE FROM THE UI. `evaluate` and `score` existed in the API client and
 * nothing called them, so:
 *
 *   * **rung 1 and rung 2 could not be climbed** — a probe whose run did not include the right
 *     evaluation sets could never be given more, and the ladder's own "Next:" instruction was
 *     therefore an instruction the panel could not carry out;
 *   * **a probe could not be pointed at your own text**, which is the most direct question anyone
 *     has of a detector and the one the whole feature exists to answer.
 *
 * Scoring is deliberately two-step and says so: the POST queues GPU work and returns a task id,
 * and the result is polled. PENDING means EITHER queued OR an id Celery has never heard of — the
 * server cannot tell those apart, so this does not pretend to either.
 */
import { useEffect, useRef, useState } from 'react';
import { FlaskConical, Play } from 'lucide-react';

import type { ProbeDataset, ProbeScoreTask } from '../../types/probeMonitor';
import { TokenTrace } from './TokenTrace';

interface ProbeActionsProps {
  probeId: string;
  /** Evaluation views, so a probe can be given sets its run did not include. */
  datasets: ProbeDataset[];
  /** Ids already evaluated — offered but pre-marked, because re-running one is legitimate. */
  alreadyEvaluated: string[];
  onEvaluate: (datasetIds: string[]) => Promise<boolean>;
  onScore: (input: { text: string }) => Promise<ProbeScoreTask | null>;
  onPollScore: (taskId: string) => Promise<ProbeScoreTask | null>;
  /** Forwarded to the trace legend so it can say what "up" is toward, and how pooling works. */
  concept?: string;
  rule?: string;
}

export function ProbeActions({
  probeId,
  datasets,
  alreadyEvaluated,
  onEvaluate,
  onScore,
  onPollScore,
  concept,
  rule,
}: ProbeActionsProps) {
  const [chosen, setChosen] = useState<string[]>([]);
  const [evaluating, setEvaluating] = useState(false);

  const [text, setText] = useState('');
  const [scoring, setScoring] = useState(false);
  const [scoreResult, setScoreResult] = useState<ProbeScoreTask | null>(null);

  // A poll that stops when the component goes away, so closing the report does not leave a timer
  // writing into a dead tree.
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    []
  );

  const evalSets = datasets.filter((d) => d.role === 'eval');

  async function poll(taskId: string, attempt = 0) {
    const result = await onPollScore(taskId);
    setScoreResult(result);
    const done = result && (result.status === 'SUCCESS' || result.status === 'FAILURE');
    if (done || attempt > 60) {
      setScoring(false);
      return;
    }
    timer.current = setTimeout(() => void poll(taskId, attempt + 1), 2000);
  }

  return (
    <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2" data-testid="probe-actions">
      {/* ── climb the ladder ─────────────────────────────────────────────── */}
      <section className="rounded border border-slate-700 bg-slate-950/40 p-3">
        <p className="flex items-center gap-2 text-sm font-medium text-slate-200">
          <FlaskConical className="h-4 w-4 text-sky-400" aria-hidden="true" />
          Evaluate on more sets
        </p>
        <p className="mt-1 text-[11px] text-slate-500">
          This is how rung 1 and rung 2 are reached. An out-of-distribution set whose confidence
          interval clears chance is what rung 2 turns on.
        </p>
        {evalSets.length === 0 ? (
          <p className="mt-2 text-xs text-amber-300">
            No evaluation views exist — create one on the Datasets tab.
          </p>
        ) : (
          <div className="mt-2 flex flex-wrap gap-2">
            {evalSets.map((set) => (
              <label
                key={set.id}
                className="flex items-center gap-1.5 rounded border border-slate-700 px-2 py-1 text-xs text-slate-300"
                title={
                  alreadyEvaluated.includes(set.id)
                    ? 'Already evaluated. Running it again replaces that result.'
                    : undefined
                }
              >
                <input
                  type="checkbox"
                  checked={chosen.includes(set.id)}
                  onChange={() =>
                    setChosen((c) =>
                      c.includes(set.id) ? c.filter((x) => x !== set.id) : [...c, set.id]
                    )
                  }
                  data-testid={`evaluate-set-${set.id}`}
                />
                {set.config || set.name}
                {alreadyEvaluated.includes(set.id) ? (
                  <span className="text-[10px] text-slate-500">done</span>
                ) : null}
                {set.distribution === 'out_of_distribution' ? (
                  <span className="rounded bg-sky-900/50 px-1 text-[10px] text-sky-300">OOD</span>
                ) : null}
              </label>
            ))}
          </div>
        )}
        <button
          type="button"
          disabled={chosen.length === 0 || evaluating}
          onClick={async () => {
            setEvaluating(true);
            const ok = await onEvaluate(chosen);
            setEvaluating(false);
            if (ok) setChosen([]);
          }}
          className="mt-2 rounded border border-sky-700 px-2 py-1 text-xs text-sky-200 hover:bg-sky-950/40 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
          data-testid="evaluate-submit"
        >
          {evaluating ? 'Queueing…' : `Evaluate on ${chosen.length || 'no'} set(s)`}
        </button>
        <p className="mt-1 text-[11px] text-slate-500">
          Queues GPU work. The rung updates when it finishes — reopen the report to see it.
        </p>
      </section>

      {/* ── try it ───────────────────────────────────────────────────────── */}
      <section className="rounded border border-slate-700 bg-slate-950/40 p-3">
        <p className="flex items-center gap-2 text-sm font-medium text-slate-200">
          <Play className="h-4 w-4 text-emerald-400" aria-hidden="true" />
          Try it on your own text
        </p>
        <textarea
          rows={3}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste anything the model might be asked…"
          className="mt-2 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1.5 text-xs text-slate-200"
          data-testid="score-text"
        />
        <button
          type="button"
          disabled={text.trim() === '' || scoring}
          onClick={async () => {
            setScoring(true);
            setScoreResult(null);
            const accepted = await onScore({ text });
            if (accepted?.task_id) void poll(accepted.task_id);
            else setScoring(false);
          }}
          className="mt-2 rounded border border-emerald-700 px-2 py-1 text-xs text-emerald-200 hover:bg-emerald-950/40 disabled:cursor-not-allowed disabled:border-slate-700 disabled:text-slate-500"
          data-testid="score-submit"
        >
          {scoring ? 'Scoring…' : 'Score this text'}
        </button>

        {scoreResult ? (
          <div className="mt-2 text-xs" data-testid="score-result">
            {scoreResult.status === 'SUCCESS' && scoreResult.result ? (
              /*
               * ⚠ THE VERDICT IS THE SERVER'S, AND THE FIRST VERSION OF THIS RECOMPUTED IT.
               * `score_one` returns `fires` — null when no threshold was placed, which is "the
               * probe has said nothing", not "no". Comparing `score >= threshold` here would be a
               * second definition of a decision the server already makes, and the two would
               * disagree the first time the comparison's edge case changed.
               *
               * `TokenTrace` renders the per-token trace and was, until now, a component only its
               * own tests had ever rendered — the seventh unreachable thing this audit found.
               */
              <TokenTrace result={scoreResult.result} concept={concept} rule={rule} />
            ) : scoreResult.status === 'FAILURE' ? (
              // The REASON, not an empty trace. An empty trace reads as "it scored nothing".
              <p className="text-amber-300">{scoreResult.error || 'the scoring task failed'}</p>
            ) : (
              <p className="text-slate-400">
                {scoreResult.status} — queued, or a task id the queue has never heard of. It cannot
                tell those apart, so neither can this.
              </p>
            )}
          </div>
        ) : null}
      </section>
      <span className="hidden" data-testid="probe-actions-for">
        {probeId}
      </span>
    </div>
  );
}

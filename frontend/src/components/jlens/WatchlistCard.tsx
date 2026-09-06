/**
 * Author a runtime watchlist (BR-025).
 *
 * THE SCORING DEFINITION IS REQUIRED, and the server refuses a watchlist
 * without one. That is not paperwork: a threshold applied to a differently
 * computed score is a DIFFERENT DETECTOR, and the consumer at the other end
 * has no way to notice it is running one. The form mirrors the refusal rather
 * than letting a user discover it after export.
 *
 * The artifact reference is required for the same reason — a watchlist scored
 * through one lens and evaluated through another is not the detector it says
 * it is.
 */

import { useState } from 'react';
import { ListChecks, Loader2 } from 'lucide-react';
import { jlensApi } from '../../api/jlens';

interface WatchlistCardProps {
  /** Slug of the artifact this watchlist scores through. */
  artifactId: string | null;
}

interface Concept {
  token: string;
  threshold: number;
}

export function parseConcepts(raw: string): Concept[] {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [token, threshold] = line.split(/[,\t]/);
      return {
        token: (token ?? '').trim(),
        // A missing threshold is NOT defaulted to zero: zero fires on
        // everything, which is a detector that always says yes.
        threshold: Number.parseFloat((threshold ?? '').trim()),
      };
    })
    .filter((c) => c.token !== '' && Number.isFinite(c.threshold));
}

export function WatchlistCard({ artifactId }: WatchlistCardProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState('');
  const [scoring, setScoring] = useState('');
  const [conceptsRaw, setConceptsRaw] = useState('');
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const concepts = parseConcepts(conceptsRaw);
  const canSave =
    !busy &&
    !!artifactId &&
    name.trim() !== '' &&
    scoring.trim() !== '' &&
    concepts.length > 0;

  const save = async () => {
    if (!canSave || !artifactId) return;
    setBusy(true);
    setError(null);
    setSaved(null);
    try {
      const res = await jlensApi.createWatchlist({
        name: name.trim(),
        artifact_ref: artifactId,
        scoring_definition: scoring.trim(),
        concepts,
      });
      setSaved(`${res.name} — ${res.concept_count} concepts, validated`);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The watchlist was refused.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Watchlist
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          disabled={!artifactId}
          title={
            artifactId
              ? undefined
              : 'A watchlist scores THROUGH an artifact; fit one first.'
          }
          className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          <ListChecks className="h-3 w-3" />
          {open ? 'Close' : 'Author…'}
        </button>
      </div>

      {open && (
        <div className="mt-3 space-y-3">
          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">Name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="evaluation-awareness"
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-slate-600 dark:text-slate-400">
                Scoring definition — required
              </span>
              <input
                value={scoring}
                onChange={(e) => setScoring(e.target.value)}
                placeholder="max softmax prob over layers 10-15"
                className="rounded border border-slate-300 bg-white px-2 py-1.5 text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </label>
          </div>

          <label className="flex flex-col gap-1">
            <span className="text-xs text-slate-600 dark:text-slate-400">
              Concepts — one per line, <code>token, threshold</code>
            </span>
            <textarea
              rows={4}
              value={conceptsRaw}
              onChange={(e) => setConceptsRaw(e.target.value)}
              placeholder={' evaluation, 0.4\n test, 0.35'}
              className="rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
            <span className="text-[10px] text-slate-500 dark:text-slate-500">
              {concepts.length} parsed. A line with no threshold is dropped, not
              defaulted — a zero threshold fires on everything.
            </span>
          </label>

          <p className="text-[10px] text-slate-500 dark:text-slate-500">
            The scoring definition travels with the list because a threshold
            applied to a differently computed score is a different detector, and
            the consumer cannot tell.
          </p>

          <button
            type="button"
            onClick={save}
            disabled={!canSave}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            {busy ? (
              <span className="flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" /> Validating…
              </span>
            ) : (
              'Validate'
            )}
          </button>

          {saved && (
            <p className="text-[11px] text-emerald-700 dark:text-emerald-400">
              {saved}
            </p>
          )}
          {error && (
            <p className="text-[11px] text-red-600 dark:text-red-400" role="alert">
              {error}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

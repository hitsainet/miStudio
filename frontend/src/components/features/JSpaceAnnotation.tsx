/**
 * Describe one SAE feature in J-space: what its decoder direction pushes TOWARD.
 *
 * WHY IT LIVES HERE AND NOT IN J-LENS. The subject is a FEATURE, and features
 * live in this modal. Putting it on the J-Lens panel would have meant asking
 * the user to retype an SAE id and a feature index they were already looking at.
 *
 * WHY IT COULD NOT EXIST BEFORE. The endpoint required a raw d_model decoder
 * direction, which a browser cannot produce. It now resolves that server-side
 * from `sae_id` + `feature_id` — the same `resolve_decoder_weight` steering
 * uses — so this component sends four scalars instead of thousands of floats.
 *
 * TWO INDEPENDENT FIELDS (BR-012). The geometric reading and the behavioural
 * class are separate, and `workspace_class` is UNKNOWN without a band report
 * for this model. UNKNOWN is a real answer: the published band boundaries were
 * measured on one specific model and do not transfer, so miStudio draws no
 * bands rather than guessing. The component says so instead of hiding the field.
 *
 * RUNG 0. A direction appearing in a readout says it was PRESENT, not that the
 * model used it. Raising that takes an intervention with a matched control.
 */

import { useState } from 'react';
import { Loader2, Sparkles } from 'lucide-react';
import { jlensApi } from '../../api/jlens';
import type { JLensAnnotation } from '../../types/jlens';

interface JSpaceAnnotationProps {
  /**
   * A miStudio feature id. That is ALL this needs.
   *
   * The SAE, its model and the layer are resolved server-side from the feature
   * row — they already live there, and this modal has neither. Restating them
   * here would mean asking the user to retype facts they are looking at.
   */
  featureId: string;
  /** The feature's current label, so a disagreement can be computed. */
  labelTokens?: string[];
}

export function JSpaceAnnotation({
  featureId,
  labelTokens = [],
}: JSpaceAnnotationProps) {
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<JLensAnnotation | null>(null);
  const [error, setError] = useState<string | null>(null);

  const ready = Boolean(featureId);

  const run = async () => {
    if (!ready) return;
    setBusy(true);
    setError(null);
    try {
      // Feature id only. The server resolves the SAE, its model, the layer and
      // the decoder direction — all of which live on the feature row, and none
      // of which this modal has.
      setResult(await jlensApi.annotate({ feature_id: featureId, label_tokens: labelTokens }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The annotation failed.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={run}
          disabled={!ready || busy}
          title={ready ? undefined : 'No feature selected.'}
          className="flex items-center gap-1.5 rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
        >
          {busy ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            <Sparkles className="h-3.5 w-3.5" />
          )}
          {busy ? 'Reading the direction…' : 'Annotate in J-space'}
        </button>
        <span className="text-[11px] text-slate-500 dark:text-slate-400">
          rung 0 · what this direction points toward, not what the model did with it
        </span>
      </div>

      {error && (
        <p className="text-xs text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}

      {result && (
        <div className="space-y-3 rounded-lg border border-slate-200 p-3 dark:border-slate-700">
          <div>
            <div className="mb-1 text-xs font-medium text-slate-600 dark:text-slate-400">
              Pushes toward
            </div>
            <div className="flex flex-wrap gap-1.5">
              {result.top_tokens.map((t, i) => (
                <span
                  key={`${t}-${i}`}
                  className="rounded border border-slate-300 bg-slate-100 px-1.5 py-0.5 font-mono text-[11px] text-slate-700 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-300"
                >
                  {t}
                </span>
              ))}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-3 text-xs">
            <div>
              <div className="text-slate-500 dark:text-slate-500">Lens kurtosis</div>
              <div className="font-mono text-slate-800 dark:text-slate-200">
                {result.lens_kurtosis === null
                  ? 'not computed'
                  : result.lens_kurtosis.toFixed(3)}
              </div>
            </div>
            <div>
              <div className="text-slate-500 dark:text-slate-500">Workspace class</div>
              <div className="font-mono text-slate-800 dark:text-slate-200">
                {result.workspace_class}
              </div>
              {result.workspace_class === 'UNKNOWN' && (
                // A REAL ANSWER, not a gap. Without boundaries measured on THIS
                // model there is no principled middle of the stack to classify
                // against, and the published ones were measured elsewhere.
                <div className="mt-0.5 text-[10px] text-amber-600 dark:text-amber-400">
                  no band report for this model — boundaries from another model
                  do not transfer
                </div>
              )}
            </div>
          </div>

          {result.disagreement_score != null && (
            <div className="text-xs">
              <span className="text-slate-500 dark:text-slate-500">
                Label disagreement{' '}
              </span>
              <span
                className={`font-mono ${
                  result.has_disagreement
                    ? 'text-amber-600 dark:text-amber-400'
                    : 'text-slate-800 dark:text-slate-200'
                }`}
              >
                {result.disagreement_score.toFixed(2)}
              </span>
              {result.has_disagreement && (
                <span className="ml-1 text-[10px] text-amber-600 dark:text-amber-400">
                  — the label and the direction disagree; neither is
                  automatically right
                </span>
              )}
            </div>
          )}

          <p className="text-[10px] text-slate-500 dark:text-slate-500">
            The geometric reading and the behavioural class are independent
            (BR-012). A sharp direction is not evidence of workspace membership —
            a motor direction is sharp too.
          </p>
        </div>
      )}
    </div>
  );
}

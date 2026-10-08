/**
 * J-lens artifact registry for the selected model.
 *
 * PRESENCE IS NOT VALIDITY, and the strip says which it is showing. An artifact
 * listed here has been found on disk; whether it can be read out is the outcome
 * of running the validation suite, and until that runs the strip reports
 * "not validated" rather than a tick.
 *
 * `passed` vs `serviceable` is surfaced rather than collapsed. Two of the six
 * checks need a live external consumer, so a validation run from the workbench
 * reports them NOT_RUN and `passed` is false — which means "not yet cleared for
 * handover", not "broken". Showing a red cross there would be a lie about a
 * perfectly good artifact.
 */

import { useState } from 'react';
import { CheckCircle2, CircleSlash, Clock, Loader2, XCircle } from 'lucide-react';
import { jlensApi } from '../../api/jlens';
import { LayerCoverage, missingLayers } from './LayerCoverage';
import type {
  JLensArtifactSummary,
  JLensValidationResponse,
} from '../../types/jlens';

interface ArtifactsStripProps {
  artifacts: JLensArtifactSummary[];
  /** Pre-fill a fit with exactly the layers this artifact is missing. */
  onFitMissing?: (layers: number[]) => void;
  /** Slug this model's repo id would produce; '' when no model is selected. */
  expectedSlug: string;
  /** Dimensions the envelope bound is derived from — of THIS model. */
  dims: { d_model: number; n_layers: number; n_vocab: number } | null;
}

const LOCAL_CHECKS = ['structural', 'naming', 'envelope', 'semantic'];

function statusIcon(status: string) {
  if (status === 'pass') {
    return <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />;
  }
  if (status === 'fail') {
    return <XCircle className="h-3.5 w-3.5 text-red-500" />;
  }
  if (status === 'deferred') {
    // DISTINCT FROM not_run, WHICH IS THE WHOLE POINT. Deferred means "known
    // to be unrunnable here, and published anyway"; not_run means "we do not
    // know". They used to render identically because `deferred` did not exist
    // as a status at all — the two consumer-interop checks were stamped `pass`,
    // so this row showed a green tick for something nothing had run.
    return (
      <Clock
        className="h-3.5 w-3.5 text-amber-500"
        aria-label="deferred — not run here"
      />
    );
  }
  return <CircleSlash className="h-3.5 w-3.5 text-slate-400" />;
}

export function ArtifactsStrip({
  artifacts,
  expectedSlug,
  dims,
  onFitMissing,
}: ArtifactsStripProps) {
  const [report, setReport] = useState<JLensValidationResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mine = expectedSlug
    ? artifacts.find((a) => a.slug === expectedSlug)
    : undefined;

  const runValidation = async () => {
    if (!mine || !dims) return;
    setRunning(true);
    setError(null);
    try {
      setReport(await jlensApi.validateArtifact(mine.slug, dims));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Validation failed.');
    } finally {
      setRunning(false);
    }
  };

  const serviceable =
    report != null &&
    LOCAL_CHECKS.every((c) =>
      report.results.some((r) => r.check === c && r.status === 'pass')
    );

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          J-lens artifact
        </span>
        {!expectedSlug ? (
          <span className="text-[11px] text-slate-500">Select a model.</span>
        ) : mine ? (
          <>
            <span className="rounded border border-slate-300 bg-slate-100 px-2 py-0.5 font-mono text-[11px] text-slate-700 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-300">
              {mine.lens_file}
            </span>
            <span className="font-mono text-[10px] text-slate-500">
              {(mine.size_bytes / 1_000_000).toFixed(1)} MB
            </span>
            {!mine.has_config && (
              <span className="text-[10px] text-amber-600 dark:text-amber-400">
                no config.yaml — the construction recipe is missing
              </span>
            )}
            <LayerCoverage
              covered={mine.layers ?? []}
              total={dims?.n_layers ?? null}
              targetLayer={mine.target_layer}
            />
            {onFitMissing &&
              dims?.n_layers &&
              (mine.layers?.length ?? 0) > 0 &&
              missingLayers(mine.layers ?? [], dims.n_layers).length > 0 && (
                <button
                  type="button"
                  onClick={() =>
                    onFitMissing(missingLayers(mine.layers ?? [], dims.n_layers))
                  }
                  title="Open the fit form with exactly the unfitted layers filled in"
                  className="rounded border border-amber-400 px-2 py-0.5 text-[10px] text-amber-700 hover:bg-amber-50 dark:border-amber-600 dark:text-amber-400 dark:hover:bg-amber-950/40"
                >
                  Fit the missing{' '}
                  {missingLayers(mine.layers ?? [], dims.n_layers).length}
                </button>
              )}
            <button
              type="button"
              onClick={runValidation}
              title={
                dims
                  ? undefined
                  : "This model's dimensions are unknown, and the envelope bound is derived from them — a guessed bound reports a verdict it never computed."
              }
              disabled={running || !dims}
              className="ml-auto rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
            >
              {running ? (
                <span className="flex items-center gap-1">
                  <Loader2 className="h-3 w-3 animate-spin" /> Validating…
                </span>
              ) : (
                'Validate'
              )}
            </button>
          </>
        ) : (
          <span className="text-[11px] text-slate-500 dark:text-slate-500">
            No artifact for this model. The logit lens needs none; the Jacobian
            lens does — fit one to enable it.
          </span>
        )}
      </div>

      {error && (
        <p className="text-xs text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}

      {report && (
        <div className="space-y-1">
          <div className="flex flex-wrap items-center gap-2 text-[11px]">
            <span
              className={
                serviceable
                  ? 'font-medium text-emerald-700 dark:text-emerald-400'
                  : 'font-medium text-red-600 dark:text-red-400'
              }
            >
              {serviceable ? 'Serviceable — readable here' : 'Not serviceable'}
            </span>
            <span className="text-slate-500 dark:text-slate-500">
              {report.passed
                ? '· cleared for handover to an external consumer'
                : '· not yet cleared for handover: two checks need a live consumer'}
            </span>
          </div>
          <ul className="space-y-0.5">
            {report.results.map((r) => (
              <li
                key={r.check}
                className="flex items-start gap-1.5 font-mono text-[10px] text-slate-600 dark:text-slate-400"
              >
                {statusIcon(r.status)}
                <span className="w-40 shrink-0">{r.check}</span>
                <span className="text-slate-500 dark:text-slate-500">{r.detail}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}

/**
 * Per-evaluation-set metrics (032 FR-9).
 *
 * ⚠ A REFUSAL RENDERS ITS REASON. NEVER A 0, NEVER A BLANK, NEVER A DASH ALONE.
 * `scored: false` means the set could not be scored — below 20 examples of a class — and
 * the row shows that sentence with both counts. A 0 reads as "measured, and terrible"; a
 * blank reads as "not run yet"; a 0.5 reads as "measured, and chance". None of those is
 * what happened, and this estate has already paid for a metric that reported a
 * comfortable 0.5 instead of refusing.
 */
import type { ProbeEvaluation } from '../../types/probeMonitor';

interface MetricsTableProps {
  evaluations: ProbeEvaluation[];
  /** id → display name, so the table names sets rather than showing ids. */
  datasetNames?: Record<string, string>;
}

function formatAuroc(value: number | null | undefined): string {
  if (value === null || value === undefined) return '—';
  return value.toFixed(3);
}

export function MetricsTable({ evaluations, datasetNames = {} }: MetricsTableProps) {
  if (evaluations.length === 0) {
    return (
      <p className="text-sm text-slate-400" data-testid="metrics-empty">
        No evaluation has been run for this probe yet.
      </p>
    );
  }
  return (
    <table className="w-full text-sm" data-testid="metrics-table">
      <thead>
        <tr className="border-b border-slate-700 text-left text-xs uppercase tracking-wide text-slate-400">
          <th className="py-2 pr-3">Set</th>
          <th className="py-2 pr-3">AUROC</th>
          <th className="py-2 pr-3">95% CI</th>
          <th className="py-2 pr-3">Recall @ 1% FPR</th>
          <th className="py-2 pr-3">n</th>
        </tr>
      </thead>
      <tbody>
        {evaluations.map((evaluation) => {
          const metrics = evaluation.metrics ?? { scored: false };
          const name =
            metrics.name || datasetNames[evaluation.dataset_id] || evaluation.dataset_id;
          if (!metrics.scored) {
            return (
              <tr
                key={evaluation.id}
                className="border-b border-slate-800"
                data-testid="metrics-refused"
              >
                <td className="py-2 pr-3 align-top font-medium text-slate-200">{name}</td>
                <td className="py-2 pr-3 align-top text-amber-300" colSpan={4}>
                  <span className="font-medium">Not scored.</span>{' '}
                  <span data-testid="refusal-reason">
                    {metrics.reason ?? 'no reason was recorded'}
                  </span>
                </td>
              </tr>
            );
          }
          const onePercent = (metrics.operating_points ?? []).find(
            (point) => Math.abs(point.target_fpr - 0.01) < 1e-9
          );
          return (
            <tr key={evaluation.id} className="border-b border-slate-800">
              <td className="py-2 pr-3 font-medium text-slate-200">
                {name}
                {metrics.out_of_distribution ? (
                  <span className="ml-2 rounded bg-slate-700/60 px-1.5 py-0.5 text-[10px] text-slate-300">
                    out-of-distribution
                  </span>
                ) : null}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-100">
                {formatAuroc(metrics.auroc)}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-400">
                {metrics.ci
                  ? `${metrics.ci.low.toFixed(3)}–${metrics.ci.high.toFixed(3)}`
                  : '—'}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-100">
                {onePercent ? (
                  <>
                    {(onePercent.recall * 100).toFixed(1)}%
                    {/* THE REALISED RATE, not the target. With 100 negatives the
                        achievable rates are multiples of 1%, so the two differ often. */}
                    <span className="ml-1 text-[10px] text-slate-500">
                      at {(onePercent.realised_fpr * 100).toFixed(1)}% spent
                    </span>
                  </>
                ) : (
                  '—'
                )}
              </td>
              <td className="py-2 pr-3 tabular-nums text-slate-400">
                {metrics.n_positive ?? '—'}/{metrics.n_negative ?? '—'}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/**
 * What a judge run measured, per evaluation set — the numbers a probe is read against.
 *
 * ⚠ THESE WERE COMPUTED SINCE 032 AND SHOWN NOWHERE. The judge's per-set AUROC sat in
 * `metrics.per_set` and no component read it, so the Judge tab listed a run's model and status and
 * never what the judge actually scored. Every figure here is the server's; a set the judge could
 * not score shows its REASON, never a zero.
 */
import type { JudgeSetResult, ProbeJudgeRun } from '../../types/probeMonitor';

const pct = (value: number | null | undefined) =>
  value == null ? '—' : `${(value * 100).toFixed(1)}%`;

interface Props {
  run: ProbeJudgeRun;
  datasetNames: Record<string, string>;
}

export function JudgeRunHeading({ run }: { run: ProbeJudgeRun }) {
  return (
    <>
      <p className="text-slate-200">
        {run.prompt_title ?? run.prompt_version}{' '}
        <span className="font-mono text-xs text-slate-400">{run.prompt_version}</span>
        {' · '}
        {run.model} · {run.status}
      </p>
      {run.counts_toward_rung != null ? (
        <p
          className={`text-xs ${run.counts_toward_rung ? 'text-emerald-300' : 'text-amber-300'}`}
          data-testid="judge-run-counts"
        >
          {run.counts_toward_rung ? 'Counts toward the probe’s evidence level: ' : 'Does not raise the probe’s evidence level: '}
          {run.rung_note}
        </p>
      ) : null}
    </>
  );
}

export function JudgeRunResults({ run, datasetNames }: Props) {
  const perSet = (run.metrics?.per_set ?? {}) as Record<string, JudgeSetResult>;
  const entries = Object.entries(perSet);
  if (entries.length === 0) return null;
  const sample = run.max_rows_per_set ?? (run.metrics?.max_rows_per_set as number | undefined);
  return (
    <div className="mt-2" data-testid="judge-run-results">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="py-1 pr-2">Set</th>
            <th className="py-1 pr-2" title="Area under the ROC (receiver operating characteristic) curve: 0.5 is chance, 1.0 is perfect separation">
              AUROC (95% interval)
            </th>
            <th className="py-1 pr-2" title="Share of present rows the judge called present, at its own answer">
              Judge recall
            </th>
            <th className="py-1 pr-2" title="Share of absent rows the judge called present">
              False positives
            </th>
            <th className="py-1 pr-2" title="The judge's recall at the false-positive rate the probe was calibrated at">
              Recall at the probe's rate
            </th>
            <th className="py-1">Rows</th>
          </tr>
        </thead>
        <tbody>
          {entries.map(([datasetId, result]) => (
            <tr key={datasetId} className="border-t border-slate-800 align-top" data-testid="judge-set-row">
              <td className="py-1 pr-2 text-slate-300">
                {result.set_name ?? datasetNames[datasetId] ?? datasetId}
                {result.out_of_distribution ? <span className="ml-1 text-[10px] text-emerald-300">unseen kind</span> : null}
              </td>
              {result.scored ? (
                <>
                  <td className="py-1 pr-2 tabular-nums text-slate-200">
                    {result.auroc?.toFixed(3)}
                    {result.ci ? ` (${result.ci.low.toFixed(3)}–${result.ci.high.toFixed(3)})` : ''}
                  </td>
                  <td className="py-1 pr-2 tabular-nums">{pct(result.decision?.recall)}</td>
                  <td className="py-1 pr-2 tabular-nums">{pct(result.decision?.false_positive_rate)}</td>
                  <td className="py-1 pr-2 tabular-nums">
                    {/* ⚠ NOT "recall 0" WHEN THE RATE IS UNREACHABLE (pmj_abc1e02877c0): a yes/no
                        judge has one operating point, and below it the only threshold is "call
                        nothing", which read as a judge that catches nothing. Said in words. */}
                    {result.at_probe_fpr
                      ? result.at_probe_fpr.reachable === false || result.at_probe_fpr.recall == null
                        ? (
                          <span className="text-slate-400" data-testid="judge-at-probe-unreachable">
                            {result.at_probe_fpr.reason ?? 'not reachable at this rate'}
                          </span>
                        )
                        : `${pct(result.at_probe_fpr.recall)} at ${pct(result.at_probe_fpr.target_fpr)}`
                      : '—'}
                  </td>
                  <td className="py-1 tabular-nums text-slate-400">
                    {(result.n_positive ?? 0) + (result.n_negative ?? 0)}
                    {result.n_dropped ? ` (${result.n_dropped} unreadable, left out)` : ''}
                  </td>
                </>
              ) : (
                <td colSpan={5} className="py-1 text-amber-300">{result.reason}</td>
              )}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mt-1 text-[11px] text-slate-500">
        {sample ? `Up to ${sample.toLocaleString()} rows per set, half of each label where possible.` : 'Every row of each set.'}
        {entries[0]?.[1]?.decision?.rule ? ` Answer: ${entries[0][1].decision!.rule}` : ''}
      </p>
    </div>
  );
}

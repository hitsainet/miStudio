/**
 * What still needs labeling in one extraction.
 *
 * This is the "what is left?" answer that did not exist. The OUTCOME of a
 * labeling attempt was recorded nowhere — it was inferred from `name`,
 * `category` and `label_source`, and a failure was written as a fake label with
 * `label_source` and `labeled_at` both set. So 16,824 failed features across the
 * estate looked finished, and one extraction carried 2,161 silent failures for
 * five months with nothing on any screen to say so.
 *
 * Deliberately not a chart. It answers three questions in one line — adjudicated,
 * failed, never attempted — and gets out of the way.
 */

import { useEffect, useState } from 'react';
import { AlertTriangle, ChevronRight, Loader } from 'lucide-react';

import { useLabelingStore } from '../../stores/labelingStore';
import type { LabelingCoverage } from '../../types/labeling';

interface LabelingCoverageStripProps {
  extractionId: string;
  /** Bumped by the parent to re-read after a labeling job finishes. */
  refreshKey?: number;
}

const numberFormat = new Intl.NumberFormat();

export function LabelingCoverageStrip({
  extractionId,
  refreshKey = 0,
}: LabelingCoverageStripProps) {
  const fetchCoverage = useLabelingStore((s) => s.fetchCoverage);
  const [coverage, setCoverage] = useState<LabelingCoverage | null>(null);
  const [failed, setFailed] = useState(false);
  const [reasonsOpen, setReasonsOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setFailed(false);
    fetchCoverage(extractionId)
      .then((result) => {
        if (!cancelled) setCoverage(result);
      })
      .catch(() => {
        // A coverage read that fails must not claim coverage. Rendering
        // nothing is honest; rendering zeros would read as "all done".
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
    };
  }, [extractionId, refreshKey, fetchCoverage]);

  if (failed) return null;

  if (!coverage) {
    return (
      <div className="flex items-center gap-1.5 text-xs text-slate-500 dark:text-slate-400">
        <Loader className="w-3 h-3 animate-spin" />
        <span>Reading labeling coverage&hellip;</span>
      </div>
    );
  }

  if (coverage.total === 0) return null;

  const attemptedFailures = coverage.by_status.failed ?? 0;
  const neverAttempted = coverage.by_status.pending ?? 0;
  const pct = (n: number) => `${(n / coverage.total) * 100}%`;

  return (
    <div className="flex flex-col gap-1 w-full">
      <div
        className="flex h-1.5 w-full overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800"
        role="img"
        aria-label={
          `${numberFormat.format(coverage.adjudicated)} of ` +
          `${numberFormat.format(coverage.total)} features adjudicated, ` +
          `${numberFormat.format(attemptedFailures)} failed, ` +
          `${numberFormat.format(neverAttempted)} never attempted`
        }
      >
        <div className="bg-emerald-500" style={{ width: pct(coverage.adjudicated) }} />
        <div className="bg-amber-500" style={{ width: pct(attemptedFailures) }} />
        <div className="bg-sky-500" style={{ width: pct(coverage.in_progress) }} />
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-0.5 text-xs">
        <span
          className="text-emerald-700 dark:text-emerald-400"
          title="A verdict exists — including an honest 'no coherent pattern', which is a result and is never redone."
        >
          {numberFormat.format(coverage.adjudicated)} adjudicated
        </span>
        {attemptedFailures > 0 && (
          <span
            className="text-amber-700 dark:text-amber-400"
            title="The judge ran and crashed. These are retried by Resume."
          >
            {numberFormat.format(attemptedFailures)} failed
          </span>
        )}
        {neverAttempted > 0 && (
          <span
            className="text-slate-600 dark:text-slate-400"
            title="The judge never ran on these."
          >
            {numberFormat.format(neverAttempted)} not attempted
          </span>
        )}
        {coverage.in_progress > 0 && (
          <span
            className="text-sky-700 dark:text-sky-400"
            title="Claimed by a labeling job running right now."
          >
            {numberFormat.format(coverage.in_progress)} in progress
          </span>
        )}
        {coverage.unclassified > 0 && (
          <span
            className="text-rose-700 dark:text-rose-400"
            title="In a state this build does not recognise. Reported rather than silently counted as finished."
          >
            {numberFormat.format(coverage.unclassified)} unrecognised
          </span>
        )}
      </div>

      {/*
        * WHY the failures failed. Collapsed by default — it is a diagnostic,
        * not a status — but one click from the count it explains.
        *
        * This is the answer to the question that decides whether the other
        * 14,536 are worth 32 GPU-hours, and until now it existed nowhere: the
        * reason was logged and discarded at the point of failure.
        */}
      {coverage.failure_reasons.length > 0 && (
        <div className="mt-0.5">
          <button
            type="button"
            onClick={() => setReasonsOpen((open) => !open)}
            aria-expanded={reasonsOpen}
            className="flex items-center gap-1 text-xs text-slate-600 dark:text-slate-400
                       hover:text-slate-900 dark:hover:text-slate-200 transition-colors"
          >
            <ChevronRight
              className={`w-3 h-3 transition-transform duration-200 ${
                reasonsOpen ? 'rotate-90' : ''
              }`}
            />
            <span>
              {reasonsOpen ? 'Hide' : 'Why'} {coverage.failure_reasons.length} failure
              {coverage.failure_reasons.length === 1 ? ' reason' : ' reasons'}
            </span>
          </button>

          {reasonsOpen && (
            <table className="mt-1 w-full text-xs">
              <tbody>
                {coverage.failure_reasons.map((r) => (
                  <tr key={r.reason} className="align-top">
                    <td className="pr-3 py-0.5 text-right tabular-nums
                                   text-amber-700 dark:text-amber-400 whitespace-nowrap">
                      {numberFormat.format(r.count)}
                    </td>
                    <td className="py-0.5 text-slate-700 dark:text-slate-300">
                      {r.reason}
                      {r.sample_feature_ids.length > 0 && (
                        <span
                          className="ml-2 text-slate-500 dark:text-slate-500"
                          title={
                            'Real features with this failure, so the count can be '
                            + `checked rather than believed: ${r.sample_feature_ids.join(', ')}`
                          }
                        >
                          e.g. {r.sample_feature_ids[0]}
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {coverage.caveat && (
        <div
          className="flex items-start gap-1.5 text-xs text-amber-700 dark:text-amber-400/90"
          title={coverage.caveat}
        >
          <AlertTriangle className="w-3 h-3 mt-0.5 shrink-0" />
          <span>
            {numberFormat.format(coverage.failures_without_a_recorded_reason)} failures
            carry no recorded reason
          </span>
        </div>
      )}
    </div>
  );
}

export default LabelingCoverageStrip;

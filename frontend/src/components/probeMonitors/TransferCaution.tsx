/**
 * The sentence that says a probe's absolute score does NOT transfer between distributions.
 *
 * ⚠ THE SERVER HAS ASSEMBLED THIS ON EVERY PROBE REPORT SINCE 032 AND NOTHING EVER RENDERED IT.
 * `ProbeReport` did not even declare the field, so the one warning that stands between a reader
 * and "this probe has AUROC 0.95, so it works here" was API- and MCP-only. It became load-bearing
 * the moment the threshold turned into a dial an operator can spin: a fast bar with no feedback
 * is an invitation to lower it until the probe fires on whatever is in front of you.
 *
 * It is measured, not asserted. On the first shipped probe the five evaluation sets' own 1%
 * thresholds spanned 24 points — wider than the shipped threshold itself — and that bar was above
 * every score `toolace_balanced` can produce while firing on 42.6% of `mental_health_balanced`'s
 * low-stakes rows.
 *
 * ⚠ ITS OWN COMPONENT, NOT INLINE IN THE PANEL. The panel is not rendered by any test — it is
 * read as raw source — so anything inline there can only be guarded by a scrape, and a scrape
 * that stops matching asserts nothing. This repo's recorded remedy is to extract the decision,
 * test it behaviourally, and assert the CALL. The prose is never composed here: `caution` is
 * server-written, because the thresholds it quotes are server-side measurements.
 */
import { AlertTriangle } from 'lucide-react';

import type { ThresholdTransfer } from '../../types/probeMonitor';

interface TransferCautionProps {
  transfer?: ThresholdTransfer | null;
}

export function TransferCaution({ transfer }: TransferCautionProps) {
  // Absent is not the same as quiet: a probe with one evaluation set has no spread to report, and
  // inventing a reassurance for it would be worse than saying nothing.
  if (!transfer) return null;
  const { caution, unreachable_sets: unreachable } = transfer;
  if (!caution && !unreachable?.length) return null;

  return (
    <div className="mt-2 space-y-1" data-testid="report-threshold-transfer">
      {caution ? (
        <p
          className="flex items-start gap-1 text-xs text-amber-300"
          data-testid="report-threshold-transfer-caution"
        >
          <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" aria-hidden="true" />
          {caution}
        </p>
      ) : null}
      {unreachable?.length ? (
        <p className="text-xs text-slate-400" data-testid="report-unreachable-sets">
          The threshold is above every score on {unreachable.join(', ')} — recall there is zero
          however well the probe ranks on it.
        </p>
      ) : null}
    </div>
  );
}

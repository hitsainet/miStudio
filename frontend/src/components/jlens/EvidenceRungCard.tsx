/**
 * Evidence rung for a readout (BR-019, FPRD §3.8).
 *
 * A readout is the WEAKEST evidence this product produces and the easiest to
 * describe in causal language by accident. The card names the rung and, more
 * importantly, names what would raise it — labelling the current rung alone
 * tells a reader they are low on a ladder without telling them how to climb.
 */

import { AlertTriangle } from 'lucide-react';

export function EvidenceRungCard() {
  return (
    <div>
      <div className="mb-1.5 flex items-center gap-1.5 text-xs font-medium text-slate-600 dark:text-slate-400">
        <AlertTriangle className="h-3.5 w-3.5 text-amber-500" /> Evidence rung
      </div>
      <div className="rounded border border-amber-300 bg-amber-50 px-2 py-1.5 dark:border-amber-900 dark:bg-amber-950">
        <div className="text-[11px] font-medium text-amber-800 dark:text-amber-200">
          Rung 0 · Readout
        </div>
        <p className="mt-0.5 text-[10px] leading-snug text-amber-700 dark:text-amber-200/70">
          A concept appearing in a readout is not a causal claim. Run a
          coordinate swap with a matched control to raise the rung.
        </p>
      </div>
    </div>
  );
}

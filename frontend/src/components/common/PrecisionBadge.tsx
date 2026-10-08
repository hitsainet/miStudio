/**
 * The precision an artifact was built at — and whether that is RECORDED or only inferred.
 *
 * ⚠ AN INFERRED VALUE IS NEVER SHOWN AS A RECORDED ONE. Before 2026-10-03 every miStudio load path
 * ran float16 and wrote nothing down; the backend labels those artifacts "float16 (inferred)". A
 * badge that dropped the word would turn a deduction into a fact on screen, which is exactly what
 * the database refuses to do.
 *
 * `servedAs`, when given, is the precision the model loads at now. A difference is called out,
 * because an SAE or probe built at one precision reads a different distribution at another.
 */
import type { PrecisionLabel } from '../../types/precision';

interface PrecisionBadgeProps {
  label?: PrecisionLabel | null;
  /** The precision this model loads at today, when known (e.g. the checkpoint's native dtype). */
  servedAs?: string | null;
}

export function PrecisionBadge({ label, servedAs }: PrecisionBadgeProps) {
  if (!label || label.source === 'unknown' || !label.value) {
    return (
      <span className="text-xs text-slate-500" data-testid="precision-badge" data-source="unknown">
        precision not recorded
      </span>
    );
  }
  const inferred = label.source === 'inferred';
  const differs = Boolean(servedAs) && servedAs !== label.value;
  const title = [
    inferred ? label.note ?? 'inferred, not recorded' : 'recorded when this was built',
    differs ? `This model now loads at ${servedAs}; rebuild to match.` : null,
  ]
    .filter(Boolean)
    .join(' ');
  return (
    <span
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 font-mono text-xs ${
        differs
          ? 'bg-amber-500/15 text-amber-700 dark:text-amber-300'
          : 'bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300'
      }`}
      title={title}
      data-testid="precision-badge"
      data-source={label.source}
    >
      {label.value}
      {inferred ? <span className="font-sans italic">(inferred)</span> : null}
      {differs ? <span className="font-sans">· now {servedAs}</span> : null}
    </span>
  );
}

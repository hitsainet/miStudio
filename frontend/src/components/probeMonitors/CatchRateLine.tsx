/**
 * The catch rate rung 2 needs (ENH-006 D84): on every out-of-distribution set, the share of positives the probe
 * fires on AT ITS SHIPPED BAR — "On unseen tasks (…): catches 10% of positives at the shipped bar (41 of 410;
 * needs ≥ 50%)". A probe that ranks well but rarely fires is not a monitor.
 *
 * Every sentence is the server's (`recompute_rung` → `catch_rate` on the report); this only lays them out.
 */
import type { CatchRateCheck } from '../../types/probeMonitor';

export function CatchRateLine({ check }: { check: CatchRateCheck | null }) {
  if (!check || check.status === 'no_sets') return null;
  const tone = check.passed ? 'text-slate-400' : 'text-amber-300';
  return (
    <div className="mt-2 text-xs" data-testid="catch-rate">
      {check.sets.map((set) => (
        <p key={set.name} className={set.meets ? 'text-slate-400' : 'text-amber-300'} data-testid="catch-rate-set">
          {set.sentence}
        </p>
      ))}
      {!check.passed ? <p className={tone}>Rung 2 not reached: {check.reason}</p> : null}
    </div>
  );
}

/**
 * How many negatives a probe's bar was cut from, and what its target allows (Feature 040 S2, FR-19):
 * "1% of 540 = 5 rows", flagged when below the floor. Every value is the server's.
 */
import type { CalibrationSize } from '../../types/probeMonitor';

export function CalibrationSizeLine({ size }: { size: CalibrationSize | null }) {
  if (!size || size.n_negatives == null) return null;
  return (
    <p className="mt-2 text-xs text-slate-400" data-testid="calibration-size">
      Calibration: {size.sentence ?? `${size.n_negatives} negatives`}
      {size.n_negatives_source === 'array_length' ? ' (counted from the stored negatives)' : ''}
      {size.few_negatives ? (
        <span className="text-amber-300">
          {' '}— fewer than the floor of {size.floor.toLocaleString()} ({size.floor_source === 'behaviour' ? "the behaviour's own" : 'the deployment setting'}); criteria over it are not assessable
        </span>
      ) : null}
    </p>
  );
}

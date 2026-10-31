/**
 * The layer × pooling sweep, as a grid (032 FR-6).
 *
 * ⚠ IT SHOWS THE WHOLE GRID, INCLUDING THE MARGIN. A sweep rendered as only its winner
 * cannot be read for a near-tie — and a near-tie is exactly when the chosen layer is
 * arbitrary. An unscored cell shows "—", never 0: a cell that could not be scored and a
 * cell that scored zero are different facts.
 */
import type { ProbeLayerSelection } from '../../types/probeMonitor';

interface SweepGridProps {
  selection: ProbeLayerSelection | null;
}

/** Green where the probe separates, slate where it does not. A tone is not a claim. */
function tone(value: number | null): string {
  if (value === null) return 'bg-slate-800 text-slate-500';
  if (value >= 0.9) return 'bg-emerald-900/60 text-emerald-200';
  if (value >= 0.75) return 'bg-emerald-900/30 text-emerald-300';
  if (value >= 0.6) return 'bg-sky-900/30 text-sky-300';
  return 'bg-slate-800 text-slate-400';
}

/**
 * The margin, never rounded to a flat `0.0000` while it is non-zero.
 *
 * ⚠ SEEN ON THE FIRST REAL SWEEP. Llama-3.1-8B chose L11 at 0.9984371 over L16 at
 * 0.9983898 — a margin of 0.0000473, which `toFixed(4)` rendered as `0.0000`. Beside the
 * words "a near-tie" that reads as an exact tie, i.e. as two layers the sweep could not
 * separate at all, when in fact it separated them and the difference is simply tiny.
 * "Rounded to nothing" and "nothing" are different facts, and the smaller one is the one
 * a reader would over-interpret.
 */
export function formatMargin(margin: number): string {
  if (margin === 0) return '0.0000';
  if (Math.abs(margin) < 0.0001) return `${margin < 0 ? '>-' : '<'}0.0001`;
  return margin.toFixed(4);
}

export function SweepGrid({ selection }: SweepGridProps) {
  if (!selection || selection.grid.length === 0) {
    return (
      <p className="text-sm text-slate-400" data-testid="sweep-empty">
        No layer sweep has been recorded for this run.
      </p>
    );
  }
  const layers = Array.from(new Set(selection.grid.map((cell) => cell.layer))).sort(
    (a, b) => a - b
  );
  const poolings = selection.poolings ?? Array.from(new Set(selection.grid.map((c) => c.pooling)));
  const lookup = new Map(selection.grid.map((cell) => [`${cell.layer}:${cell.pooling}`, cell]));
  const chosen = new Set(selection.chosen);

  return (
    <div data-testid="sweep-grid">
      <div className="overflow-x-auto">
        <table className="text-xs">
          <thead>
            <tr>
              <th className="px-2 py-1 text-left text-slate-400">pooling</th>
              {layers.map((layer) => (
                <th
                  key={layer}
                  className={`px-2 py-1 text-center ${
                    chosen.has(layer) ? 'text-emerald-300' : 'text-slate-400'
                  }`}
                >
                  L{layer}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {poolings.map((pooling) => (
              <tr key={pooling}>
                <td className="px-2 py-1 text-slate-300">{pooling}</td>
                {layers.map((layer) => {
                  const cell = lookup.get(`${layer}:${pooling}`);
                  const value = cell?.val_auroc ?? null;
                  return (
                    <td
                      key={`${layer}:${pooling}`}
                      className={`px-2 py-1 text-center tabular-nums ${tone(value)} ${
                        chosen.has(layer) ? 'ring-1 ring-emerald-600' : ''
                      }`}
                      title={
                        cell
                          ? `layer ${layer}, ${pooling}: ${
                              value === null ? 'not scored' : value.toFixed(4)
                            } (n=${cell.n_train}/${cell.n_val})`
                          : undefined
                      }
                    >
                      {value === null ? '—' : value.toFixed(3)}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="mt-2 text-xs text-slate-400" data-testid="sweep-margin">
        Chosen: {selection.chosen.map((l) => `L${l}`).join(', ') || '—'}
        {selection.margin !== null && selection.margin !== undefined ? (
          <>
            {' · '}
            margin over the runner-up{' '}
            <span className="tabular-nums">{formatMargin(selection.margin)}</span>
            {selection.margin < 0.01 ? (
              // SAID OUT LOUD. A layer chosen by a hundredth of AUROC is an arbitrary
              // choice, and a reader who does not know that will over-read the layer.
              <span className="ml-1 text-amber-300">
                — a near-tie, so the chosen layer is close to arbitrary
              </span>
            ) : null}
          </>
        ) : null}
      </p>
    </div>
  );
}

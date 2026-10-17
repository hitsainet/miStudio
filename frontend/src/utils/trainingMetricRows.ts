/**
 * Which metric rows a training's history is drawn from (review R1-C, 2026-09-15).
 *
 * One logged step writes several `training_metrics` rows: the aggregate over every
 * SAE (`layer_idx` null), one per SAE (`layer_idx` = the layer), and held-out rows
 * (`layer_idx` = -1 - layer). Runs from before 2026-09-15 also carry spliced-CE rows
 * (`layer_idx` = -1000 - layer) whose `loss` column holds a cross-entropy.
 *
 * The card asked for the last 20 rows and deduplicated them by step, keeping
 * whichever row of a step came first — so its FVU chart and history line could show
 * one layer's value, or a held-out value, as the run's. The endpoint did not even
 * serve `layer_idx`, so the rows could not be told apart.
 */

/**
 * Logged steps the card's history shows. The card asks the endpoint for aggregate rows
 * only (`aggregate_only`, review R1-A A5): a raw window is shared by one row per SAE and
 * one held-out row per SAE for EVERY hook type, so the 400 raw rows it used to request
 * held 12 steps of a 5-layer, 3-hook run.
 */
/**
 * How many AGGREGATE metric rows the card charts.
 *
 * Was 20, which is not a trend — it is a keyhole. One aggregate row is written
 * per `log_interval` (250 by default), so twenty points span 5,000 steps: about
 * fourteen minutes of a 2h45m, 60,000-step run. L0 falling 825 -> 113 -> 49 over
 * such a run showed on the card as 115 -> 113, and the convergence shape that
 * actually tells you whether a run is healthy was never visible.
 *
 * A full 60,000-step run is 240 aggregate rows, and the endpoint accepts a limit
 * up to 10,000 with `aggregate_only` filtering server-side, so the whole history
 * is one cheap request. 1,000 covers runs to 250,000 steps at the default
 * interval and still bounds the response.
 */
export const METRICS_WINDOW_STEPS = 1000;

/**
 * The aggregate rows only, in their original order. Kept beside `aggregate_only` as a
 * second guard. It needs no hook type: every per-SAE and held-out row, of every hook,
 * carries a layer index.
 */
export function aggregateMetricRows<T extends { layer_idx?: number | null }>(rows: T[]): T[] {
  return rows.filter((row) => row.layer_idx === null || row.layer_idx === undefined);
}

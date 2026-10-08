/**
 * TrainingEvaluationPanel — what a training's SAEs cost the model (remediation item 6).
 *
 * Shows `training.evaluation`: per layer the cross-entropy with the SAE's
 * reconstruction spliced in, loss recovered against MEAN ablation (the headline)
 * and against zero ablation (reference only — on these models zeroing a residual
 * layer makes the model emit a uniform distribution, so almost any SAE scores
 * near 1 against it), KL from the untouched model, L0 and FVU. Measured on
 * blocks the training never read.
 *
 * "Evaluate" queues the GPU job behind POST /trainings/{id}/evaluate; while it is
 * pending or running the panel refreshes the training until a result arrives.
 */

import { useEffect, useState } from 'react';
import { Gauge, RefreshCw } from 'lucide-react';
import type { Training, TrainingEvaluation, TrainingEvaluationLayer } from '../../types/training';
import { COMPONENTS } from '../../config/brand';

interface TrainingEvaluationPanelProps {
  training: Training;
  onEvaluate?: (trainingId: string, opts?: { force?: boolean }) => Promise<void>;
  onRefresh?: (trainingId: string, silent?: boolean) => Promise<void>;
  /** Poll interval while an evaluation is pending or running. */
  pollMs?: number;
  /**
   * How long an evaluation may sit pending or running without a new write before
   * the panel offers to force a new one. The backend reaps an abandoned RUNNING
   * evaluation on its own; a PENDING one may be queued behind a long job, so only
   * the operator can decide it is lost.
   */
  staleMs?: number;
}

const ACTIVE = new Set(['pending', 'running']);

/** 15 minutes: fifteen missed heartbeats for a running evaluation. */
export const EVALUATION_STALE_MS = 15 * 60 * 1000;

/** Milliseconds since the evaluation last wrote its record, or null when it carries no time. */
export function evaluationAgeMs(evaluation: TrainingEvaluation | null, now: number = Date.now()): number | null {
  const stamp = evaluation?.updated_at ?? evaluation?.requested_at ?? evaluation?.started_at;
  if (!stamp) return null;
  const parsed = Date.parse(stamp);
  return Number.isFinite(parsed) ? Math.max(0, now - parsed) : null;
}

function num(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function pct(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '—';
}

function signed(value: number | null | undefined): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return `${value >= 0 ? '+' : ''}${value.toFixed(3)}`;
}

const STATUS_STYLE: Record<string, string> = {
  completed: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10',
  failed: 'text-red-400 border-red-500/40 bg-red-500/10',
  skipped: 'text-slate-400 border-slate-500/40 bg-slate-500/10',
  cancelled: 'text-slate-400 border-slate-500/40 bg-slate-500/10',
  pending: 'text-amber-400 border-amber-500/40 bg-amber-500/10',
  running: 'text-amber-400 border-amber-500/40 bg-amber-500/10',
};

/**
 * What each recorded stage is called in the panel.
 *
 * The backend writes `evaluation.progress.stage` as it goes and nothing read it,
 * so a running evaluation said only "running…" however long it took. Loading the
 * base model is the slow part — minutes on a large model — and naming it is the
 * difference between "working" and "stuck".
 *
 * An unknown stage falls back to its raw value rather than being hidden: a stage
 * the backend adds later should still appear, not vanish because this map is
 * behind. `stage` is typed `(string & {})` precisely so that can happen.
 */
const STAGE_LABEL: Record<string, string> = {
  loading_model: 'loading the base model',
  means: 'computing means',
  cross_entropy: 'measuring cross-entropy',
};

function LayerRow({ layer }: { layer: TrainingEvaluationLayer }) {
  return (
    <tr className="border-t border-slate-200 dark:border-slate-800" data-testid={`evaluation-layer-${layer.layer}`}>
      <td className="py-1 pr-2 text-slate-700 dark:text-slate-300">L{layer.layer}</td>
      <td className="py-1 pr-2 font-mono">
        {num(layer.ce_spliced)} <span className="text-slate-500">({signed(layer.ce_delta)})</span>
      </td>
      <td className="py-1 pr-2 font-mono text-emerald-400 font-semibold">{pct(layer.loss_recovered_vs_mean)}</td>
      <td className="py-1 pr-2 font-mono text-slate-500">{pct(layer.loss_recovered_vs_zero)}</td>
      <td className="py-1 pr-2 font-mono">{num(layer.kl)}</td>
      <td className="py-1 pr-2 font-mono">{num(layer.l0, 1)}</td>
      <td className="py-1 font-mono text-amber-400">{num(layer.fvu_centred)}</td>
    </tr>
  );
}

function Result({ evaluation }: { evaluation: TrainingEvaluation }) {
  const layers = evaluation.layers ?? [];
  const all = evaluation.all_layers_spliced;
  const sources = evaluation.sources ?? [];
  return (
    <div className="space-y-1.5">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-left text-slate-500">
              <th className="pr-2 font-medium">Layer</th>
              <th className="pr-2 font-medium" title="Cross-entropy with the SAE reconstruction spliced in (change from the untouched model)">
                CE spliced (Δ)
              </th>
              <th className="pr-2 font-medium" title="(mean-ablated − spliced) / (mean-ablated − base): the headline">
                Recovered vs mean
              </th>
              <th className="pr-2 font-medium" title={evaluation.zero_ablation_note ?? 'Zero ablation is a uniform-output floor, for reference only'}>
                vs zero*
              </th>
              <th className="pr-2 font-medium" title="KL(untouched model ‖ spliced), nats per token">KL</th>
              <th className="pr-2 font-medium" title="Active latents per token">L0</th>
              <th className="font-medium" title="Per-dimension-centred FVU on these tokens">FVU</th>
            </tr>
          </thead>
          <tbody>
            {layers.map((layer) => (
              <LayerRow key={`${layer.layer}-${layer.hook_type ?? 'residual'}`} layer={layer} />
            ))}
            {all && layers.length > 1 && (
              <tr className="border-t border-slate-300 dark:border-slate-700" data-testid="evaluation-all-layers">
                <td className="py-1 pr-2 text-slate-700 dark:text-slate-300">All</td>
                <td className="py-1 pr-2 font-mono">
                  {num(all.ce)} <span className="text-slate-500">({signed(all.ce_delta)})</span>
                </td>
                <td className="py-1 pr-2" />
                <td className="py-1 pr-2" />
                <td className="py-1 pr-2 font-mono">{num(all.kl)}</td>
                <td className="py-1 pr-2" />
                <td className="py-1" />
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="text-xs text-slate-500">
        Base CE {num(evaluation.ce_base)} over {(evaluation.tokens ?? 0).toLocaleString()} tokens the training never
        read, from {sources.length} source{sources.length === 1 ? '' : 's'}.
      </p>
      <p className="text-xs text-slate-500" data-testid="evaluation-zero-note">
        * Zero ablation is a uniform-output floor on these models; read recovered vs mean.
      </p>
    </div>
  );
}

export function TrainingEvaluationPanel({
  training,
  onEvaluate,
  onRefresh,
  pollMs = 10_000,
  staleMs = EVALUATION_STALE_MS,
}: TrainingEvaluationPanelProps) {
  const evaluation = training.evaluation ?? null;
  const status = evaluation?.status ?? null;
  const active = status !== null && ACTIVE.has(status);
  const age = active ? evaluationAgeMs(evaluation) : null;
  // A pending or running record that has not been written for a while may belong
  // to a worker that is gone. The button stays disabled for a live one; a stale
  // one gets a forced re-run, which the endpoint otherwise refuses with 409.
  const stale = active && age !== null && age > staleMs;
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!active || !onRefresh) return undefined;
    const timer = setInterval(() => {
      onRefresh(training.id, true).catch(() => undefined);
    }, pollMs);
    return () => clearInterval(timer);
  }, [active, onRefresh, training.id, pollMs]);

  const handleEvaluate = async (force = false) => {
    if (!onEvaluate) return;
    setSubmitting(true);
    setError(null);
    try {
      await (force ? onEvaluate(training.id, { force: true }) : onEvaluate(training.id));
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Could not start the evaluation');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="rounded-lg border border-slate-200 dark:border-slate-800 p-2 space-y-1.5" data-testid="training-evaluation">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 text-xs font-medium text-slate-700 dark:text-slate-300">
          <Gauge className="w-3.5 h-3.5" />
          <span>Model-cost evaluation</span>
          {status && (
            <span className={`px-1.5 py-0.5 rounded border ${STATUS_STYLE[status] ?? STATUS_STYLE.skipped}`}>
              {status}
            </span>
          )}
        </div>
        {onEvaluate && (
          <div className="flex items-center gap-1">
            {stale && (
              <button
                type="button"
                onClick={() => handleEvaluate(true)}
                disabled={submitting}
                data-testid="evaluation-force"
                title="Queue a new evaluation even though this one is still recorded as pending or running"
                className={`flex items-center gap-1 text-xs rounded-lg px-2 py-0.5 ${COMPONENTS.button.secondary} disabled:opacity-50`}
              >
                <RefreshCw className="w-3 h-3" />
                Force re-run
              </button>
            )}
            <button
              type="button"
              // Not `onClick={handleEvaluate}`: the click event would arrive as `force`.
              onClick={() => handleEvaluate()}
              disabled={active || submitting}
              className={`flex items-center gap-1 text-xs rounded-lg px-2 py-0.5 ${COMPONENTS.button.secondary} disabled:opacity-50`}
            >
              <RefreshCw className="w-3 h-3" />
              {evaluation ? 'Re-evaluate' : 'Evaluate'}
            </button>
          </div>
        )}
      </div>

      {!evaluation && (
        <p className="text-xs text-slate-500">
          Not evaluated. Evaluating splices each SAE into the model on text the training never read.
        </p>
      )}
      {active && (
        <p className="text-xs text-amber-400">
          Evaluation {status}
          {/* THE STAGE, WHICH THE RECORD HAS CARRIED ALL ALONG. The backend writes
              `progress.stage` as it goes — loading_model, means, cross_entropy — and
              nothing read it, so a long evaluation showed "running…" and no more.
              Loading the base model is the slow part, so saying which stage it is in
              is the difference between "working" and "stuck". */}
          {evaluation?.progress?.stage && (
            <span data-testid="evaluation-stage">
              {': '}{STAGE_LABEL[evaluation.progress.stage] ?? evaluation.progress.stage}
            </span>
          )}
          {/* Counts only when BOTH are present: batches_done alone has no denominator
              and would read as a total. */}
          {typeof evaluation?.progress?.batches_done === 'number' &&
            typeof evaluation?.progress?.batches === 'number' && (
              <span data-testid="evaluation-batches">
                {' '}({evaluation.progress.batches_done.toLocaleString()} of{' '}
                {evaluation.progress.batches.toLocaleString()} batches)
              </span>
            )}
          {'…'}
          {stale && age !== null && (
            <span data-testid="evaluation-stale">
              {' '}No update for {Math.round(age / 60_000)} min; its worker may be gone.
            </span>
          )}
        </p>
      )}
      {(status === 'failed' || status === 'skipped' || status === 'cancelled') && evaluation?.reason && (
        <p className={`text-xs ${status === 'failed' ? 'text-red-400' : 'text-slate-500'}`}>{evaluation.reason}</p>
      )}
      {status === 'completed' && evaluation && <Result evaluation={evaluation} />}
      {error && <p className="text-xs text-red-400">{error}</p>}
    </div>
  );
}

export default TrainingEvaluationPanel;

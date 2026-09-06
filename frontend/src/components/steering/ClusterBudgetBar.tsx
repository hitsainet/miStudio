/**
 * ClusterBudgetBar (Feature 013 + 015) — shows the cluster's computed strength
 * budget(s), consumption by the current member strengths, the resultant-norm
 * gain G, and any coherence flags. Rendered only while a cluster allocation
 * governs the selection.
 *
 * Feature 015: when the selection spans multiple layers, `layerBudgets` holds
 * one budget per layer and the bar renders one row per layer (each with a layer
 * chip + its own B/λ consumption). A single-layer selection renders EXACTLY as
 * before 015 — the single mirrored `clusterBudget` drives an identical bar.
 */

import { AlertTriangle, Gauge, Layers } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';
import type { ClusterBudget, SelectedFeature } from '../../types/steering';

const FLAG_COPY: Record<string, string> = {
  cancellation: 'Members partially cancel each other — check the flagged pair',
  default_budget: 'No activation frequencies known — conservative default budget',
  cap_bound: 'Budget capped at the sum of solo optima',
  approximate: 'Decoder unavailable — constant-budget approximation (G=1)',
  // 015 R1 #4: these were silently dropped. nonunit_decoder in particular
  // means the budget law is EXTRAPOLATING and the user should know.
  nonunit_decoder: 'Decoder columns are not unit-norm — the budget law extrapolates',
  low_cohesion: 'Cluster failed the cohesion gate — strengths are unreliable',
  grain_limited: 'Allocation limited by strength granularity',
  uniform_weights: 'Similarity unavailable — equal weights across members',
  inactive_member: 'A member rarely activates — its contribution is uncertain',
};

/**
 * One budget row: consumption against B, gain G, coherence flags. `used` is the
 * sum of |strength| across the features governed by THIS budget.
 */
function BudgetRow({
  budget,
  used,
  layer,
}: {
  budget: ClusterBudget;
  used: number;
  layer?: number;
}) {
  const over = used > budget.B + 0.05;
  const pct = budget.B > 0 ? Math.min(100, (used / budget.B) * 100) : 0;
  const warnings = budget.flags.filter((f) => FLAG_COPY[f]);

  return (
    <div className="rounded-lg border border-slate-300 dark:border-slate-700/60 bg-slate-100 dark:bg-slate-900/60 px-3 py-2">
      <div className="flex items-center justify-between text-[11px] mb-1">
        <span className="flex items-center gap-1 text-slate-600 dark:text-slate-400">
          {layer != null ? (
            <span
              className="inline-flex items-center gap-0.5 rounded bg-slate-100 dark:bg-slate-800 px-1 py-px text-[10px] text-cyan-300"
              title={`Budget for layer ${layer}`}
            >
              <Layers className="w-2.5 h-2.5" />
              L{layer}
            </span>
          ) : (
            <Gauge className="w-3 h-3 text-cyan-400" />
          )}
          {layer != null ? 'budget' : 'Cluster budget'}
        </span>
        <span className={`font-mono ${over ? 'text-amber-400' : 'text-slate-600 dark:text-slate-400'}`}>
          {used.toFixed(1)} / {budget.B.toFixed(1)}
          <span className="text-slate-600 ml-2" title="Resultant-norm gain of the injected direction">
            G {budget.G.toFixed(2)}
          </span>
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-white dark:bg-slate-800 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${over ? 'bg-amber-500' : 'bg-cyan-500'}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      {warnings.length > 0 && (
        <div className="mt-1.5 space-y-0.5">
          {warnings.map((f) => (
            <p key={f} className="flex items-center gap-1 text-[10px] text-amber-400/90">
              <AlertTriangle className="w-2.5 h-2.5 shrink-0" />
              {FLAG_COPY[f]}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

export function ClusterBudgetBar() {
  const { clusterBudget, layerBudgets, clusterNotice, selectedFeatures } = useSteeringStore();

  // Multi-layer (Feature 015): one row per layer. clusterBudget is null in this
  // mode; layerBudgets carries a budget per distinct layer.
  const layerKeys = layerBudgets ? Object.keys(layerBudgets).map(Number).sort((a, b) => a - b) : [];
  const isMultiLayer = !clusterBudget && layerKeys.length > 1;

  if (isMultiLayer) {
    const usedByLayer = (layer: number) =>
      selectedFeatures
        .filter((f: SelectedFeature) => f.layer === layer)
        .reduce((s, f) => s + Math.abs(f.strength), 0);
    return (
      <div className="px-4 pb-2 space-y-1.5">
        {layerKeys.map((layer) => (
          <BudgetRow
            key={layer}
            budget={layerBudgets![layer]}
            used={usedByLayer(layer)}
            layer={layer}
          />
        ))}
      </div>
    );
  }

  // Gated cluster (e.g. low cohesion): no governing budget, just the notice.
  if (!clusterBudget) {
    if (!clusterNotice) return null;
    return (
      <div className="px-4 pb-2">
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 px-3 py-2">
          <p className="flex items-center gap-1 text-[10px] text-amber-400/90">
            <AlertTriangle className="w-2.5 h-2.5 shrink-0" />
            {clusterNotice}
          </p>
        </div>
      </div>
    );
  }

  // Single-layer (Feature 013): byte-identical to the pre-015 bar.
  const used = selectedFeatures.reduce((s, f) => s + Math.abs(f.strength), 0);
  return (
    <div className="px-4 pb-2">
      <BudgetRow budget={clusterBudget} used={used} />
    </div>
  );
}

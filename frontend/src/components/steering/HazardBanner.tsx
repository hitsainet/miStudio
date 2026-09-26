/**
 * HazardBanner (Feature 015, BR-024) — cross-layer steering hazard warning.
 *
 * Rendered ABOVE the Generate button when a multi-layer cluster allocation
 * returned `hazards`. Lists each hazard pair (upstream → downstream, layer
 * -labeled) with its evidence.
 *
 * COPY DISCIPLINE (IDL-35): the banner renders the backend `evidence` string as
 * its source of truth, and NEVER upgrades a heuristic to a causal claim.
 *   - `validated:ES=X` (rung ≥ 2, quantified) → "compounding (validated, ES=0.8)"
 *   - `heuristic:weight_prior=X` (labeled prior) → "possible compounding
 *     (heuristic — weight prior)". The word "causal" never appears for these.
 *
 * The banner is a WARNING ONLY — it never mutates the config. It is dismissible
 * per run; the dismissal RESETS whenever the selection changes (same
 * selection-clearing discipline as clusterContext), so a stale dismissal can
 * never hide a hazard for a different selection.
 */

import { useMemo, useState } from 'react';
import { AlertTriangle, X } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';
import type { Hazard } from '../../types/steering';

/** Parse a backend evidence string into a display-safe, non-causal phrasing. */
function describeHazard(h: Hazard): { headline: string; detail: string } {
  const kind = h.type === 'compounding' ? 'compounding' : 'cancellation';
  const validated = h.evidence.startsWith('validated:');
  if (validated) {
    // Quantified, rung ≥ 2 — the only case allowed to read as established.
    const es =
      h.quantified_effect != null
        ? h.quantified_effect.toFixed(2)
        : (h.evidence.split('ES=')[1] ?? '').trim();
    const esPart = es ? `, ES=${es}` : '';
    return {
      headline: `${kind} (validated${esPart})`,
      detail: h.evidence,
    };
  }
  // Heuristic weight prior — must NEVER read as causal.
  const prior = (h.evidence.split('weight_prior=')[1] ?? '').trim();
  const priorPart = prior ? ` — weight prior ${prior}` : ' — weight prior';
  return {
    headline: `possible ${kind} (heuristic${priorPart})`,
    detail: h.evidence,
  };
}

export function HazardBanner() {
  const hazards = useSteeringStore((s) => s.hazards);
  const selectedFeatures = useSteeringStore((s) => s.selectedFeatures);

  // Selection signature: any add/remove/reorder/strength-source change to the
  // governed set produces a new signature, which resets the dismissal.
  const signature = useMemo(
    () => selectedFeatures.map((f) => `${f.instance_id}:${f.layer}:${f.feature_idx}`).join('|'),
    [selectedFeatures],
  );
  const [dismissedFor, setDismissedFor] = useState<string | null>(null);

  if (!hazards || hazards.length === 0) return null;
  if (dismissedFor === signature) return null;

  return (
    <div className="px-4 pb-2">
      <div className="rounded-lg border border-amber-500/40 bg-amber-500/5 px-3 py-2">
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-1.5 text-[11px] font-medium text-amber-300">
            <AlertTriangle className="w-3 h-3 shrink-0" />
            Cross-layer steering {hazards.length === 1 ? 'hazard' : `hazards (${hazards.length})`}
          </div>
          <button
            type="button"
            onClick={() => setDismissedFor(signature)}
            className="p-0.5 -m-0.5 rounded text-amber-400/70 hover:text-amber-200 hover:bg-amber-500/10 transition-colors shrink-0"
            title="Dismiss for this selection"
            aria-label="Dismiss hazard warning"
          >
            <X className="w-3 h-3" />
          </button>
        </div>
        <ul className="mt-1.5 space-y-1">
          {hazards.map((h, i) => {
            const { headline, detail } = describeHazard(h);
            return (
              <li
                key={`${h.up.layer}-${h.up.feature_idx}-${h.down.layer}-${h.down.feature_idx}-${i}`}
                className="text-[10px] text-amber-200/90"
              >
                <span className="font-mono text-amber-300">
                  L{h.up.layer} #{h.up.feature_idx}
                </span>
                <span className="mx-1 text-amber-400/60">→</span>
                <span className="font-mono text-amber-300">
                  L{h.down.layer} #{h.down.feature_idx}
                </span>
                {/* Human copy inline; the raw evidence string is a tooltip
                    only — it read as leaked debug text next to the polished
                    phrase in a demo (015 R1 PROD-2). */}
                <span className="ml-1.5 text-amber-200/80" title={detail}>{headline}</span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

/**
 * AppliedFeaturesSummary — trust-by-inspection surface for Blended results
 * (Feature 012 + 015).
 *
 * Renders the server-returned `features_applied` list from a combined run so
 * users can verify that EVERY cluster member contributed its assigned
 * strength — answering the "is it really combining all of them?" doubt with
 * server truth rather than request state. Collapsed by default.
 *
 * Feature 015: members are GROUPED BY LAYER, and each member shows the SAE it
 * steered through (`features_applied[].sae_id` — server truth for which SAE
 * actually applied it). Single-layer runs render a single group (visually the
 * same flat list as before, under one layer heading).
 */

import { useState } from 'react';
import { ChevronDown, ChevronRight, CheckCircle2, Layers } from 'lucide-react';
import { CombinedFeatureApplied, FEATURE_COLORS } from '../../types/steering';

interface AppliedFeaturesSummaryProps {
  applied: CombinedFeatureApplied[];
}

function MemberChip({ f, i }: { f: CombinedFeatureApplied; i: number }) {
  // Server data: fall back to teal if an unknown color name arrives.
  const colors = FEATURE_COLORS[f.color] ?? FEATURE_COLORS.teal;
  return (
    <span
      key={`${f.feature_idx}-${i}`}
      className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[11px] border ${colors.border} ${colors.light} ${colors.text}`}
      title={
        f.sae_id
          ? `${f.label ? `${f.label} · ` : ''}steered through SAE ${f.sae_id}`
          : f.label || undefined
      }
    >
      #{f.feature_idx}
      {f.label && <span className="text-slate-400 max-w-[10rem] truncate">{f.label}</span>}
      <span className="font-mono">
        @ {f.strength > 0 ? '+' : ''}
        {f.strength}
      </span>
      {f.sae_id && (
        <span className="text-slate-500 max-w-[7rem] truncate font-mono" title={`SAE ${f.sae_id}`}>
          {f.sae_id}
        </span>
      )}
    </span>
  );
}

export function AppliedFeaturesSummary({ applied }: AppliedFeaturesSummaryProps) {
  const [open, setOpen] = useState(false);

  if (!applied || applied.length === 0) return null;

  // Group by layer, preserving first-seen layer order (Feature 015).
  const byLayer = new Map<number, CombinedFeatureApplied[]>();
  for (const f of applied) {
    const arr = byLayer.get(f.layer);
    if (arr) arr.push(f);
    else byLayer.set(f.layer, [f]);
  }
  const layers = Array.from(byLayer.keys()).sort((a, b) => a - b);

  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-300 transition-colors"
        title="Every feature the server applied in this blended generation"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <CheckCircle2 className="w-3 h-3 text-emerald-500" />
        Applied features ({applied.length})
      </button>

      {open && (
        <div className="mt-1.5 pl-4 space-y-2">
          {layers.map((layer) => (
            <div key={layer}>
              <div className="flex items-center gap-1 text-[10px] text-cyan-300 mb-1">
                <Layers className="w-2.5 h-2.5" />
                Layer {layer}
                <span className="text-slate-600">
                  · {byLayer.get(layer)!.length} member
                  {byLayer.get(layer)!.length !== 1 ? 's' : ''}
                </span>
              </div>
              <div className="flex flex-wrap gap-1.5">
                {byLayer.get(layer)!.map((f, i) => (
                  <MemberChip key={`${f.feature_idx}-${layer}-${i}`} f={f} i={i} />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

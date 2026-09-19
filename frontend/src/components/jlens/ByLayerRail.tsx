/**
 * Per-layer readout at the selected position, output end first.
 *
 * DIFFUSE IS MARKED BY HUE, NOT BY DIMNESS. These rows used to be greyed to
 * near-invisibility, which made the least confident readouts also the least
 * legible — and on the DIFF grid the same treatment hid the cells carrying the
 * most interesting signal, where the Jacobian lens names a token the logit
 * lens does not rank at all. Pink says "low confidence" while staying readable,
 * so legibility no longer trades against meaning.
 *
 * Layers whose top-1 probability is below the diffuse threshold are marked
 * *expected to be uninterpretable* rather than presented as content (FPRD
 * §3.7). That marking is measured from the readout's own distribution — it is
 * not a claim about which layers are sensory or workspace, which needs a band
 * report this feature does not have.
 */

import { displayToken, isDiffuse } from './utils';
import type { LensTypeSlice } from '../../types/jlens';

interface ByLayerRailProps {
  axis: number[];
  slice: LensTypeSlice | undefined;
  pinned: string[];
  selLayerIdx: number;
  positionToken: string;
  selPos: number;
  onSelectLayer: (layerIdx: number) => void;
}

export function ByLayerRail({
  axis,
  slice,
  pinned,
  selLayerIdx,
  positionToken,
  selPos,
  onSelectLayer,
}: ByLayerRailProps) {
  const rows = axis.map((layer, i) => ({ layer, i })).reverse();

  return (
    <div>
      <div className="mb-2 text-xs font-medium text-slate-600 dark:text-slate-400">
        By layer · position {selPos}{' '}
        <span className="font-mono text-slate-500 dark:text-slate-500">
          {displayToken(positionToken)}
        </span>
      </div>
      <div className="max-h-72 space-y-0.5 overflow-y-auto pr-1">
        {rows.map(({ layer, i }) => {
          const row = slice?.top_tokens[i] ?? [];
          const diffuse = isDiffuse(slice?.top_probs[i]?.[0]);
          return (
            <button
              key={layer}
              type="button"
              onClick={() => onSelectLayer(i)}
              title={
                diffuse
                  ? 'Diffuse readout — expected to be uninterpretable, not a null result'
                  : undefined
              }
              className={`flex w-full items-start gap-2 rounded px-1.5 py-1 text-left ${
                selLayerIdx === i
                  ? 'bg-slate-200 dark:bg-slate-700'
                  : 'hover:bg-slate-100 dark:hover:bg-slate-700'
              }`}
            >
              <span
                className={`mt-px w-9 shrink-0 font-mono text-[10px] ${
                  diffuse
                    ? 'text-pink-600 dark:text-pink-300'
                    : 'text-slate-700 dark:text-slate-300'
                }`}
              >
                L{layer}
              </span>
              <span className="flex flex-wrap gap-1">
                {/* A display truncation of the row, NOT a top-n: the rail is a
                    scan surface and the full top-k lives in the detail panel.
                    Nothing here is derived from meta.top_n on purpose. */}
                {row.slice(0, 4).map((t, k) => (
                  <span
                    key={k}
                    className={`font-mono text-[10px] ${
                      pinned.includes(t)
                        ? 'rounded bg-emerald-100 px-1 text-emerald-700 dark:bg-emerald-900 dark:text-emerald-200'
                        : diffuse
                          ? 'text-pink-600 dark:text-pink-300'
                          : k === 0
                            ? 'text-slate-800 dark:text-slate-200'
                            : 'text-slate-500 dark:text-slate-500'
                    }`}
                  >
                    {displayToken(t)}
                  </span>
                ))}
                {diffuse && (
                  <span className="text-[10px] italic text-pink-600 dark:text-pink-300">
                    diffuse
                  </span>
                )}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

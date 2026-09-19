/**
 * Lens-mode tabs — Jacobian / Logit / Diff.
 *
 * Enablement comes from `modeAvailability(meta, mode)`, i.e. from what the
 * stream ACTUALLY CARRIES. A disabled tab always states why: an unexplained
 * disabled control reads as a bug, and — worse — an ENABLED Jacobian tab over a
 * logit-only stream would render logit readouts under a Jacobian label, which
 * is a lower evidence rung wearing a higher rung's clothes (BR-019).
 */

import { Eye, GitCompare, Sparkles } from 'lucide-react';
import { modeAvailability } from '../../stores/jlensStore';
import type { LensMetaMessage, LensMode } from '../../types/jlens';

const MODES: { id: LensMode; label: string; Icon: typeof Eye }[] = [
  { id: 'JACOBIAN_LENS', label: 'Jacobian', Icon: Sparkles },
  { id: 'LOGIT_LENS', label: 'Logit', Icon: Eye },
  { id: 'DIFF', label: 'Diff', Icon: GitCompare },
];

interface LensModeTabsProps {
  /** Whether a J-lens artifact is mounted for this model — changes WHICH
   * step the disabled reason names. */
  hasArtifact?: boolean;
  meta: LensMetaMessage | null;
  mode: LensMode;
  onChange: (mode: LensMode) => void;
}

export function LensModeTabs({
  meta,
  mode,
  onChange,
  hasArtifact = false,
}: LensModeTabsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      {MODES.map(({ id, label, Icon }) => {
        const { enabled, reason } = modeAvailability(meta, id, hasArtifact);
        const active = mode === id;
        return (
          <div key={id} className="flex flex-col items-start">
            <button
              type="button"
              onClick={() => enabled && onChange(id)}
              disabled={!enabled}
              title={reason ?? undefined}
              aria-disabled={!enabled}
              className={`flex items-center gap-1.5 rounded border px-2.5 py-1 text-xs font-medium transition ${
                active && enabled
                  ? 'border-emerald-500 bg-emerald-50 text-emerald-700 dark:border-emerald-600 dark:bg-emerald-900/40 dark:text-emerald-200'
                  : enabled
                    ? 'border-slate-300 bg-white text-slate-600 hover:text-slate-900 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-400 dark:hover:text-slate-200'
                    : 'cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-600'
              }`}
            >
              <Icon className="h-3.5 w-3.5" />
              {label}
            </button>
            {!enabled && reason && (
              <span className="mt-0.5 max-w-[13rem] text-[10px] leading-tight text-slate-500 dark:text-slate-500">
                {reason}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}

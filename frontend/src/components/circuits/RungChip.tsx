/**
 * Evidence-rung chip (Feature 018, IDL-35). Displays the SERVER-rendered
 * rung_language verbatim; the tooltip states what moves the artifact up one
 * rung. Colors encode rung strength without adding language of their own.
 */

import type { EvidenceRung } from '../../types/evidenceLadder';

const RUNG_STYLE: Record<number, string> = {
  0: 'bg-slate-100 dark:bg-slate-700/60 text-slate-700 dark:text-slate-300 border-slate-300 dark:border-slate-600',
  1: 'bg-sky-500/10 text-sky-300 border-sky-500/30',
  2: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
  3: 'bg-violet-500/10 text-violet-300 border-violet-500/30',
};

interface Props {
  rung: EvidenceRung;
  language: string;   // server-rendered — display verbatim
  nextStep?: string;  // server-rendered tooltip
  compact?: boolean;
}

export function RungChip({ rung, language, nextStep, compact }: Props) {
  return (
    <span
      className={`inline-flex items-center gap-1 rounded border px-1.5 py-0.5 text-[10px] font-medium ${RUNG_STYLE[rung] ?? RUNG_STYLE[0]}`}
      title={nextStep ? `Next rung: ${nextStep}` : undefined}
    >
      <span className="font-mono">R{rung}</span>
      {!compact && <span>{language}</span>}
    </span>
  );
}

export default RungChip;

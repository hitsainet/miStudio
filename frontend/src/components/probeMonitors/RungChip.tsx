/**
 * A probe's rung, worded BY THE SERVER (032, IDL-52).
 *
 * ⚠ THERE IS NO MAP FROM A RUNG NUMBER TO A PHRASE IN THIS FILE, AND THAT IS THE POINT.
 * The wording arrives as `rung_language` / `rung_next_step`. A client-side map would be a
 * second source of a detector's language, and miLLM mirrors the server's strings verbatim
 * — so two sources means two vocabularies that can drift apart, with the UI's version
 * being the one a person actually reads.
 */
import { Radar } from 'lucide-react';

interface RungChipProps {
  rung: number;
  /** From the server. Required, so the component cannot be used without it. */
  language: string;
  nextStep?: string;
  reasons?: string[];
}

/** Colour by rung. A colour is not a claim, so this is the only rung-keyed map here. */
const TONE: Record<number, string> = {
  0: 'bg-slate-700/60 text-slate-300 border-slate-600',
  1: 'bg-sky-900/40 text-sky-300 border-sky-700',
  2: 'bg-emerald-900/40 text-emerald-300 border-emerald-700',
  3: 'bg-violet-900/40 text-violet-300 border-violet-700',
};

export function RungChip({ rung, language, nextStep, reasons }: RungChipProps) {
  const tone = TONE[rung] ?? TONE[0];
  return (
    <div className="inline-flex flex-col gap-1" data-testid="rung-chip">
      <span
        className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${tone}`}
        title={nextStep}
      >
        <Radar className="h-3.5 w-3.5" aria-hidden="true" />
        <span data-testid="rung-language">{language}</span>
        <span className="text-[10px] opacity-70">rung {rung}</span>
      </span>
      {nextStep ? (
        <span className="text-[11px] text-slate-400" data-testid="rung-next-step">
          Next: {nextStep}
        </span>
      ) : null}
      {reasons && reasons.length > 0 ? (
        <ul className="mt-0.5 space-y-0.5 text-[11px] text-slate-500">
          {reasons.map((reason) => (
            <li key={reason}>· {reason}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

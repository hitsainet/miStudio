/**
 * The below-rung-2 acknowledgement (033 FR-5, task 6.2).
 *
 * ⚠ THIS IS NOT A CONFIRMATION DIALOG. A confirmation asks "are you sure"; this records WHO
 * accepted thin evidence and WHY, and the answer is written into the exported definition so
 * whoever serves the probe can read it. That is the whole mechanism — the reason travels with the
 * artifact, not with a session.
 *
 * ⚠ THE RUNG WORDING COMES FROM THE SERVER. There is no rung→phrase map in this file, for the same
 * reason `RungChip` has none: miLLM mirrors the server's strings verbatim, so a second vocabulary
 * here would be the one a person reads while the machine reads another.
 */
import { useState } from 'react';
import { AlertTriangle } from 'lucide-react';

interface AcknowledgeDialogProps {
  rung: number;
  /** `rung_language` from the server. Required — the dialog cannot state the claim without it. */
  rungLanguage: string;
  /** `rung_next_step`, so the operator can see the alternative to acknowledging. */
  nextStep?: string;
  busy?: boolean;
  onCancel: () => void;
  onConfirm: (reason: string) => void;
}

/** The contract's floor. "ok" records nothing, which is the point of having one. */
export const MIN_REASON_LENGTH = 10;

export function AcknowledgeDialog({
  rung,
  rungLanguage,
  nextStep,
  busy = false,
  onCancel,
  onConfirm,
}: AcknowledgeDialogProps) {
  const [reason, setReason] = useState('');
  const tooShort = reason.trim().length < MIN_REASON_LENGTH;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4"
      data-testid="probe-acknowledge-dialog"
    >
      <div className="w-full max-w-lg rounded-lg border border-amber-700/60 bg-slate-900 p-5">
        <div className="flex items-start gap-3">
          <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-amber-400" />
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-slate-100">
              Export below rung 2
            </h3>
            <p className="mt-1 text-sm text-slate-300">
              This probe is at <span className="font-mono">rung {rung}</span> —{' '}
              <span className="text-amber-300">{rungLanguage}</span>. An export states that it
              detects the concept; at this rung that is not what has been shown.
            </p>
            {nextStep ? (
              <p className="mt-2 text-xs text-slate-400">
                To raise it instead: {nextStep}
              </p>
            ) : null}
          </div>
        </div>

        <label className="mt-4 block text-xs font-medium uppercase tracking-wide text-slate-400">
          Why this is acceptable
        </label>
        <textarea
          value={reason}
          onChange={(event) => setReason(event.target.value)}
          rows={3}
          placeholder="e.g. exploratory monitor for a narrow concept; not used to gate anything"
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 text-sm text-slate-200 placeholder:text-slate-600"
          data-testid="probe-acknowledge-reason"
        />
        <p className="mt-1 text-xs text-slate-500">
          Recorded in the exported file as{' '}
          <span className="font-mono">evidence.acknowledgement</span>, with your name and the time.
          At least {MIN_REASON_LENGTH} characters.
        </p>

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
            data-testid="probe-acknowledge-cancel"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={tooShort || busy}
            onClick={() => onConfirm(reason.trim())}
            className="rounded bg-amber-600 px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
            data-testid="probe-acknowledge-confirm"
          >
            {busy ? 'Building…' : 'Acknowledge and export'}
          </button>
        </div>
      </div>
    </div>
  );
}

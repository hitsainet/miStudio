/**
 * The operating point, as a dial rather than a property of the run that produced it.
 *
 * ⚠ THIS EXISTED AS AN ARRAY ON DISK FOR A WEEK BEFORE ANYTHING COULD TURN IT. A threshold is the
 * `(1 - target_fpr)` quantile of the negative scores the run already saved, so re-cutting it is
 * arithmetic — milliseconds. Until this panel, the only code that wrote a threshold was inside the
 * `calibrating` stage of a full training run, so asking "what would 0.5% look like" cost ~2.6
 * hours of GPU and every calibration question got settled by argument instead.
 *
 * ⚠ AND THE REASON IT PREVIEWS BEFORE IT COMMITS IS NOT POLITENESS. A probe's absolute score does
 * NOT transfer between distributions: on the first shipped probe the five evaluation sets' own 1%
 * thresholds spanned 24 points, so one number behaves very differently on each. A fast dial with
 * no feedback is an invitation to lower the bar until the probe fires on whatever is in front of
 * you, which is not calibration. So the preview shows what the candidate bar does to EVERY set it
 * was measured on, beside the bar in force, and the server's own `caution` sentence above both.
 */
import { useState } from 'react';
import { AlertTriangle, Gauge } from 'lucide-react';

import type { ProbeRecalibration, ThresholdTransfer } from '../../types/probeMonitor';

interface ThresholdDialProps {
  probeId: string;
  currentThreshold: number | null;
  currentTargetFpr: number | null;
  onRecalibrate: (body: {
    target_fpr: number;
    preview: boolean;
    reason?: string;
    allow_fire_on_nothing?: boolean;
  }) => Promise<ProbeRecalibration | null>;
}

/** The budgets worth one click. Anything else is typed. */
const PRESETS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1];

function pct(value: number | null | undefined): string {
  return value === null || value === undefined ? '—' : `${(value * 100).toFixed(2)}%`;
}

function bar(value: number | null | undefined): string {
  // ⚠ `null` MEANS FIRE ON NOTHING, NOT ZERO. Rendering it as "0.0000" would read as the loosest
  // possible bar when it is the tightest — the NULL-is-not-zero confusion the probe model warns
  // about, one layer out.
  return value === null || value === undefined ? 'fires on nothing' : value.toFixed(4);
}

function TransferTable({ transfer, label }: { transfer: ThresholdTransfer; label: string }) {
  if (!transfer?.per_set?.length) return null;
  return (
    <div>
      <p className="mb-1 text-[11px] uppercase tracking-wide text-slate-500">{label}</p>
      <table className="w-full text-xs tabular-nums">
        <thead>
          <tr className="text-left text-slate-500">
            <th className="pr-2 font-normal">set</th>
            <th className="pr-2 text-right font-normal">recall</th>
            <th className="pr-2 text-right font-normal">spends</th>
          </tr>
        </thead>
        <tbody>
          {transfer.per_set.map((set) => (
            <tr key={set.name ?? 'unnamed'} className="border-t border-slate-800">
              <td className="pr-2 py-0.5 text-slate-300">{set.name ?? 'unnamed'}</td>
              <td className="pr-2 py-0.5 text-right text-slate-300">
                {set.unreachable ? (
                  <span
                    className="text-amber-300"
                    title={
                      'the bar is above every score this set produces, so recall here is zero ' +
                      'however well the probe ranks on it'
                    }
                  >
                    unreachable
                  </span>
                ) : (
                  pct(set.recall_at_shipped)
                )}
              </td>
              <td className="pr-2 py-0.5 text-right text-slate-400">
                {set.unreachable ? '0.00%' : pct(set.fpr_at_shipped)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function ThresholdDial({
  probeId,
  currentThreshold,
  currentTargetFpr,
  onRecalibrate,
}: ThresholdDialProps) {
  const [targetFpr, setTargetFpr] = useState<string>(
    currentTargetFpr !== null ? String(currentTargetFpr) : '0.01'
  );
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [proposal, setProposal] = useState<ProbeRecalibration | null>(null);
  const [refusal, setRefusal] = useState<string | null>(null);

  const parsed = Number(targetFpr);
  const valid = Number.isFinite(parsed) && parsed > 0 && parsed < 1;

  async function run(preview: boolean) {
    setBusy(true);
    setRefusal(null);
    try {
      const outcome = await onRecalibrate({
        target_fpr: parsed,
        preview,
        reason: reason.trim(),
      });
      if (outcome === null) {
        // The store put the server's refusal in `error`; say plainly that nothing moved rather
        // than leaving a stale proposal on screen looking like it applied.
        setRefusal('the server refused this move — see the error above');
        if (!preview) setProposal(null);
      } else {
        setProposal(outcome);
      }
    } finally {
      setBusy(false);
    }
  }

  const committed = proposal?.committed ?? false;

  return (
    <section
      className="mt-4 rounded border border-slate-700 bg-slate-900/60 p-3"
      data-testid="threshold-dial"
    >
      <div className="mb-2 flex items-center gap-2">
        <Gauge className="h-4 w-4 text-slate-400" />
        <h4 className="text-sm font-medium text-slate-200">Operating point</h4>
        <span className="text-xs text-slate-500">
          now {bar(currentThreshold)} at {pct(currentTargetFpr)}
        </span>
      </div>

      <p className="mb-2 text-xs text-slate-400">
        A threshold is the {'('}1 − target FPR{')'} quantile of the negatives this probe was
        calibrated on, so moving it needs no GPU. It does{' '}
        <strong className="text-slate-300">not</strong> make one number transfer between
        distributions — the table below is what the new bar would actually do on each set.
      </p>

      <div className="flex flex-wrap items-end gap-2">
        <label className="text-xs text-slate-400">
          target FPR
          <input
            id={`recal-target-${probeId}`}
            type="text"
            inputMode="decimal"
            value={targetFpr}
            onChange={(e) => setTargetFpr(e.target.value)}
            className="ml-2 w-24 rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-100"
          />
        </label>
        <div className="flex gap-1">
          {PRESETS.map((preset) => (
            <button
              key={preset}
              type="button"
              onClick={() => setTargetFpr(String(preset))}
              className={`rounded px-1.5 py-1 text-[11px] ${
                Number(targetFpr) === preset
                  ? 'bg-emerald-700 text-white'
                  : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
              }`}
            >
              {preset * 100}%
            </button>
          ))}
        </div>
        <button
          type="button"
          disabled={!valid || busy}
          onClick={() => void run(true)}
          className="rounded bg-slate-700 px-2 py-1 text-xs text-slate-100 hover:bg-slate-600 disabled:opacity-40"
        >
          {busy ? 'working…' : 'Preview'}
        </button>
      </div>

      {refusal ? (
        <p className="mt-2 text-xs text-amber-300" data-testid="recalibrate-refusal">
          {refusal}
        </p>
      ) : null}

      {proposal ? (
        <div className="mt-3 space-y-3" data-testid="recalibrate-proposal">
          <p className="text-xs text-slate-300">
            {committed ? 'Moved' : 'Would move'} from{' '}
            <span className="tabular-nums">{bar(proposal.current.threshold)}</span> at{' '}
            {pct(proposal.current.target_fpr)} to{' '}
            <span className="tabular-nums text-emerald-300">
              {bar(proposal.proposed.threshold)}
            </span>{' '}
            at {pct(proposal.proposed.target_fpr)}
            {proposal.proposed.realised_fpr !== null &&
            proposal.proposed.realised_fpr !== proposal.proposed.target_fpr ? (
              <span
                className="ml-1 text-slate-500"
                title={
                  'the realised rate is reported because it is usually not the target: with ' +
                  'N negatives the achievable rates are multiples of 1/N'
                }
              >
                (really spends {pct(proposal.proposed.realised_fpr)})
              </span>
            ) : null}
            {proposal.proposed.revision ? (
              <span className="ml-1 text-slate-500">· revision {proposal.proposed.revision}</span>
            ) : null}
          </p>

          {proposal.proposed.fires_on_nothing ? (
            <p className="flex items-start gap-1 text-xs text-amber-300">
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              this bar is above every negative, so the probe would fire on nothing — silence that
              is indistinguishable from nothing being wrong
            </p>
          ) : null}

          {proposal.transfer_proposed?.caution ? (
            <p
              className="flex items-start gap-1 text-xs text-amber-300"
              data-testid="transfer-caution"
            >
              <AlertTriangle className="mt-0.5 h-3 w-3 shrink-0" />
              {proposal.transfer_proposed.caution}
            </p>
          ) : null}

          <div className="grid gap-3 sm:grid-cols-2">
            <TransferTable transfer={proposal.transfer_current} label="at the current bar" />
            <TransferTable transfer={proposal.transfer_proposed} label="at the proposed bar" />
          </div>

          {proposal.window_decisions &&
          Object.keys(proposal.window_decisions).length > 0 ? (
            <p className="text-xs text-slate-400">
              Per-window bars move with it:{' '}
              {Object.entries(proposal.window_decisions)
                .map(([name, entry]) => `${name} ${bar(entry.threshold as number | null)}`)
                .join(' · ')}
            </p>
          ) : null}
          {proposal.length_bands && proposal.length_bands.length > 0 ? (
            <p className="text-xs text-slate-400">
              And {proposal.length_bands.length} per-length bands are re-cut at the same budget.
            </p>
          ) : null}

          {proposal.published_copies_go_stale ? (
            <p className="text-xs text-amber-300">
              This probe has been published. A published definition is append-only and cannot be
              reached, so any copy already on HuggingFace will keep stating the old bar.
            </p>
          ) : null}

          {committed ? (
            <p className="text-xs text-emerald-300" data-testid="recalibrate-committed">
              Committed.
              {proposal.definition_invalidated
                ? ' The cached definition was invalidated — build it again before exporting.'
                : ''}
            </p>
          ) : (
            <div className="flex flex-wrap items-end gap-2">
              <label className="text-xs text-slate-400">
                why
                <input
                  id={`recal-reason-${probeId}`}
                  type="text"
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="recorded on the probe"
                  className="ml-2 w-56 rounded border border-slate-600 bg-slate-800 px-2 py-1 text-xs text-slate-100"
                />
              </label>
              <button
                type="button"
                disabled={busy}
                onClick={() => void run(false)}
                className="rounded bg-emerald-700 px-2 py-1 text-xs text-white hover:bg-emerald-600 disabled:opacity-40"
              >
                {busy ? 'working…' : 'Commit this bar'}
              </button>
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
}

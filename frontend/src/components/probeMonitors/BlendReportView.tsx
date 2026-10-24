/**
 * One blend report (Feature 041 FR-12, FR-13, FR-28): candidates in the server's ranking with recall and its
 * Wilson interval on each test view, the cross-validated (in-fold) figures, calibration size, criterion
 * verdicts and the members' own figures. `not assessable` and `few groups` are shown neutrally with their
 * counts. A stale report names what moved and the call that reruns it. Nothing here is computed: every
 * figure is the report's.
 */
import { useState } from 'react';

import { useProbeFamiliesStore } from '../../stores/probeFamiliesStore';
import type { BlendEnvelope, BlendFigures } from '../../types/probeFamily';

const pct = (value: number | null | undefined) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`);

function Recall({ f }: { f: BlendFigures | undefined }) {
  if (!f) return <span>—</span>;
  return (
    <span>
      {pct(f.recall)} {f.wilson ? `[${pct(f.wilson[0])}, ${pct(f.wilson[1])}]` : ''}
    </span>
  );
}

export function BlendReportView({ envelope, behaviourKey }: { envelope: BlendEnvelope; behaviourKey: string }) {
  const { markSuggested, pruneReport, busy } = useProbeFamiliesStore();
  const [reason, setReason] = useState('');
  const report = envelope.report;
  return (
    <section aria-label={`Blend report ${envelope.report_id}`} className="space-y-2 text-xs text-slate-300">
      <p>
        Report <span className="font-mono">{envelope.report_id}</span>: {envelope.status}
        {envelope.failure ? ` — ${envelope.failure.message}` : ''}
      </p>
      {envelope.stale ? (
        <p role="status" className="text-amber-300" data-testid="blend-stale">
          Stale: {envelope.stale_reasons.map((r) => r.code.replace(/_/g, ' ') + (r.probe_id ? ` (${r.probe_id})` : '')).join('; ')}.
          Rerun it with rerun_blend_report, or from an agent.
        </p>
      ) : null}
      {report ? (
        <>
          <p className="text-slate-400">{report.statement}</p>
          <p>
            Ranked by {report.ranking.basis === 'held_out_min' ? 'the worst held-out recall' : 'in-distribution recall only (no held-out view)'}
            {report.flags.length ? `; ${report.flags.map((f) => f.replace(/_/g, ' ')).join(', ')}` : ''}.
          </p>
          <table aria-label="Blend candidates">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="pr-2">Candidate</th>
                {report.views.tests.map((t) => (
                  <th key={t.id} className="pr-2">{t.held_out ? 'held out' : 'in distribution'} {t.id}</th>
                ))}
                <th className="pr-2">Cross-validated (in fold)</th>
                <th className="pr-2">Calibration</th>
                <th>Criteria</th>
              </tr>
            </thead>
            <tbody>
              {report.ranking.order.map((name) => {
                const c = report.candidates.find((x) => x.name === name)!;
                return (
                  <tr key={name} data-testid={`candidate-${name}`}>
                    <td className="pr-2 font-mono">{name}</td>
                    {report.views.tests.map((t) => <td key={t.id} className="pr-2"><Recall f={c.tests[t.id]} /></td>)}
                    <td className="pr-2">
                      {c.cv.not_assessable ? (
                        <span className="rounded-sm border border-slate-600 px-1" data-testid="not-assessable">
                          not assessable: {(c.cv.reason ?? []).map((r) => r.replace(/_/g, ' ')).join(', ')}
                          {c.cv.groups != null ? ` (${c.cv.groups} groups)` : ''}
                        </span>
                      ) : `${pct(c.cv.recall)}${c.cv.auroc != null ? `, AUROC ${c.cv.auroc.toFixed(3)}` : ''}`}
                    </td>
                    <td className="pr-2">
                      {c.calibration.n_negatives} negatives{c.calibration.few_negatives ? ' (few negatives)' : ''}
                    </td>
                    <td>{c.criteria.map((k) => `${k.id}: ${k.verdict.replace(/_/g, ' ')}`).join('; ') || '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <details>
            <summary>Members' own figures</summary>
            <ul>
              {report.views.tests.map((t) => (
                <li key={t.id}>
                  {t.id}:{' '}
                  {Object.entries(report.members_own_figures[t.id] ?? {}).map(([p, f]) => (
                    <span key={p} className="mr-2">{p} <Recall f={f} /></span>
                  ))}
                </li>
              ))}
            </ul>
          </details>
          {envelope.status === 'done' && !envelope.stale ? (
            <div className="flex items-center gap-2">
              <label>
                Reason{' '}
                <input aria-label="Suggestion reason" value={reason} onChange={(e) => setReason(e.target.value)}
                  className="rounded-sm border border-slate-700 bg-slate-900" />
              </label>
              <button type="button" disabled={busy} className="text-sky-400 underline"
                onClick={() => void markSuggested(behaviourKey, envelope.model_id, envelope.report_id,
                  report.ranking.order[0], reason || null)}>
                Suggest {report.ranking.order[0]}
              </button>
            </div>
          ) : null}
          {envelope.rows_pruned_at ? <p>Per-row values pruned {envelope.rows_pruned_at}.</p> : (
            <button type="button" disabled={busy} className="text-slate-400 underline"
              onClick={() => void pruneReport(envelope.report_id)}>
              Prune per-row values (keeps this summary)
            </button>
          )}
        </>
      ) : null}
    </section>
  );
}

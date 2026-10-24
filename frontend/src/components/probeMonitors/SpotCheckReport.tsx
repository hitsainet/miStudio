/**
 * One spot-check's report (Feature 038 FR-10, D32): the statement first, then per probe the 2 × 2 table,
 * agreement with its confidence interval, kappa and the shortfall, then the disagreeing turns.
 * Every number is the server's.
 */
import { useEffect, useState } from 'react';

import { spotChecksApi, type SpotCheck, type SpotCheckItem, type SpotCheckProbeReport } from '../../api/spotChecks';

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

function ProbeTable({ probeId, report }: { probeId: string; report: SpotCheckProbeReport }) {
  const o = report.overall;
  return (
    <div className="rounded-sm border border-slate-700 p-2" data-testid={`spot-probe-${probeId}`}>
      <p className="text-sm text-slate-200">Probe {probeId}{report.newest_turn_only ? ' — the judge saw only the newest turn' : ''}</p>
      <table className="text-xs text-slate-300" aria-label={`Two by two table for ${probeId}`}>
        <thead><tr><th /><th>Judge: positive</th><th>Judge: negative</th></tr></thead>
        <tbody>
          <tr><th>Probe fired</th><td>{o.tp}</td><td>{o.fp}</td></tr>
          <tr><th>Probe did not fire</th><td>{o.fn}</td><td>{o.tn}</td></tr>
        </tbody>
      </table>
      <p className="text-xs text-slate-400">
        Agreement {pct(o.agreement)}
        {o.agreement_ci ? ` (95% confidence interval ${pct(o.agreement_ci.low)} to ${pct(o.agreement_ci.high)})` : ''}
        {o.kappa != null ? `, kappa ${o.kappa.toFixed(2)}` : ''} over {o.n} judged turns.
        {report.n_not_recorded ? ` ${report.n_not_recorded} sampled turns were not recorded in miLLM (transcripts are off by default).` : ''}
        {report.n_parse_failed ? ` ${report.n_parse_failed} judge replies could not be read.` : ''}
      </p>
    </div>
  );
}

export function SpotCheckReport({ check }: { check: SpotCheck }) {
  const [items, setItems] = useState<SpotCheckItem[]>([]);
  useEffect(() => {
    if (check.status === 'completed' || check.partial) {
      void spotChecksApi.items(check.id).then((r) => setItems(r.items)).catch(() => setItems([]));
    }
  }, [check.id, check.status, check.partial]);
  const m = check.metrics;
  if (!m) return <p className="text-xs text-slate-400">No report yet ({check.status}).</p>;
  const short = Object.entries(m.shortfall).filter(([, s]) => s.short > 0);
  return (
    <div className="space-y-2" aria-label="Spot-check report">
      <p className="text-sm font-semibold text-amber-200">{m.statement}</p>
      <p className="text-xs text-slate-400">{m.overall_note}</p>
      {m.partial ? <p className="text-xs text-amber-300">Partial: the check stopped before every sampled turn was judged.</p> : null}
      {short.map(([name, s]) => (
        <p key={name} className="text-xs text-slate-400">Only {s.returned} of {s.requested} {name.replace('_', ' ')} turns were available; nothing was padded.</p>
      ))}
      {Object.entries(m.probes).map(([id, r]) => <ProbeTable key={id} probeId={id} report={r} />)}
      {Object.entries(m.newest_turn_probes).map(([id, r]) => <ProbeTable key={id} probeId={id} report={r} />)}
      <h5 className="text-xs uppercase text-slate-400">Disagreements</h5>
      <ul aria-label="Disagreements" className="space-y-1 text-xs text-slate-300">
        {items.map((i) => (
          <li key={i.id}>
            {i.probe_fired ? 'Probe fired' : 'Probe did not fire'} (score {i.probe_score ?? '—'}, threshold {i.threshold ?? '—'});
            {' '}judge said {i.judge_value ?? '—'}. Session {i.session_id}.
            {i.judged_text ? <span className="block text-slate-500">{i.judged_text.map((t) => `${t.role}: ${t.content}`).join(' / ')}</span> : <span className="block text-slate-500">Excerpt expired.</span>}
          </li>
        ))}
      </ul>
      <a className="text-xs text-sky-400 underline" href={spotChecksApi.exportUrl(check.id)}>
        Download disagreements (JSON Lines; contains recorded user messages)
      </a>
    </div>
  );
}

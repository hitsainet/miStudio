/**
 * Rules ∪ probe on the behaviour screen (Feature 040 S3, FR-33 – FR-37, FR-43): start an evaluation of one probe
 * with the behaviour's rule set on the views it was evaluated on, poll it, cancel it, and read rules alone, probe
 * alone and rules or probe per view, the unknown rows and the union bar. Every figure is the server's.
 */
import { useEffect, useState } from 'react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import { useBehaviourDetectorsStore } from '../../stores/behaviourDetectorsStore';
import type { AdoptedProbe } from '../../types/behaviour';
import type { DetectorRates } from '../../types/probeMonitor';
import { RefusalNotice } from './DetectorReports';

const pct = (value: number | null | undefined) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`);
const LIVE = new Set(['pending', 'running']);

function Rates({ name, r }: { name: string; r: DetectorRates }) {
  return (
    <td className="pr-2" data-testid={`rates-${name}`}>
      recall {pct(r.positives.recall)} ({r.positives.flagged}/{r.positives.n}) · false positives {pct(r.negatives.fpr)} (
      {r.negatives.flagged}/{r.negatives.n})
    </td>
  );
}

export function RulesEvaluation({ behaviourKey, probes }: { behaviourKey: string; probes: AdoptedProbe[] }) {
  const { rulesEvaluation, rulesEvaluations, refusal, busy, startRulesEvaluation, pollRulesEvaluation,
    listRulesEvaluations, cancelRulesEvaluation } = useBehaviourDetectorsStore();
  const [probeId, setProbeId] = useState(probes[0]?.id ?? '');
  const [viewIds, setViewIds] = useState<string[]>([]);
  const [calibration, setCalibration] = useState(true);

  useEffect(() => {
    void listRulesEvaluations(behaviourKey);
  }, [behaviourKey, listRulesEvaluations]);

  async function loadViews() {
    const report = await probeMonitorsApi.getReport(probeId);
    setViewIds([...new Set(report.evaluations.map((e) => e.dataset_id))]);
  }

  const live = rulesEvaluation != null && LIVE.has(rulesEvaluation.status);
  return (
    <section aria-label="Rules and probe" className="space-y-1 text-xs text-slate-300">
      <h4 className="text-sm font-semibold text-slate-200">Rules and the probe together</h4>
      <div className="flex flex-wrap gap-1">
        <select aria-label="Rules probe" value={probeId} onChange={(e) => setProbeId(e.target.value)} className="bg-slate-900">
          {probes.map((p) => <option key={p.id} value={p.id}>{p.id}</option>)}
        </select>
        <button type="button" className="underline" onClick={() => void loadViews()}>Use its evaluated views</button>
        <label>
          <input type="checkbox" checked={calibration} onChange={(e) => setCalibration(e.target.checked)} /> union bar
        </label>
        <button type="button" disabled={busy || !viewIds.length} className="rounded-control bg-sky-800 px-2 text-white"
          onClick={() => void startRulesEvaluation(probeId, { view_ids: viewIds, calibration })}>
          Evaluate rules with the probe ({viewIds.length} view{viewIds.length === 1 ? '' : 's'})
        </button>
      </div>
      <RefusalNotice refusal={refusal} />
      {rulesEvaluation ? (
        <div data-testid="rules-evaluation">
          <p>
            {rulesEvaluation.evaluation_id}: <strong data-testid="rules-status">{rulesEvaluation.status}</strong>
            {rulesEvaluation.progress ? ` (${rulesEvaluation.progress.rows_done} of ${rulesEvaluation.progress.rows_total} rows)` : ''}
            {rulesEvaluation.error ? ` — ${rulesEvaluation.error.message ?? rulesEvaluation.error.code}` : ''}
            {rulesEvaluation.stale?.stale ? ` — stale: ${rulesEvaluation.stale.moved.map((m) => m.input).join(', ')}` : ''}
          </p>
          <button type="button" className="underline" onClick={() => void pollRulesEvaluation(rulesEvaluation.evaluation_id)}>
            Refresh
          </button>{' '}
          {live ? (
            <button type="button" className="underline" onClick={() => void cancelRulesEvaluation(rulesEvaluation.evaluation_id)}>
              Cancel
            </button>
          ) : null}
          {rulesEvaluation.result ? (
            <table className="mt-1">
              <tbody>
                {rulesEvaluation.result.views.map((v) => (
                  <tr key={v.view_id} data-testid="rules-view">
                    <td className="pr-2">{v.view_id}</td>
                    <Rates name="rules_alone" r={v.detectors.rules_alone} />
                    <Rates name="probe_alone" r={v.detectors.probe_alone} />
                    <Rates name="rules_or_probe" r={v.detectors.rules_or_probe} />
                    <td className="pr-2">unknown {v.unknown.n}{v.unknown.over_share ? ' (over the share)' : ''}</td>
                    <td>{v.reasons.map((r) => r.replace(/_/g, ' ')).join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : null}
          {rulesEvaluation.result?.calibration?.global ? (
            <p data-testid="union-bar">
              Union bar {rulesEvaluation.result.calibration.global.threshold ?? 'none'}: the union spends{' '}
              {pct(rulesEvaluation.result.calibration.global.realised_union_fpr)}, the rules{' '}
              {pct(rulesEvaluation.result.calibration.global.rule_share)} (reported, not written)
            </p>
          ) : null}
        </div>
      ) : null}
      {rulesEvaluations?.rules_evaluations.length ? (
        <ul aria-label="Earlier rules evaluations" className="text-slate-400">
          {rulesEvaluations.rules_evaluations.map((r) => (
            <li key={r.evaluation_id}>
              <button type="button" className="underline" onClick={() => void pollRulesEvaluation(r.evaluation_id)}>
                {r.evaluation_id}
              </button>{' '}
              {r.probe_id}: {r.status}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

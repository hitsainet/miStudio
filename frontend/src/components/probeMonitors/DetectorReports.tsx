/**
 * Behaviour-detector reports on the behaviour screen (Feature 040 S2, FR-43): the success report, the
 * model comparison and the per-column breakdown. Every figure and every word about a criterion comes from
 * the server's body; a refusal shows its message and the `next` call that unblocks it (FR-40).
 */
import { useEffect, useState } from 'react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import { useBehaviourDetectorsStore, type Refusal } from '../../stores/behaviourDetectorsStore';
import type { AdoptedProbe } from '../../types/behaviour';
import type { CriterionResult, ProbeDataset, ProbeEvaluation } from '../../types/probeMonitor';

const pct = (value: number | null | undefined) => (value == null ? '—' : `${(value * 100).toFixed(1)}%`);

export function RefusalNotice({ refusal }: { refusal: Refusal | null }) {
  if (!refusal) return null;
  return (
    <div role="alert" className="rounded-sm border border-amber-700 p-2 text-xs text-amber-200" data-testid="detector-refusal">
      <p>{refusal.message}</p>
      {refusal.next.length ? (
        <ul>
          {refusal.next.map((n, i) => (
            <li key={i}>Next: <code>{n.tool}</code> — {n.why}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

function CriterionRow({ c }: { c: CriterionResult }) {
  return (
    <tr data-testid={`criterion-${c.id}`}>
      <td className="pr-2 font-mono">{c.id}</td>
      <td className="pr-2">{c.detector}</td>
      <td className="pr-2">{c.held_out_test ?? 'in distribution'}</td>
      <td className="pr-2">
        {pct(c.figures?.recall)} {c.figures?.recall_ci ? `[${pct(c.figures.recall_ci[0])}, ${pct(c.figures.recall_ci[1])}]` : ''}
      </td>
      <td className="pr-2">≥ {pct(c.min_recall)} at {pct(c.target_fpr)}</td>
      <td className="pr-2" data-testid={`verdict-${c.id}`}>{c.verdict.replace(/_/g, ' ')}</td>
      <td className="pr-2">{c.reasons.map((r) => r.code.replace(/_/g, ' ')).join(', ')}</td>
      <td>{c.calibration?.sentence ?? ''}{c.calibration?.few_negatives ? ' (few negatives)' : ''}</td>
    </tr>
  );
}

export function DetectorSuccessReport({ behaviourKey }: { behaviourKey: string }) {
  const { report, reports, refusal, busy, fetchSuccessReport, listSuccessReports, readSuccessReport } =
    useBehaviourDetectorsStore();
  useEffect(() => {
    void listSuccessReports(behaviourKey);
  }, [behaviourKey, listSuccessReports]);
  return (
    <section aria-label="Success report" className="space-y-2">
      <h4 className="text-sm font-semibold text-slate-200">Does the detector meet its bar?</h4>
      <button type="button" disabled={busy} className="rounded-sm bg-sky-800 px-2 py-1 text-xs text-white"
        onClick={() => void fetchSuccessReport(behaviourKey, {})}>
        Build success report
      </button>
      <RefusalNotice refusal={refusal} />
      {report ? (
        <div className="text-xs text-slate-300" data-testid="success-report">
          {report.stale.stale ? (
            <p role="status" className="text-amber-300" data-testid="report-stale">
              Stale: {report.stale.moved.map((m) => m.input.replace(/_/g, ' ')).join(', ') || 'an input was removed'}.{' '}
              {report.stale.next.map((n) => `Next: ${n.tool}`).join('; ')}
            </p>
          ) : null}
          <p>
            Version {report.report.behaviour_version} on {report.report.model_id}, probe {report.report.probe.id} (
            {report.report.probe.chosen_by.replace(/_/g, ' ')}): <strong data-testid="overall-verdict">
              {report.report.overall_verdict.replace(/_/g, ' ')}</strong>
          </p>
          <table className="mt-1">
            <tbody>{report.report.criteria.map((c) => <CriterionRow key={c.id} c={c} />)}</tbody>
          </table>
        </div>
      ) : null}
      {reports?.reports.length ? (
        <ul aria-label="Earlier reports" className="text-xs text-slate-400">
          {reports.reports.map((r) => (
            <li key={r.id}>
              <button type="button" className="underline" onClick={() => void readSuccessReport(r.id)}>
                {r.id}
              </button>{' '}
              v{r.behaviour_version}, {r.model_id}: {r.overall_verdict.replace(/_/g, ' ')}
            </li>
          ))}
        </ul>
      ) : null}
    </section>
  );
}

export function ModelComparison({ behaviourKey }: { behaviourKey: string }) {
  const { comparison, fetchModelComparison, busy } = useBehaviourDetectorsStore();
  return (
    <section aria-label="Model comparison" className="space-y-1 text-xs text-slate-300">
      <button type="button" disabled={busy} className="rounded-sm border border-slate-600 px-2 py-1"
        onClick={() => void fetchModelComparison(behaviourKey)}>
        Compare models
      </button>
      {comparison ? (
        <table data-testid="model-comparison">
          <tbody>
            {comparison.models.map((m) => (
              <tr key={m.model_id}>
                <td className="pr-2">{m.model_id}</td>
                <td className="pr-2">{m.refused ? m.refused.message : (m.overall_verdict ?? '').replace(/_/g, ' ')}</td>
                {(m.criteria ?? []).map((c) => (
                  <td key={c.id} className="pr-2">{c.id}: {c.verdict.replace(/_/g, ' ')} {pct(c.recall)}</td>
                ))}
                <td>blend: {m.blend ? m.blend.reasons.join(', ').replace(/_/g, ' ') : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
    </section>
  );
}

interface BreakdownProps {
  probes: AdoptedProbe[];
}

export function ProbeBreakdown({ probes }: BreakdownProps) {
  const { breakdown, fetchBreakdown, updateViewColumns, busy } = useBehaviourDetectorsStore();
  const [probeId, setProbeId] = useState(probes[0]?.id ?? '');
  const [evaluations, setEvaluations] = useState<ProbeEvaluation[]>([]);
  const [views, setViews] = useState<ProbeDataset[]>([]);
  const [evaluationId, setEvaluationId] = useState('');
  const [column, setColumn] = useState('');
  const [newColumn, setNewColumn] = useState('');
  const [mode, setMode] = useState<'recorded' | 'target_fpr'>('recorded');
  const [target, setTarget] = useState('');

  async function loadEvaluations() {
    const [report, datasets] = await Promise.all([probeMonitorsApi.getReport(probeId), probeMonitorsApi.listDatasets()]);
    setEvaluations(report.evaluations);
    setViews(datasets);
    setEvaluationId(report.evaluations[0]?.id ?? '');
  }

  const evaluation = evaluations.find((e) => e.id === evaluationId);
  const view = views.find((v) => v.id === evaluation?.dataset_id);
  const declared = view?.group_columns ?? [];

  return (
    <section aria-label="Breakdown" className="space-y-1 text-xs text-slate-300">
      <div className="flex flex-wrap gap-1">
        <select aria-label="Breakdown probe" value={probeId} onChange={(e) => setProbeId(e.target.value)} className="bg-slate-900">
          {probes.map((p) => <option key={p.id} value={p.id}>{p.id}</option>)}
        </select>
        <button type="button" className="underline" onClick={() => void loadEvaluations()}>Load evaluations</button>
        {evaluations.length ? (
          <select aria-label="Breakdown evaluation" value={evaluationId} onChange={(e) => setEvaluationId(e.target.value)} className="bg-slate-900">
            {evaluations.map((e) => (
              <option key={e.id} value={e.id}>{views.find((v) => v.id === e.dataset_id)?.name ?? e.dataset_id}</option>
            ))}
          </select>
        ) : null}
        {view ? (
          <select aria-label="Breakdown column" value={column} onChange={(e) => setColumn(e.target.value)} className="bg-slate-900">
            <option value="">choose a declared column</option>
            {declared.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        ) : null}
        <select aria-label="Breakdown mode" value={mode} onChange={(e) => setMode(e.target.value as 'recorded' | 'target_fpr')} className="bg-slate-900">
          <option value="recorded">the decision in force</option>
          <option value="target_fpr">a target false-positive rate</option>
        </select>
        {mode === 'target_fpr' ? (
          <input aria-label="Breakdown target false-positive rate" value={target} onChange={(e) => setTarget(e.target.value)} className="w-20 bg-slate-900" />
        ) : null}
        <button type="button" disabled={busy || !column || !evaluationId} className="rounded-sm bg-sky-800 px-2 text-white"
          onClick={() => void fetchBreakdown(probeId, evaluationId, {
            by_column: column, mode, ...(mode === 'target_fpr' && target ? { target_fpr: Number(target) } : {}),
          })}>
          Show breakdown
        </button>
      </div>
      {view ? (
        <div className="flex gap-1">
          <input aria-label="Column to declare" placeholder="column in the split" value={newColumn} onChange={(e) => setNewColumn(e.target.value)} className="bg-slate-900" />
          <button type="button" className="underline" disabled={!newColumn.trim()}
            onClick={async () => {
              if (await updateViewColumns(view.id, [...declared, newColumn.trim()])) {
                setNewColumn('');
                await loadEvaluations();
              }
            }}>
            Declare column on {view.name}
          </button>
        </div>
      ) : null}
      {breakdown ? (
        <div data-testid="breakdown">
          <p>
            {breakdown.decision.mode === 'recorded'
              ? `Decision revision ${breakdown.decision.revision}, window ${breakdown.decision.window ?? 'none'}`
              : `Bar cut at ${pct(breakdown.decision.target_fpr)} on the calibration negatives`}
            {breakdown.calibration?.sentence ? ` · calibration: ${breakdown.calibration.sentence}` : ''}
            {breakdown.calibration?.few_negatives ? ' (few negatives)' : ''}
          </p>
          <table>
            <tbody>
              {breakdown.groups.map((g) => (
                <tr key={JSON.stringify(g.value)} data-testid="breakdown-group">
                  <td className="pr-2">{String(g.value)}</td>
                  <td className="pr-2">recall {pct(g.positives.recall)} ({g.positives.flagged}/{g.positives.n})</td>
                  <td className="pr-2">false positives {pct(g.negatives.fpr)} ({g.negatives.flagged}/{g.negatives.n})</td>
                  <td>{g.too_small ? `too small (under ${breakdown.min_rows} rows)` : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}

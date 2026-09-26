/**
 * Probe Monitors panel (Feature 032 FR-14).
 *
 * Tabs: Datasets / Runs / Probes / Judge.
 *
 * ⚠ EVERY NUMBER HERE COMES FROM THE SERVER, INCLUDING THE RUNG'S WORDING. Nothing is
 * derived, averaged or rounded into existence in this file. A refused evaluation renders
 * its REASON rather than a zero — see `MetricsTable`.
 */
import { useEffect, useState } from 'react';
import { AlertTriangle, Loader2, Radar, X } from 'lucide-react';

import { MetricsTable } from '../probeMonitors/MetricsTable';
import { ProbeExportSection } from '../probeMonitors/ProbeExportSection';
import { ProbeDatasetForm } from '../probeMonitors/ProbeDatasetForm';
import { ProbeRunForm } from '../probeMonitors/ProbeRunForm';
import { RungChip } from '../probeMonitors/RungChip';
import { useModelsStore } from '../../stores/modelsStore';
import { RunTile } from '../probeMonitors/RunTile';
import { SweepGrid } from '../probeMonitors/SweepGrid';
import { useProbeMonitorWebSocket } from '../../hooks/useProbeMonitorWebSocket';
import { useProbeMonitorsStore, type ProbeTab } from '../../stores/probeMonitorsStore';

const TABS: { id: ProbeTab; label: string }[] = [
  { id: 'datasets', label: 'Datasets' },
  { id: 'runs', label: 'Runs' },
  { id: 'probes', label: 'Probes' },
  { id: 'judge', label: 'Judge' },
];

export function ProbeMonitorsPanel() {
  const {
    activeTab,
    setActiveTab,
    datasets,
    runs,
    probes,
    judgeRuns,
    report,
    isLoading,
    error,
    clearError,
    loadDatasets,
    loadRuns,
    loadProbes,
    loadJudgeRuns,
    openReport,
    closeReport,
    cancelRun,
    deleteRun,
  } = useProbeMonitorsStore();

  const [followedRun, setFollowedRun] = useState<string | null>(null);
  useProbeMonitorWebSocket(followedRun);

  const fetchModels = useModelsStore((state) => state.fetchModels);
  useEffect(() => {
    void loadRuns();
    void loadDatasets();
    // The tiles name the model a run read; without this they show a bare `m_…` id.
    void fetchModels();
  }, [loadRuns, loadDatasets, fetchModels]);

  /**
   * ⚠ THE LIST REFRESHES WHILE ANYTHING IS LIVE, AND IT DID NOT BEFORE.
   *
   * `loadRuns()` ran once on mount and the WebSocket hook follows ONE run, so a list containing a
   * running job sat frozen at whatever stage it was at when the panel opened. With stages that
   * legitimately take fifteen minutes — `pooled_capture` is a single forward pass over 8,000 rows
   * — a stale tile is indistinguishable from a stuck job, which is exactly the question an
   * operator opens this panel to answer.
   *
   * Polled rather than pushed because the list is many runs and the channel is per-run; 5 s is
   * well inside the 60 s at which the server heartbeats, so the bar moves when the server says it
   * does. The interval is torn down the moment nothing is live, so an idle panel is silent.
   */
  const anyLive = runs.some((run) =>
    ['pending', 'running', 'cancelling'].includes(run.status)
  );
  useEffect(() => {
    if (!anyLive) return undefined;
    const timer = setInterval(() => void loadRuns(), 5000);
    return () => clearInterval(timer);
  }, [anyLive, loadRuns]);

  useEffect(() => {
    if (activeTab === 'probes') void loadProbes();
    if (activeTab === 'judge') void loadJudgeRuns();
  }, [activeTab, loadProbes, loadJudgeRuns]);

  // Follow whichever run is live, so the socket carries its progress.
  useEffect(() => {
    const live = runs.find((run) => run.status === 'running' || run.status === 'pending');
    setFollowedRun(live ? live.id : null);
  }, [runs]);

  // A probe VIEW's display name: its config is what distinguishes five views of one repo
  // ("anthropic_hh_balanced"), and the bare name repeats the repo for every row.
  const datasetNames = Object.fromEntries(
    datasets.map((d) => [d.id, d.config || d.name])
  );
  const { models } = useModelsStore();
  const modelNames = Object.fromEntries(models.map((m) => [m.id, m.name]));

  return (
    <div className="p-6" data-testid="probe-monitors-panel">
      <header className="mb-4 flex items-center gap-3">
        <Radar className="h-6 w-6 text-emerald-400" aria-hidden="true" />
        <div>
          <h1 className="text-xl font-semibold text-slate-100">Probe Monitors</h1>
          <p className="text-sm text-slate-400">
            A linear readout over one layer's residual stream, evaluated on data it was
            not fitted on.
          </p>
        </div>
      </header>

      {error ? (
        <div
          className="mb-4 flex items-start gap-2 rounded border border-amber-700 bg-amber-950/40 p-3 text-sm text-amber-200"
          data-testid="probe-error"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          {/* VERBATIM. A 422 here names both class counts, and rewording it would lose
              the numbers a user needs to fix their mapping. */}
          <span className="flex-1">{error}</span>
          <button type="button" onClick={clearError} className="text-amber-300 hover:text-amber-100">
            <X className="h-4 w-4" />
          </button>
        </div>
      ) : null}

      <nav className="mb-4 flex gap-1 border-b border-slate-700" role="tablist">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-3 py-2 text-sm ${
              activeTab === tab.id
                ? 'border-b-2 border-emerald-500 text-slate-100'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {isLoading ? (
        <p className="mb-3 flex items-center gap-2 text-sm text-slate-400">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Loading…
        </p>
      ) : null}

      {activeTab === 'datasets' ? (
        <section data-testid="tab-datasets">
          <div className="mb-4">
            <ProbeDatasetForm onCreated={() => void loadDatasets()} />
          </div>
          {datasets.length === 0 ? (
            <p className="text-sm text-slate-400">
              No probe datasets yet. A probe dataset is a label-mapped view over a
              downloaded dataset — it records the columns and the mapping, not a copy of
              the rows.
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-slate-700 text-left text-xs uppercase text-slate-400">
                  <th className="py-2 pr-3">Name</th>
                  <th className="py-2 pr-3">Role</th>
                  <th className="py-2 pr-3">Config / split</th>
                  <th className="py-2 pr-3">Positive / negative</th>
                  <th className="py-2 pr-3">Excluded</th>
                  <th className="py-2 pr-3">Unparseable</th>
                </tr>
              </thead>
              <tbody>
                {datasets.map((view) => (
                  <tr key={view.id} className="border-b border-slate-800">
                    <td className="py-2 pr-3 text-slate-200">{view.name}</td>
                    <td className="py-2 pr-3 text-slate-400">
                      {view.role}
                      {view.distribution ? ` · ${view.distribution}` : ''}
                    </td>
                    <td className="py-2 pr-3 text-slate-400">
                      {view.config ?? '—'} / {view.split ?? '—'}
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-slate-200">
                      {view.counts?.positive ?? '—'} / {view.counts?.negative ?? '—'}
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-slate-400">
                      {view.counts?.excluded ?? '—'}
                    </td>
                    <td className="py-2 pr-3 tabular-nums text-slate-400">
                      {/* SHOWN, not hidden: a high count here means the input column is
                          being read wrongly, which is invisible in an AUROC. */}
                      {view.counts?.unparseable ?? '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </section>
      ) : null}

      {activeTab === 'runs' ? (
        <section data-testid="tab-runs">
          <div className="mb-4">
            <ProbeRunForm onSubmitted={() => void loadRuns()} />
          </div>
          {runs.length === 0 ? (
            <p className="text-sm text-slate-400">No probe runs yet.</p>
          ) : (
            <ul className="space-y-2">
              {runs.map((run) => (
                <RunTile
                  key={run.id}
                  run={run}
                  modelNames={modelNames}
                  datasetNames={datasetNames}
                  onCancel={(id) => void cancelRun(id)}
                  onDelete={(id) => void deleteRun(id)}
                >
                  {run.layer_selection ? (
                    <div className="mt-3">
                      <SweepGrid selection={run.layer_selection} />
                    </div>
                  ) : null}
                </RunTile>
              ))}
            </ul>
          )}
        </section>
      ) : null}

      {activeTab === 'probes' ? (
        <section data-testid="tab-probes">
          {probes.length === 0 ? (
            <p className="text-sm text-slate-400">No probes have been trained yet.</p>
          ) : (
            <ul className="space-y-2">
              {probes.map((probe) => (
                <li
                  key={probe.id}
                  className="flex items-center justify-between gap-3 rounded border border-slate-700 bg-slate-900/60 p-3"
                  data-testid="probe-row"
                >
                  <div>
                    <p className="text-sm text-slate-200">
                      L{probe.layer} · {probe.rule} · {probe.variant}
                      {probe.selected ? (
                        <span className="ml-2 rounded bg-emerald-900/50 px-1.5 py-0.5 text-[10px] text-emerald-300">
                          selected
                        </span>
                      ) : null}
                    </p>
                    <p className="text-xs text-slate-400">
                      {probe.threshold === null ? (
                        // NOT "0". No threshold was placed at all.
                        <span className="text-amber-300">no threshold placed</span>
                      ) : (
                        <>
                          threshold {probe.threshold.toFixed(4)} · spends{' '}
                          {probe.realised_fpr !== null
                            ? `${(probe.realised_fpr * 100).toFixed(1)}%`
                            : '—'}
                          {probe.threshold_source ? ` (${probe.threshold_source})` : ''}
                        </>
                      )}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => void openReport(probe.id)}
                    className="rounded border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                  >
                    Report
                  </button>
                </li>
              ))}
            </ul>
          )}

          {report ? (
            <div
              className="mt-4 rounded border border-slate-700 bg-slate-900/80 p-4"
              data-testid="probe-report"
            >
              <div className="mb-3 flex items-start justify-between">
                <RungChip
                  rung={report.probe.rung}
                  language={report.rung_language}
                  nextStep={report.rung_next_step}
                  reasons={report.probe.rung_reasons}
                />
                <button
                  type="button"
                  onClick={closeReport}
                  className="text-slate-400 hover:text-slate-200"
                  aria-label="Close report"
                >
                  <X className="h-4 w-4" />
                </button>
              </div>
              <MetricsTable
                evaluations={report.evaluations}
                datasetNames={datasetNames}
              />
              {report.paired_probe_id ? (
                <p className="mt-3 text-xs text-slate-400">
                  Paired with{' '}
                  <button
                    type="button"
                    className="underline"
                    onClick={() => void openReport(report.paired_probe_id!)}
                  >
                    {report.paired_probe_id}
                  </button>{' '}
                  — the other variant at the same layer and rule.
                </p>
              ) : null}
              {/* 033: the export flow lives ON the report, because every refusal it can give is
                  about the evidence shown above it — the rung, and whether the SAE has a home. */}
              <div className="mt-4">
                <ProbeExportSection
                  probe={report.probe}
                  rungLanguage={report.rung_language}
                  rungNextStep={report.rung_next_step}
                  definitionBuiltAt={report.probe.definition_built_at ?? null}
                  definitionSha256={report.probe.definition_sha256 ?? null}
                  definitionBuild={report.probe.definition_build ?? null}
                  published={report.probe.published ?? []}
                  saeHfRepo={report.sae_hf_repo ?? null}
                  onBuilt={() => void openReport(report.probe.id)}
                />
              </div>
            </div>
          ) : null}
        </section>
      ) : null}

      {activeTab === 'judge' ? (
        <section data-testid="tab-judge">
          <p className="mb-3 text-sm text-slate-400">
            {/* SAID PLAINLY. Rung 3 claims the two were measured on the same data, and
                nothing about the judge being right. */}
            A judge run is the <strong>baseline a probe is compared against</strong>, not a
            grader of the probe.
          </p>
          {judgeRuns.length === 0 ? (
            <p className="text-sm text-slate-400">No judge runs yet.</p>
          ) : (
            <ul className="space-y-2">
              {judgeRuns.map((run) => (
                <li
                  key={run.id}
                  className="rounded border border-slate-700 bg-slate-900/60 p-3 text-sm"
                  data-testid="judge-row"
                >
                  <p className="text-slate-200">
                    {run.model} · {run.status} · {run.prompt_version}
                  </p>
                  <p className="text-xs text-slate-400">
                    {run.parse_failures > 0 ? (
                      <span className="text-amber-300">
                        {run.parse_failures} replies could not be parsed
                      </span>
                    ) : (
                      'every reply parsed'
                    )}
                  </p>
                  {run.error_message ? (
                    <p className="mt-1 text-xs text-amber-300">{run.error_message}</p>
                  ) : null}
                </li>
              ))}
            </ul>
          )}
        </section>
      ) : null}
    </div>
  );
}

/**
 * A behaviour's blend reports on the chosen model (Feature 041 FR-16, FR-28), newest first; opening one
 * polls it every five seconds while it is queued or running and stops at a terminal status or on unmount.
 */
import { useEffect } from 'react';

import { useProbeFamiliesStore } from '../../stores/probeFamiliesStore';
import { BlendReportView } from './BlendReportView';

export const POLL_MS = 5000;

export function BlendReportList({ behaviourKey, model }: { behaviourKey: string; model: string | null }) {
  const { reports, openReport, loadReports, openReportById } = useProbeFamiliesStore();

  useEffect(() => {
    void loadReports(behaviourKey, model);
  }, [behaviourKey, model, loadReports]);

  const live = openReport && (openReport.status === 'queued' || openReport.status === 'running');
  useEffect(() => {
    if (!live || !openReport) return undefined;
    const id = openReport.report_id;
    const timer = setInterval(() => void openReportById(id), POLL_MS);
    return () => clearInterval(timer);
  }, [live, openReport, openReportById]);

  return (
    <section aria-label="Blend reports" className="space-y-1 text-xs text-slate-300">
      <h5 className="font-semibold text-slate-200">Blend reports</h5>
      {reports?.reports.length ? (
        <ul>
          {reports.reports.map((r) => (
            <li key={r.report_id}>
              <button type="button" className="text-sky-400 underline" onClick={() => void openReportById(r.report_id)}>
                {r.report_id}
              </button>{' '}
              {r.status}{r.stale ? ', stale' : ''}
            </li>
          ))}
        </ul>
      ) : <p className="text-slate-500">No blend report yet; an agent starts one with evaluate_probe_blends.</p>}
      {openReport ? <BlendReportView envelope={openReport} behaviourKey={behaviourKey} /> : null}
    </section>
  );
}

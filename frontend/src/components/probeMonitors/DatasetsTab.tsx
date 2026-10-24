/**
 * The Probes screen's Datasets tab: create, browse, archive and delete probe datasets.
 *
 * Operator decisions 2026-10-21 — the suite's pattern (miForge, miDataworks):
 *   * ARCHIVE hides a probe dataset; nothing is deleted and Unarchive undoes it. An archived
 *     dataset is not offered to new runs (the server refuses it too).
 *   * DELETE PERMANENTLY is for archived datasets, with "Show archived" on: a checkbox per
 *     archived row, Select all / Deselect all (indeterminate when partial), a live "N selected",
 *     and rows the server says cannot be deleted show why, inline, with their box disabled.
 *   * "Delete…" on any row is the one-step path the operator asked for ("enable me to delete the
 *     datasets"): it opens the same confirmation, with the same blockers, archives the dataset
 *     and then deletes it.
 *   * Clicking a name opens its rows (read-only).
 *
 * Like the Runs tab's Delete, deletion is gated by an explicit confirmation, not the Settings
 * PIN (which guards Settings only, decision D8).
 */
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import { formatJobTimestamp } from '../../utils/formatters';
import type { ProbeDataset, ProbeDatasetPurgePlan } from '../../types/probeMonitor';
import { useProbeMonitorsStore } from '../../stores/probeMonitorsStore';
import { blockedReason, deletable, isArchived } from './datasetArchive';
import { DatasetPurgeDialog } from './DatasetPurgeDialog';
import { DatasetRowsBrowser } from './DatasetRowsBrowser';
import { ProbeDatasetForm } from './ProbeDatasetForm';

type Api = Pick<
  typeof probeMonitorsApi,
  'planDatasetPurge' | 'purgeDatasets' | 'archiveDataset' | 'datasetRows'
>;

export interface DatasetsTabProps {
  api?: Api;
}

function archivedNote(view: ProbeDataset): string {
  const when = formatJobTimestamp(view.archived_at ?? null) ?? 'at an unknown time';
  const who = view.archived_by?.startsWith('agent:') ? `an agent (${view.archived_by.slice(6)})` : 'you';
  return `Archived ${when} by ${who}${view.archive_reason ? ` · ${view.archive_reason}` : ''}`;
}

export function DatasetsTab({ api = probeMonitorsApi }: DatasetsTabProps) {
  const datasets = useProbeMonitorsStore((state) => state.datasets);
  const loadDatasets = useProbeMonitorsStore((state) => state.loadDatasets);
  const archiveDataset = useProbeMonitorsStore((state) => state.archiveDataset);
  const unarchiveDataset = useProbeMonitorsStore((state) => state.unarchiveDataset);

  const [showArchived, setShowArchived] = useState(false);
  const [plans, setPlans] = useState<Record<string, ProbeDatasetPurgePlan>>({});
  const [planError, setPlanError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [dialog, setDialog] = useState<{ views: ProbeDataset[]; afterArchive: boolean } | null>(null);
  const [browsing, setBrowsing] = useState<ProbeDataset | null>(null);
  const [revision, setRevision] = useState(0);

  const visible = useMemo(
    () => datasets.filter((view) => showArchived || !isArchived(view)),
    [datasets, showArchived]
  );
  const archivedIds = useMemo(
    () => (showArchived ? visible.filter(isArchived).map((view) => view.id) : []),
    [visible, showArchived]
  );
  const archivedKey = archivedIds.join('\n');
  const hiddenCount = datasets.filter(isArchived).length;

  // The server's plan for every archived row on screen: the only authority on what is deletable.
  const latest = useRef(0);
  useEffect(() => {
    if (!archivedKey) {
      setPlans({});
      setPlanError(null);
      return;
    }
    const ticket = ++latest.current;
    api
      .planDatasetPurge(archivedKey.split('\n'))
      .then((items) => {
        if (ticket !== latest.current) return;
        setPlanError(null);
        setPlans(Object.fromEntries(items.map((plan) => [plan.id, plan])));
      })
      .catch((error: unknown) => {
        if (ticket === latest.current) setPlanError(error instanceof Error ? error.message : String(error));
      });
  }, [api, archivedKey, revision]);

  const selectable = useMemo(
    () => archivedIds.filter((id) => deletable(plans[id])),
    [archivedIds, plans]
  );

  // Never keep a selection the screen no longer offers.
  useEffect(() => {
    setSelected((current) => {
      const kept = [...current].filter((id) => selectable.includes(id));
      return kept.length === current.size ? current : new Set(kept);
    });
  }, [selectable]);

  const count = selected.size;
  const all = selectable.length > 0 && count === selectable.length;
  const some = count > 0 && !all;
  const headerBox = useRef<HTMLInputElement>(null);
  useEffect(() => {
    if (headerBox.current) headerBox.current.indeterminate = some;
  }, [some]);

  const toggle = (id: string) => {
    if (!selectable.includes(id)) return;
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const refresh = useCallback(() => {
    void loadDatasets();
    setRevision((n) => n + 1);
  }, [loadDatasets]);

  const archive = async (view: ProbeDataset) => {
    if (await archiveDataset(view.id)) setRevision((n) => n + 1);
  };
  const unarchive = async (view: ProbeDataset) => {
    if (await unarchiveDataset(view.id)) setRevision((n) => n + 1);
  };

  return (
    <section data-testid="tab-datasets">
      <div className="mb-4">
        <ProbeDatasetForm onCreated={() => void loadDatasets()} />
      </div>

      <div className="mb-2 flex flex-wrap items-center justify-between gap-3">
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(event) => setShowArchived(event.target.checked)}
            data-testid="datasets-show-archived"
          />
          Show archived{hiddenCount > 0 ? ` (${hiddenCount})` : ''}
        </label>
      </div>

      {showArchived && archivedIds.length > 0 ? (
        <div
          className="mb-2 flex flex-wrap items-center justify-between gap-3 rounded-sm border border-slate-700 bg-slate-900 px-3 py-2 text-sm"
          data-testid="datasets-purge-header"
        >
          <label className="flex items-center gap-2 text-slate-300">
            <input
              ref={headerBox}
              type="checkbox"
              checked={all}
              disabled={selectable.length === 0}
              aria-checked={some ? 'mixed' : all}
              onChange={() => setSelected(all || some ? new Set() : new Set(selectable))}
              data-testid="datasets-select-all"
            />
            <span data-testid="datasets-select-all-label">{all || some ? 'Deselect all' : 'Select all'}</span>
            <span className="text-slate-500">
              {selectable.length === archivedIds.length
                ? 'archived datasets'
                : `${selectable.length} of ${archivedIds.length} archived datasets can be deleted`}
            </span>
          </label>
          <div className="flex items-center gap-3">
            <span className="text-slate-400" aria-live="polite" data-testid="datasets-selected-count">
              {`${count} selected`}
            </span>
            <button
              type="button"
              disabled={count === 0}
              onClick={() =>
                setDialog({
                  views: archivedIds.filter((id) => selected.has(id)).map(
                    (id) => datasets.find((view) => view.id === id) as ProbeDataset
                  ),
                  afterArchive: false,
                })
              }
              className="rounded-sm border border-red-700 px-3 py-1 text-red-300 hover:bg-red-950 disabled:opacity-40"
              title="Delete the selected archived datasets for good. You confirm first."
              data-testid="datasets-delete-selected"
            >
              Delete permanently
            </button>
          </div>
          {planError !== null ? (
            <p className="w-full text-amber-300" data-testid="datasets-plan-error">
              Could not check which archived datasets can be deleted: {planError}
            </p>
          ) : null}
        </div>
      ) : null}

      {visible.length === 0 ? (
        <p className="text-sm text-slate-400">
          {datasets.length === 0
            ? 'No probe datasets yet. A probe dataset is a label-mapped view over a downloaded dataset — it records the columns and the mapping, not a copy of the rows.'
            : 'Every probe dataset is archived. Tick "Show archived" to see them.'}
        </p>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-700 text-left text-xs uppercase text-slate-400">
              {showArchived ? <th className="w-8 py-2 pr-2" aria-label="Select" /> : null}
              <th className="py-2 pr-3">Name</th>
              <th className="py-2 pr-3">Role</th>
              <th className="py-2 pr-3">Config / split</th>
              <th className="py-2 pr-3">Positive / negative</th>
              <th className="py-2 pr-3">Excluded</th>
              <th className="py-2 pr-3">Unparseable</th>
              <th className="py-2 pr-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((view) => {
              const archived = isArchived(view);
              const plan = plans[view.id];
              const reason = archived ? blockedReason(plan) : null;
              return (
                <tr
                  key={view.id}
                  className={`border-b border-slate-800 align-top ${archived ? 'opacity-70' : ''}`}
                  data-testid={`dataset-row-${view.id}`}
                >
                  {showArchived ? (
                    <td className="py-2 pr-2">
                      {archived ? (
                        <input
                          type="checkbox"
                          checked={selected.has(view.id)}
                          disabled={plan === undefined || reason !== null}
                          onChange={() => toggle(view.id)}
                          aria-label={`Select ${view.name} for permanent deletion`}
                          data-testid={`dataset-select-${view.id}`}
                        />
                      ) : null}
                    </td>
                  ) : null}
                  <td className="py-2 pr-3">
                    <button
                      type="button"
                      onClick={() => setBrowsing(view)}
                      className="text-left text-slate-200 hover:text-emerald-300 hover:underline"
                      title="Browse this dataset's rows"
                      data-testid={`dataset-browse-${view.id}`}
                    >
                      {view.name}
                    </button>
                    {archived ? (
                      <div className="text-xs text-slate-500" data-testid={`dataset-archived-note-${view.id}`}>
                        {archivedNote(view)}
                      </div>
                    ) : null}
                    {archived && reason !== null ? (
                      <div className="text-xs text-amber-300" data-testid={`dataset-blocked-${view.id}`}>
                        Cannot be deleted: {reason}
                      </div>
                    ) : null}
                  </td>
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
                  <td className="py-2 pr-3 tabular-nums text-slate-400">{view.counts?.excluded ?? '—'}</td>
                  <td className="py-2 pr-3 tabular-nums text-slate-400">
                    {/* SHOWN, not hidden: a high count here means the input column is being read
                        wrongly, which is invisible in an AUROC. */}
                    {view.counts?.unparseable ?? '—'}
                  </td>
                  <td className="whitespace-nowrap py-2 pr-0 text-right text-xs">
                    <button
                      type="button"
                      onClick={() => setBrowsing(view)}
                      className="mr-2 text-slate-300 hover:text-emerald-300"
                    >
                      Rows
                    </button>
                    {archived ? (
                      <button
                        type="button"
                        onClick={() => void unarchive(view)}
                        className="mr-2 text-slate-300 hover:text-emerald-300"
                        data-testid={`dataset-unarchive-${view.id}`}
                      >
                        Unarchive
                      </button>
                    ) : (
                      <button
                        type="button"
                        onClick={() => void archive(view)}
                        className="mr-2 text-slate-300 hover:text-amber-300"
                        title="Hide it from this list and from new runs. Nothing is deleted."
                        data-testid={`dataset-archive-${view.id}`}
                      >
                        Archive
                      </button>
                    )}
                    <button
                      type="button"
                      onClick={() => setDialog({ views: [view], afterArchive: !archived })}
                      className="text-red-300 hover:text-red-200"
                      title={
                        archived
                          ? 'Delete this archived dataset for good. You confirm first.'
                          : 'Archive this dataset and delete it for good. You confirm first.'
                      }
                      data-testid={`dataset-delete-${view.id}`}
                    >
                      Delete…
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}

      {dialog !== null ? (
        <DatasetPurgeDialog
          views={dialog.views}
          afterArchive={dialog.afterArchive}
          api={api}
          onClose={() => setDialog(null)}
          onDone={() => {
            setSelected(new Set());
            refresh();
          }}
        />
      ) : null}
      {browsing !== null ? (
        <DatasetRowsBrowser view={browsing} api={api} onClose={() => setBrowsing(null)} />
      ) : null}
    </section>
  );
}

/**
 * Confirm a permanent delete of probe dataset views (operator decision 2026-10-21).
 *
 * The dialog RE-READS the server's plan when it opens, lists every selected view with what goes
 * and what is kept, lists separately any that are blocked (skipped), and then sends EXACTLY the
 * remaining ids. The server purges each on its own and answers per view; this shows that answer.
 *
 * `afterArchive` is the one-step path from a visible row ("Delete…"): the view is archived first
 * and then purged, with the same blockers and the same confirmation. If the purge is then refused
 * the view stays archived, which is what the dialog says.
 */
import { useEffect, useRef, useState } from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import type {
  ProbeDataset,
  ProbeDatasetPurgeBatch,
  ProbeDatasetPurgePlan,
} from '../../types/probeMonitor';
import { blockedReason, deletable } from './datasetArchive';

type Api = Pick<typeof probeMonitorsApi, 'planDatasetPurge' | 'purgeDatasets' | 'archiveDataset'>;

export interface DatasetPurgeDialogProps {
  /** The selection, in screen order. */
  views: ProbeDataset[];
  afterArchive?: boolean;
  onClose: () => void;
  /** After the server answered: the tab re-reads its list. */
  onDone: (batch: ProbeDatasetPurgeBatch) => void;
  api?: Api;
}

function plural(n: number, one: string, many = `${one}s`): string {
  return `${n} ${n === 1 ? one : many}`;
}

export function DatasetPurgeDialog({
  views,
  afterArchive = false,
  onClose,
  onDone,
  api = probeMonitorsApi,
}: DatasetPurgeDialogProps) {
  const [plans, setPlans] = useState<ProbeDatasetPurgePlan[] | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [batch, setBatch] = useState<ProbeDatasetPurgeBatch | null>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const ids = views.map((view) => view.id);
  const key = ids.join('\n');

  useEffect(() => {
    let live = true;
    api
      .planDatasetPurge(key.split('\n'))
      .then((items) => {
        if (live) setPlans(items);
      })
      .catch((error: unknown) => {
        if (live) setFailure(error instanceof Error ? error.message : String(error));
      });
    return () => {
      live = false;
    };
  }, [api, key]);

  useEffect(() => {
    cancelRef.current?.focus();
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !busy) onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose, busy]);

  const going = (plans ?? []).filter((plan) => deletable(plan, afterArchive));
  const skipped = (plans ?? []).filter((plan) => !deletable(plan, afterArchive));

  const confirm = async () => {
    setBusy(true);
    setFailure(null);
    try {
      const ready = going.map((plan) => plan.id);
      if (afterArchive) {
        for (const plan of going) {
          if (!plan.archived) await api.archiveDataset(plan.id, reason.trim() || undefined);
        }
      }
      const result = await api.purgeDatasets(ready, reason.trim() || undefined);
      setBatch(result);
      onDone(result);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  };

  const title =
    batch !== null
      ? 'Delete finished'
      : `Delete ${plural(ids.length, 'probe dataset')} permanently?`;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4"
      data-testid="dataset-purge-backdrop"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-purge-title"
        className="mt-12 w-full max-w-2xl rounded-card border border-slate-700 bg-slate-900 p-4 text-sm text-slate-200"
        data-testid="dataset-purge-dialog"
      >
        <h2 id="dataset-purge-title" className="mb-2 text-base font-semibold text-slate-100">
          {title}
        </h2>

        {failure !== null ? (
          <p className="mb-3 flex items-start gap-2 text-amber-300" data-testid="dataset-purge-failure">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            {failure}
          </p>
        ) : null}

        {batch !== null ? (
          <div data-testid="dataset-purge-outcome">
            <p className="mb-2">
              {`Deleted ${batch.purged}${
                batch.blocked === 0 ? '.' : `; ${batch.blocked} could not be deleted.`
              } A record of each deletion is kept.`}
            </p>
            <ul className="mb-3 space-y-1 text-slate-400">
              {batch.results.map((result) => (
                <li key={result.id} data-testid={`dataset-purge-result-${result.id}`}>
                  {result.outcome === 'purged'
                    ? `${result.name}: deleted`
                    : `${views.find((v) => v.id === result.id)?.name ?? result.id}: not deleted — ${
                        result.reason ?? 'refused'
                      }${afterArchive ? ' (it stays archived)' : ''}`}
                </li>
              ))}
            </ul>
            <div className="flex justify-end">
              <button
                type="button"
                onClick={onClose}
                className="rounded-control border border-slate-600 px-3 py-1 text-slate-200 hover:bg-slate-800"
                data-testid="dataset-purge-close"
              >
                Close
              </button>
            </div>
          </div>
        ) : plans === null ? (
          failure === null ? (
            <p className="flex items-center gap-2 text-slate-400">
              <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> Checking what each
              selected dataset is used by…
            </p>
          ) : null
        ) : (
          <>
            <p className="mb-3 text-slate-400">
              This cannot be undone. Each probe dataset's definition (its columns, label mapping
              and counts) is removed. The downloaded data it reads is kept — delete that from the
              Datasets screen if you want it gone. A record of who deleted what, and why, is kept.
              {afterArchive ? ' The dataset is archived first.' : ''}
            </p>
            {going.length > 0 ? (
              <ul className="mb-3 space-y-2" data-testid="dataset-purge-going">
                {going.map((plan) => (
                  <li
                    key={plan.id}
                    className="rounded-control border border-slate-700 p-2"
                    data-testid={`dataset-purge-going-${plan.id}`}
                  >
                    <div className="font-medium text-slate-100">{plan.name}</div>
                    {plan.kept.map((kept) => (
                      <div key={kept.dataset_id} className="text-xs text-slate-400">
                        {`Keeps the download${kept.name ? ` "${kept.name}"` : ''}${
                          kept.other_views > 0
                            ? ` (${plural(kept.other_views, 'other probe dataset')} still read it)`
                            : ''
                        }`}
                      </div>
                    ))}
                  </li>
                ))}
              </ul>
            ) : null}
            {skipped.length > 0 ? (
              <div className="mb-3" data-testid="dataset-purge-skipped">
                <div className="mb-1 font-medium text-amber-300">
                  {going.length === 0
                    ? 'Nothing can be deleted:'
                    : `${plural(skipped.length, 'dataset')} cannot be deleted and will be skipped:`}
                </div>
                <ul className="space-y-1 text-slate-300">
                  {skipped.map((plan) => (
                    <li key={plan.id} data-testid={`dataset-purge-skipped-${plan.id}`}>
                      <span className="font-medium">{plan.name ?? plan.id}</span>
                      {': '}
                      {blockedReason(plan, afterArchive)}
                      {plan.blockers.length > 1 ? (
                        <ul className="ml-4 list-disc text-xs text-slate-400">
                          {plan.blockers.map((blocker, index) => (
                            <li key={`${blocker.code}-${index}`}>{blocker.message}</li>
                          ))}
                        </ul>
                      ) : null}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
            <label className="mb-3 block">
              <span className="mb-1 block text-slate-400">Why (optional, kept in the record)</span>
              <input
                type="text"
                value={reason}
                maxLength={2000}
                onChange={(event) => setReason(event.target.value)}
                className="w-full rounded-control border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                data-testid="dataset-purge-reason"
              />
            </label>
            <div className="flex justify-end gap-2">
              <button
                ref={cancelRef}
                type="button"
                onClick={onClose}
                disabled={busy}
                className="rounded-control border border-slate-600 px-3 py-1 text-slate-200 hover:bg-slate-800"
                data-testid="dataset-purge-cancel"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => void confirm()}
                disabled={busy || going.length === 0}
                className="rounded-control border border-red-700 px-3 py-1 text-red-300 hover:bg-red-950 disabled:opacity-40"
                data-testid="dataset-purge-confirm"
              >
                {busy
                  ? 'Deleting…'
                  : `${afterArchive ? 'Archive and delete' : 'Delete'} ${plural(
                      going.length,
                      'dataset'
                    )} permanently`}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

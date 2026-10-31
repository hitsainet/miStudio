/**
 * Browse a probe dataset's rows — read-only (operator request 2026-10-21).
 *
 * Every row shows the label the dataset's mapping gives it (positive / negative / excluded /
 * filtered out / unreadable) beside the raw value. Those labels come from the server, from the
 * same per-row rule that produced the counts at the top, so clicking a count filters the table
 * to exactly that many rows. Paged on the server; the page size, the visible columns, a text
 * search over the input and a label filter are the controls.
 */
import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, ChevronLeft, ChevronRight, Loader2, X } from 'lucide-react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import type {
  ProbeDataset,
  ProbeDatasetRowsPage,
  ProbeRowOutcome,
} from '../../types/probeMonitor';
import { OUTCOME_LABELS, PAGE_SIZES, PREVIEW_CHARS, cellText } from './datasetRowsFormat';

type Api = Pick<typeof probeMonitorsApi, 'datasetRows'>;

const OUTCOME_STYLE: Record<ProbeRowOutcome, string> = {
  positive: 'border-emerald-700 text-emerald-300',
  negative: 'border-sky-700 text-sky-300',
  excluded: 'border-slate-600 text-slate-400',
  filtered_out: 'border-slate-600 text-slate-400',
  unparseable: 'border-amber-700 text-amber-300',
};

function Cell({ value, cut }: { value: unknown; cut: boolean }) {
  const [open, setOpen] = useState(false);
  const text = cellText(value);
  const long = text.length > PREVIEW_CHARS;
  return (
    <div className="whitespace-pre-wrap wrap-break-word">
      {open || !long ? text : `${text.slice(0, PREVIEW_CHARS)}…`}
      {long ? (
        <button
          type="button"
          onClick={() => setOpen(!open)}
          className="ml-1 text-xs text-emerald-400 hover:text-emerald-300"
        >
          {open ? 'less' : 'more'}
        </button>
      ) : null}
      {open && cut ? (
        <div className="text-xs text-amber-300">Cut at 20,000 characters by the server.</div>
      ) : null}
    </div>
  );
}

export interface DatasetRowsBrowserProps {
  view: ProbeDataset;
  onClose: () => void;
  api?: Api;
}

export function DatasetRowsBrowser({ view, onClose, api = probeMonitorsApi }: DatasetRowsBrowserProps) {
  const [page, setPage] = useState<ProbeDatasetRowsPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const [limit, setLimit] = useState<number>(50);
  const [label, setLabel] = useState<ProbeRowOutcome | null>(null);
  const [searchDraft, setSearchDraft] = useState('');
  const [search, setSearch] = useState('');
  /** Other columns the operator HID; the input, label and pair columns always show. */
  const [hidden, setHidden] = useState<string[]>([]);
  const [knownColumns, setKnownColumns] = useState<string[] | null>(null);

  const projection = useMemo(
    // Nothing hidden = no projection, so the first answer (which tells us the columns) does not
    // trigger a second request for the same page.
    () =>
      knownColumns === null || hidden.length === 0
        ? null
        : knownColumns.filter((c) => !hidden.includes(c)),
    [knownColumns, hidden]
  );
  const projectionKey = projection === null ? '*' : projection.join(',');

  useEffect(() => {
    let live = true;
    setLoading(true);
    setError(null);
    api
      .datasetRows(view.id, {
        offset,
        limit,
        label,
        search: search || null,
        columns: projectionKey === '*' ? null : projectionKey ? projectionKey.split(',') : [],
      })
      .then((result) => {
        if (!live) return;
        setPage(result);
        setKnownColumns((current) => current ?? result.other_columns);
      })
      .catch((reason: unknown) => {
        if (live) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (live) setLoading(false);
      });
    return () => {
      live = false;
    };
  }, [api, view.id, offset, limit, label, search, projectionKey]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  const counts = view.counts;
  const outcomes = Object.keys(OUTCOME_LABELS) as ProbeRowOutcome[];
  const shownOthers = page ? page.returned_columns.filter((c) => page.other_columns.includes(c)) : [];
  const first = page && page.rows.length > 0 ? page.offset + 1 : 0;
  const last = page ? page.offset + page.rows.length : 0;

  const filterBy = (next: ProbeRowOutcome | null) => {
    setLabel(next);
    setOffset(0);
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4"
      data-testid="dataset-rows-backdrop"
    >
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="dataset-rows-title"
        className="w-full max-w-6xl rounded-card border border-slate-700 bg-slate-900 p-4 text-sm text-slate-200"
        data-testid="dataset-rows-browser"
      >
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 id="dataset-rows-title" className="text-base font-semibold text-slate-100">
              {view.name}
            </h2>
            <p className="text-xs text-slate-400">
              {page?.dataset_name ? `From the download "${page.dataset_name}"` : 'Rows'}
              {view.split ? ` · split ${view.split}` : ''} · {view.role} · read-only
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close the row browser"
            className="text-slate-400 hover:text-slate-100"
          >
            <X className="h-5 w-5" />
          </button>
        </header>

        <div className="mb-3 flex flex-wrap gap-2" data-testid="dataset-rows-counts">
          <button
            type="button"
            onClick={() => filterBy(null)}
            className={`rounded-control border px-2 py-0.5 text-xs ${
              label === null ? 'border-emerald-600 text-slate-100' : 'border-slate-700 text-slate-400'
            }`}
          >
            All rows{page ? ` ${page.total_rows.toLocaleString()}` : ''}
          </button>
          {outcomes.map((outcome) => (
            <button
              key={outcome}
              type="button"
              onClick={() => filterBy(outcome)}
              aria-pressed={label === outcome}
              className={`rounded-control border px-2 py-0.5 text-xs ${OUTCOME_STYLE[outcome]} ${
                label === outcome ? 'bg-slate-800 ring-1 ring-emerald-600' : ''
              }`}
              data-testid={`dataset-rows-count-${outcome}`}
            >
              {OUTCOME_LABELS[outcome]} {(counts?.[outcome] ?? 0).toLocaleString()}
            </button>
          ))}
        </div>
        {page?.counts_agree === false ? (
          <p className="mb-3 flex items-start gap-2 text-amber-300" data-testid="dataset-rows-drift">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            The downloaded data no longer matches the counts recorded when this probe dataset was
            created (now: {outcomes.map((o) => `${OUTCOME_LABELS[o].toLowerCase()} ${page.recount?.[o] ?? 0}`).join(', ')}).
            Runs on it were submitted against the recorded counts.
          </p>
        ) : null}

        <div className="mb-3 flex flex-wrap items-end gap-3">
          <form
            onSubmit={(event) => {
              event.preventDefault();
              setSearch(searchDraft.trim());
              setOffset(0);
            }}
            className="flex items-end gap-2"
          >
            <label className="block">
              <span className="mb-1 block text-xs text-slate-400">Search the input text</span>
              <input
                type="search"
                value={searchDraft}
                maxLength={500}
                onChange={(event) => setSearchDraft(event.target.value)}
                className="w-64 rounded-control border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
                data-testid="dataset-rows-search"
              />
            </label>
            <button
              type="submit"
              className="rounded-control border border-slate-600 px-2 py-1 text-slate-200 hover:bg-slate-800"
            >
              Search
            </button>
          </form>
          <label className="block">
            <span className="mb-1 block text-xs text-slate-400">Label</span>
            <select
              value={label ?? ''}
              onChange={(event) => filterBy((event.target.value || null) as ProbeRowOutcome | null)}
              className="rounded-control border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
              data-testid="dataset-rows-label"
            >
              <option value="">All labels</option>
              {outcomes.map((outcome) => (
                <option key={outcome} value={outcome}>
                  {OUTCOME_LABELS[outcome]}
                </option>
              ))}
            </select>
          </label>
          <label className="block">
            <span className="mb-1 block text-xs text-slate-400">Rows per page</span>
            <select
              value={limit}
              onChange={(event) => {
                setLimit(Number(event.target.value));
                setOffset(0);
              }}
              className="rounded-control border border-slate-700 bg-slate-950 px-2 py-1 text-slate-100"
              data-testid="dataset-rows-page-size"
            >
              {PAGE_SIZES.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
          {knownColumns && knownColumns.length > 0 ? (
            <fieldset className="flex flex-wrap items-center gap-2" data-testid="dataset-rows-columns">
              <legend className="mb-1 text-xs text-slate-400">Other columns</legend>
              {knownColumns.map((column) => (
                <label key={column} className="flex items-center gap-1 text-xs text-slate-300">
                  <input
                    type="checkbox"
                    checked={!hidden.includes(column)}
                    onChange={() =>
                      setHidden((current) =>
                        current.includes(column)
                          ? current.filter((c) => c !== column)
                          : [...current, column]
                      )
                    }
                    data-testid={`dataset-rows-column-${column}`}
                  />
                  {column}
                </label>
              ))}
            </fieldset>
          ) : null}
        </div>

        {error !== null ? (
          <p className="mb-3 flex items-start gap-2 text-amber-300" data-testid="dataset-rows-error">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
            {error}
          </p>
        ) : null}

        <div className="mb-2 flex items-center justify-between text-xs text-slate-400">
          <span data-testid="dataset-rows-range">
            {page
              ? page.matched === 0
                ? 'No rows match.'
                : `Rows ${first.toLocaleString()}–${last.toLocaleString()} of ${page.matched.toLocaleString()}${
                    page.matched !== page.total_rows ? ` matching (${page.total_rows.toLocaleString()} in all)` : ''
                  }`
              : ''}
            {loading ? <Loader2 className="ml-2 inline h-3 w-3 animate-spin" aria-hidden="true" /> : null}
          </span>
          <span className="flex gap-1">
            <button
              type="button"
              onClick={() => setOffset(Math.max(0, offset - limit))}
              disabled={offset === 0 || loading}
              aria-label="Previous page"
              className="rounded-control border border-slate-700 p-1 disabled:opacity-40"
              data-testid="dataset-rows-prev"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => setOffset(offset + limit)}
              disabled={!page || offset + limit >= page.matched || loading}
              aria-label="Next page"
              className="rounded-control border border-slate-700 p-1 disabled:opacity-40"
              data-testid="dataset-rows-next"
            >
              <ChevronRight className="h-4 w-4" />
            </button>
          </span>
        </div>

        {page && page.rows.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs" data-testid="dataset-rows-table">
              <thead>
                <tr className="border-b border-slate-700 text-left uppercase text-slate-400">
                  <th className="py-1 pr-2">Row</th>
                  <th className="py-1 pr-2">Input ({page.input_column})</th>
                  <th className="py-1 pr-2">Label ({page.label_column})</th>
                  {page.pair_column ? <th className="py-1 pr-2">Pair ({page.pair_column})</th> : null}
                  {shownOthers.map((column) => (
                    <th key={column} className="py-1 pr-2">
                      {column}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {page.rows.map((row) => (
                  <tr
                    key={row.position}
                    className="border-b border-slate-800 align-top"
                    data-testid={`dataset-row-${row.position}`}
                  >
                    <td className="py-1 pr-2 tabular-nums text-slate-500">{row.position + 1}</td>
                    <td className="max-w-md py-1 pr-2 text-slate-200">
                      <Cell
                        value={row.values[page.input_column]}
                        cut={row.truncated.includes(page.input_column)}
                      />
                    </td>
                    <td className="py-1 pr-2">
                      <span
                        className={`rounded-full border px-1.5 py-0.5 ${OUTCOME_STYLE[row.outcome]}`}
                        data-testid={`dataset-row-outcome-${row.position}`}
                      >
                        {OUTCOME_LABELS[row.outcome]}
                      </span>
                      <div className="mt-0.5 text-slate-500">raw: {cellText(row.raw_label) || '—'}</div>
                    </td>
                    {page.pair_column ? (
                      <td className="py-1 pr-2 text-slate-400">{cellText(row.values[page.pair_column])}</td>
                    ) : null}
                    {shownOthers.map((column) => (
                      <td key={column} className="max-w-xs py-1 pr-2 text-slate-400">
                        <Cell value={row.values[column]} cut={row.truncated.includes(column)} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </div>
  );
}

/**
 * Create a probe dataset: pick columns from real samples, map every label value (032 FR-1).
 *
 * ⚠ THE COLUMN AND VALUE LISTS COME FROM THE DATA, NOT FROM A TEXT BOX. Typing a column
 * name means a typo becomes a 422 after the fact; reading `Object.keys(sample.data)` off
 * `GET /datasets/{id}/samples` means only columns that exist can be chosen. The distinct
 * label values are read the same way, which is what makes the mapping table exhaustive.
 *
 * ⚠ EVERY DISTINCT VALUE MUST BE MAPPED, INCLUDING TO `excluded`. An unmapped value is a
 * counted refusal on the server, not a silent drop — so the form makes the choice explicit
 * rather than letting a third class vanish between the input and the counts. `ambiguous`
 * in the reference data is exactly this case.
 *
 * ⚠ AND THE VALUES ARE SAMPLED, SO THE LIST CAN BE INCOMPLETE. It is drawn from the first
 * N rows, and the form says so: a value that appears only at row 50,000 will not be
 * offered here and will arrive as `unparseable` in the server's counts. That is stated in
 * the UI rather than left for someone to discover from a count they cannot explain.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';

import { getDatasetSamples } from '../../api/datasets';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useProbeMonitorsStore } from '../../stores/probeMonitorsStore';
import type { DatasetSample } from '../../types/dataset';

/** How many rows to read for the column and value lists. */
const SAMPLE_ROWS = 50;

type LabelTarget = 'positive' | 'negative' | 'excluded';
type Role = 'train' | 'eval' | 'calibration';

interface ProbeDatasetFormProps {
  onCreated?: (id: string) => void;
}

export function ProbeDatasetForm({ onCreated }: ProbeDatasetFormProps) {
  const { datasets, fetchDatasets } = useDatasetsStore();
  const createDataset = useProbeMonitorsStore((state) => state.createDataset);

  const [name, setName] = useState('');
  const [datasetId, setDatasetId] = useState('');
  const [config, setConfig] = useState('');
  const [split, setSplit] = useState('');
  const [inputColumn, setInputColumn] = useState('');
  const [labelColumn, setLabelColumn] = useState('');
  const [pairColumn, setPairColumn] = useState('');
  const [role, setRole] = useState<Role>('eval');
  const [distribution, setDistribution] = useState('');
  const [mapping, setMapping] = useState<Record<string, LabelTarget>>({});
  const [filterTerms, setFilterTerms] = useState('');
  const [filterMode, setFilterMode] = useState<'any' | 'all'>('any');
  const [caseSensitive, setCaseSensitive] = useState(false);

  const [samples, setSamples] = useState<DatasetSample[]>([]);
  const [loadingSamples, setLoadingSamples] = useState(false);
  const [sampleError, setSampleError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    void fetchDatasets();
  }, [fetchDatasets]);

  // Read real rows whenever the dataset changes. The column pickers have nothing to
  // offer until this lands, which is deliberate: an empty picker is honest, a free-text
  // box invites a typo that only fails at submit.
  useEffect(() => {
    if (!datasetId) {
      setSamples([]);
      return;
    }
    let cancelled = false;
    setLoadingSamples(true);
    setSampleError(null);
    getDatasetSamples(datasetId, { limit: SAMPLE_ROWS })
      .then((response) => {
        if (!cancelled) setSamples(response.data ?? []);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setSampleError(error instanceof Error ? error.message : String(error));
          setSamples([]);
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingSamples(false);
      });
    return () => {
      cancelled = true;
    };
  }, [datasetId]);

  // Union across rows, not the first row's keys: a sparse column is missing from some
  // rows, and offering only the first row's keys would hide it.
  const columns = useMemo(() => {
    const seen = new Set<string>();
    for (const sample of samples) {
      for (const key of Object.keys(sample.data ?? {})) seen.add(key);
    }
    return Array.from(seen).sort();
  }, [samples]);

  const labelValues = useMemo(() => {
    if (!labelColumn) return [];
    const seen = new Set<string>();
    for (const sample of samples) {
      const value = (sample.data ?? {})[labelColumn];
      // `null` is a real label value and must be mappable — the server matches it by
      // the literal key "None", so it is offered rather than skipped.
      seen.add(value === null || value === undefined ? 'None' : String(value));
    }
    return Array.from(seen).sort();
  }, [samples, labelColumn]);

  const unmapped = labelValues.filter((value) => !mapping[value]);
  const targets = new Set(Object.values(mapping));
  const needsBothClasses = role !== 'calibration';
  const missingClass =
    needsBothClasses && (!targets.has('positive') || !targets.has('negative'));
  const calibrationHasPositive = role === 'calibration' && targets.has('positive');

  const canSubmit =
    !!name &&
    !!datasetId &&
    !!inputColumn &&
    !!labelColumn &&
    labelValues.length > 0 &&
    unmapped.length === 0 &&
    !missingClass &&
    !calibrationHasPositive &&
    !submitting;

  const setTarget = useCallback((value: string, target: LabelTarget) => {
    setMapping((current) => ({ ...current, [value]: target }));
  }, []);

  const submit = async () => {
    setSubmitting(true);
    const terms = filterTerms
      .split(',')
      .map((term) => term.trim())
      .filter(Boolean);
    const created = await createDataset({
      name,
      dataset_id: datasetId,
      config: config || null,
      split: split || null,
      input_column: inputColumn,
      label_column: labelColumn,
      label_mapping: mapping,
      keyword_filter: terms.length
        ? { terms, mode: filterMode, case_sensitive: caseSensitive }
        : null,
      pair_column: pairColumn || null,
      role,
      distribution:
        role === 'eval' && distribution
          ? (distribution as 'in_distribution' | 'out_of_distribution')
          : null,
    });
    setSubmitting(false);
    if (created && onCreated) onCreated(created.id);
  };

  const field = 'w-full rounded border border-slate-600 bg-slate-900 px-2 py-1 text-sm text-slate-100';
  const labelClass = 'mb-1 block text-xs uppercase tracking-wide text-slate-400';

  return (
    <form
      className="space-y-4 rounded border border-slate-700 bg-slate-900/60 p-4"
      onSubmit={(event) => {
        event.preventDefault();
        if (canSubmit) void submit();
      }}
      data-testid="probe-dataset-form"
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className={labelClass} htmlFor="pmd-name">Name</label>
          <input
            id="pmd-name"
            className={field}
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="training view"
          />
        </div>
        <div>
          <label className={labelClass} htmlFor="pmd-dataset">Dataset</label>
          <select
            id="pmd-dataset"
            className={field}
            value={datasetId}
            onChange={(event) => {
              setDatasetId(event.target.value);
              // Reset the downstream choices: columns from the previous dataset almost
              // certainly do not exist in this one, and keeping them would submit a
              // mapping for a column that is not there.
              setInputColumn('');
              setLabelColumn('');
              setPairColumn('');
              setMapping({});
            }}
          >
            <option value="">select a dataset…</option>
            {datasets.map((dataset) => (
              <option key={dataset.id} value={dataset.id}>
                {dataset.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {loadingSamples ? (
        <p className="flex items-center gap-2 text-sm text-slate-400">
          <Loader2 className="h-4 w-4 animate-spin" aria-hidden="true" /> reading rows…
        </p>
      ) : null}
      {sampleError ? (
        <p className="text-sm text-amber-300" data-testid="sample-error">
          Could not read this dataset's rows: {sampleError}
        </p>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-3">
        <div>
          <label className={labelClass} htmlFor="pmd-input">Input column</label>
          <select
            id="pmd-input"
            className={field}
            value={inputColumn}
            onChange={(event) => setInputColumn(event.target.value)}
            disabled={columns.length === 0}
          >
            <option value="">{columns.length ? 'select…' : 'pick a dataset first'}</option>
            {columns.map((column) => (
              <option key={column} value={column}>{column}</option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass} htmlFor="pmd-label">Label column</label>
          <select
            id="pmd-label"
            className={field}
            value={labelColumn}
            onChange={(event) => {
              setLabelColumn(event.target.value);
              setMapping({});
            }}
            disabled={columns.length === 0}
          >
            <option value="">{columns.length ? 'select…' : 'pick a dataset first'}</option>
            {columns.map((column) => (
              <option key={column} value={column}>{column}</option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass} htmlFor="pmd-pair">
            Pair column <span className="normal-case text-slate-500">(optional)</span>
          </label>
          <select
            id="pmd-pair"
            className={field}
            value={pairColumn}
            onChange={(event) => setPairColumn(event.target.value)}
            disabled={columns.length === 0}
          >
            <option value="">none</option>
            {columns.map((column) => (
              <option key={column} value={column}>{column}</option>
            ))}
          </select>
          <p className="mt-1 text-[11px] text-slate-500">
            Keeps contrastive pairs on one side of the train/validation split. Without it a
            near-identical pair can straddle the boundary, and the validation score then
            measures recall of a memorised passage.
          </p>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-4">
        <div>
          <label className={labelClass} htmlFor="pmd-role">Role</label>
          <select
            id="pmd-role"
            className={field}
            value={role}
            onChange={(event) => setRole(event.target.value as Role)}
          >
            <option value="train">train</option>
            <option value="eval">eval</option>
            <option value="calibration">calibration</option>
          </select>
        </div>
        <div>
          <label className={labelClass} htmlFor="pmd-dist">Distribution</label>
          <select
            id="pmd-dist"
            className={field}
            value={distribution}
            onChange={(event) => setDistribution(event.target.value)}
            disabled={role !== 'eval'}
          >
            <option value="">—</option>
            <option value="in_distribution">in_distribution</option>
            <option value="out_of_distribution">out_of_distribution</option>
          </select>
          <p className="mt-1 text-[11px] text-slate-500">
            Only meaningful on an eval set; it is what rung 2 turns on.
          </p>
        </div>
        <div>
          <label className={labelClass} htmlFor="pmd-config">Config</label>
          <input
            id="pmd-config"
            className={field}
            value={config}
            onChange={(event) => setConfig(event.target.value)}
            placeholder="e.g. training"
          />
        </div>
        <div>
          <label className={labelClass} htmlFor="pmd-split">Split</label>
          <input
            id="pmd-split"
            className={field}
            value={split}
            onChange={(event) => setSplit(event.target.value)}
            placeholder="e.g. test"
          />
        </div>
      </div>

      {labelColumn ? (
        <fieldset data-testid="label-mapping">
          <legend className={labelClass}>
            Label mapping — every value must be mapped
          </legend>
          {labelValues.length === 0 ? (
            <p className="text-sm text-slate-400">
              No values found for <code>{labelColumn}</code> in the sampled rows.
            </p>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-xs uppercase text-slate-500">
                  <th className="py-1 pr-3">Value</th>
                  <th className="py-1 pr-3">positive</th>
                  <th className="py-1 pr-3">negative</th>
                  <th className="py-1 pr-3">excluded</th>
                </tr>
              </thead>
              <tbody>
                {labelValues.map((value) => (
                  <tr key={value} className="border-t border-slate-800">
                    <td className="py-1 pr-3 font-mono text-slate-200">{value}</td>
                    {(['positive', 'negative', 'excluded'] as LabelTarget[]).map((target) => (
                      <td key={target} className="py-1 pr-3">
                        <input
                          type="radio"
                          name={`map-${value}`}
                          aria-label={`${value} → ${target}`}
                          checked={mapping[value] === target}
                          onChange={() => setTarget(value, target)}
                        />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          <p className="mt-2 text-[11px] text-slate-500" data-testid="sampling-caveat">
            Values are read from the first {SAMPLE_ROWS} rows. A value that appears only
            later will not be offered here and will arrive in the server's counts as
            <em> unparseable</em> rather than being silently dropped.
          </p>
        </fieldset>
      ) : null}

      <fieldset>
        <legend className={labelClass}>Keyword filter (optional)</legend>
        <div className="grid gap-3 sm:grid-cols-3">
          <input
            className={field}
            value={filterTerms}
            onChange={(event) => setFilterTerms(event.target.value)}
            placeholder="comma-separated terms"
            aria-label="filter terms"
          />
          <select
            className={field}
            value={filterMode}
            onChange={(event) => setFilterMode(event.target.value as 'any' | 'all')}
            aria-label="filter mode"
          >
            <option value="any">any term</option>
            <option value="all">all terms</option>
          </select>
          <label className="flex items-center gap-2 text-sm text-slate-300">
            <input
              type="checkbox"
              checked={caseSensitive}
              onChange={(event) => setCaseSensitive(event.target.checked)}
            />
            case sensitive
          </label>
        </div>
        <p className="mt-1 text-[11px] text-slate-500">
          A filter NARROWS the set and never assigns a label. Rows it removes are counted
          separately as <em>filtered_out</em>.
        </p>
      </fieldset>

      {unmapped.length > 0 || missingClass || calibrationHasPositive ? (
        <div
          className="flex items-start gap-2 rounded border border-amber-700 bg-amber-950/30 p-2 text-sm text-amber-200"
          data-testid="mapping-warning"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <div>
            {unmapped.length > 0 ? (
              <p>
                {unmapped.length} value(s) unmapped: {unmapped.join(', ')}. An unmapped
                value is counted as unparseable rather than dropped.
              </p>
            ) : null}
            {missingClass ? (
              <p>
                A {role} set needs at least one positive and one negative, or it cannot be
                scored at all.
              </p>
            ) : null}
            {calibrationHasPositive ? (
              <p>
                A calibration set supplies negatives for the false-positive threshold and
                is not labelled for the concept — it must map nothing to positive.
              </p>
            ) : null}
          </div>
        </div>
      ) : null}

      <button
        type="submit"
        disabled={!canSubmit}
        className="rounded bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
      >
        {submitting ? 'Creating…' : 'Create probe dataset'}
      </button>
    </form>
  );
}

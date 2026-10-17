/**
 * Start a judge run — the baseline a probe is compared against.
 *
 * ⚠ WHY THIS EXISTS. `POST /judge-runs` and its cancel route were reachable only by hand. An audit
 * of the nineteen client methods against their callers found four with none, and this was the one
 * that mattered most: **rung 3 is "detects on unseen tasks, compared with a judge", so without a
 * way to start a judge run the top of the evidence ladder was unreachable from the UI.** A probe
 * could be trained, evaluated and exported in the panel, and then had to be finished with `curl`.
 *
 * ⚠ A JUDGE RUN IS A BASELINE, NOT A GRADER. Rung 3 claims the two were measured on the same rows;
 * it claims nothing about the judge being right, and on this estate the judge has beaten the probe.
 * The copy says so where someone starting one will read it.
 */

import {
  usePersistentState,
  useValidSelection,
  useValidSelections,
} from '../../hooks/usePersistentState';
import { Scale } from 'lucide-react';

import type { ProbeDataset, ProbeMonitorSummary } from '../../types/probeMonitor';

interface JudgeRunFormProps {
  /** Evaluation views only — a judge run scores the same sets the probe was evaluated on. */
  datasets: ProbeDataset[];
  probes: ProbeMonitorSummary[];
  busy?: boolean;
  onSubmit: (body: {
    endpoint: string;
    model: string;
    dataset_ids: string[];
    probe_id?: string | null;
    max_rows_per_set?: number;
    parse_failure_limit?: number;
  }) => void;
}

export function JudgeRunForm({ datasets, probes, busy = false, onSubmit }: JudgeRunFormProps) {
  const [endpoint, setEndpoint] = usePersistentState('judge.endpoint', '');
  const [model, setModel] = usePersistentState('judge.model', '');
  const [probeId, setProbeId] = usePersistentState('judge.probeId', '');
  const [datasetIds, setDatasetIds] = usePersistentState<string[]>('judge.datasetIds', []);
  const [maxRows, setMaxRows] = usePersistentState('judge.maxRows', 1000);
  const [parseLimit, setParseLimit] = usePersistentState('judge.parseLimit', 50);

  const evalSets = datasets.filter((d) => d.role === 'eval');

  // A restored selection must still exist — see `useValidSelection`. A probe or a view can be
  // deleted between the draft being saved and the form being reopened.
  useValidSelection(probeId, setProbeId, probes.map((p) => p.id), probes.length > 0);
  useValidSelections(datasetIds, setDatasetIds, evalSets.map((d) => d.id), datasets.length > 0);
  const ready = endpoint.trim() !== '' && model.trim() !== '' && datasetIds.length > 0;

  function toggle(id: string) {
    setDatasetIds((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id]
    );
  }

  return (
    <form
      className="mb-4 rounded border border-slate-700 bg-slate-900/60 p-4"
      data-testid="judge-run-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!ready || busy) return;
        onSubmit({
          endpoint: endpoint.trim(),
          model: model.trim(),
          dataset_ids: datasetIds,
          probe_id: probeId || null,
          max_rows_per_set: maxRows,
          parse_failure_limit: parseLimit,
        });
      }}
    >
      <p className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-200">
        <Scale className="h-4 w-4 text-emerald-400" aria-hidden="true" />
        Run a judge baseline
      </p>

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="text-xs text-slate-400">
          ENDPOINT
          <input
            className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
            placeholder="http://…/v1"
            value={endpoint}
            onChange={(e) => setEndpoint(e.target.value)}
            data-testid="judge-endpoint"
          />
          <span className="mt-1 block text-[11px] text-slate-500">
            An OpenAI-compatible endpoint. The judge is a separate model from the one the probe
            reads.
          </span>
        </label>

        <label className="text-xs text-slate-400">
          MODEL
          <input
            className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
            placeholder="Qwen2.5-7B-Instruct"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            data-testid="judge-model"
          />
        </label>

        <label className="text-xs text-slate-400">
          COMPARE AGAINST (optional)
          <select
            className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
            value={probeId}
            onChange={(e) => setProbeId(e.target.value)}
            data-testid="judge-probe"
          >
            <option value="">no probe — just a baseline</option>
            {probes.map((probe) => (
              <option key={probe.id} value={probe.id}>
                L{probe.layer} · {probe.rule} · {probe.variant} ({probe.id})
              </option>
            ))}
          </select>
          <span className="mt-1 block text-[11px] text-slate-500">
            Naming a probe is what lets it reach rung 3. Leave it blank to measure the judge alone.
          </span>
        </label>

        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-slate-400">
            MAX ROWS PER SET
            <input
              type="number"
              min={1}
              className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
              value={maxRows}
              onChange={(e) => setMaxRows(Number(e.target.value))}
              data-testid="judge-max-rows"
            />
          </label>
          <label className="text-xs text-slate-400">
            PARSE FAILURE LIMIT
            <input
              type="number"
              min={0}
              className="mt-1 w-full rounded border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
              value={parseLimit}
              onChange={(e) => setParseLimit(Number(e.target.value))}
              data-testid="judge-parse-limit"
            />
          </label>
        </div>
      </div>

      <fieldset className="mt-3">
        <legend className="text-xs text-slate-400">EVALUATION SETS</legend>
        {evalSets.length === 0 ? (
          <p className="mt-1 text-xs text-amber-300">
            No evaluation views exist yet — create one on the Datasets tab first.
          </p>
        ) : (
          <div className="mt-1 flex flex-wrap gap-2">
            {evalSets.map((set) => (
              <label
                key={set.id}
                className="flex items-center gap-1.5 rounded border border-slate-700 px-2 py-1 text-xs text-slate-300"
              >
                <input
                  type="checkbox"
                  checked={datasetIds.includes(set.id)}
                  onChange={() => toggle(set.id)}
                  data-testid={`judge-set-${set.id}`}
                />
                {set.config || set.name}
              </label>
            ))}
          </div>
        )}
        <p className="mt-1 text-[11px] text-slate-500">
          Use the same sets the probe was evaluated on. A comparison over different rows is not a
          comparison.
        </p>
      </fieldset>

      <button
        type="submit"
        disabled={!ready || busy}
        className="mt-3 rounded bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        data-testid="judge-submit"
      >
        {busy ? 'Starting…' : 'Start judge run'}
      </button>
    </form>
  );
}

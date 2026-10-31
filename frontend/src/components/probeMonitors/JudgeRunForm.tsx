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

import { useEffect, useState } from 'react';
import { isArchived } from './datasetArchive';
import {
  usePersistentState,
  useValidSelection,
  useValidSelections,
} from '../../hooks/usePersistentState';
import { Scale } from 'lucide-react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import type {
  JudgePrompt,
  ProbeDataset,
  ProbeMonitorSummary,
  SubmitJudgeRunBody,
} from '../../types/probeMonitor';

interface JudgeRunFormProps {
  /** Evaluation views only — a judge run scores the same sets the probe was evaluated on. */
  datasets: ProbeDataset[];
  probes: ProbeMonitorSummary[];
  busy?: boolean;
  onSubmit: (body: SubmitJudgeRunBody) => void;
}

/**
 * ⚠ THE PARSE-FAILURE LIMIT IS A PERCENTAGE HERE AND A FRACTION ON THE WIRE. This field used to
 * default to 50 and send `parse_failure_limit: 50`, which the server's `lt=1.0` refuses — so every
 * judge run started from this form failed with a 422 until the number was edited to below 1. The
 * person types "5" (per cent); the request carries 0.05.
 */
export function parseLimitFraction(percent: number): number {
  return percent / 100;
}

export function JudgeRunForm({ datasets, probes, busy = false, onSubmit }: JudgeRunFormProps) {
  const [endpoint, setEndpoint] = usePersistentState('judge.endpoint', '');
  const [model, setModel] = usePersistentState('judge.model', '');
  const [probeId, setProbeId] = usePersistentState('judge.probeId', '');
  const [datasetIds, setDatasetIds] = usePersistentState<string[]>('judge.datasetIds', []);
  const [maxRows, setMaxRows] = usePersistentState('judge.maxRows', 1000);
  // A new key: the old 'judge.parseLimit' held 50, which the server always refused.
  const [parseLimitPercent, setParseLimitPercent] = usePersistentState('judge.parseLimitPercent', 5);
  const [promptId, setPromptId] = usePersistentState('judge.promptId', '');
  const [prompts, setPrompts] = useState<JudgePrompt[]>([]);
  const [promptsError, setPromptsError] = useState<string | null>(null);

  useEffect(() => {
    probeMonitorsApi
      .listJudgePrompts()
      .then((list) => setPrompts(list.prompts))
      .catch((error: unknown) => setPromptsError(error instanceof Error ? error.message : String(error)));
  }, []);

  // Archived views are out of use: the server refuses them as new inputs (409), so they are not
  // offered here — filtered in the component, so every caller gets it.
  const evalSets = datasets.filter((d) => d.role === 'eval' && !isArchived(d));

  // A restored selection must still exist — see `useValidSelection`. A probe or a view can be
  // deleted between the draft being saved and the form being reopened.
  useValidSelection(probeId, setProbeId, probes.map((p) => p.id), probes.length > 0);
  useValidSelections(datasetIds, setDatasetIds, evalSets.map((d) => d.id), datasets.length > 0);
  useValidSelection(promptId, setPromptId, prompts.map((p) => p.id), prompts.length > 0);

  /*
   * THE PICKER IS FILTERED BY THE PROBE'S BEHAVIOUR. Rung 3 counts a judge run only when the
   * prompt judges the probe's own behaviour, so those prompts come first; the rest stay choosable
   * (a baseline of another behaviour is a legitimate measurement) under a heading that says they
   * will not raise the rung.
   */
  const probe = probes.find((p) => p.id === probeId);
  const probeBehaviour = probe?.behaviour ?? null;
  const matching = probeBehaviour ? prompts.filter((p) => p.behaviour === probeBehaviour) : prompts;
  const others = probeBehaviour ? prompts.filter((p) => p.behaviour !== probeBehaviour) : [];
  const chosenPrompt = prompts.find((p) => p.id === promptId) ?? null;
  // 040 FR-9: the default choice is labelled from the library row the server marks as the default,
  // never from text in this file (the generality guard, noBehaviourNamesInSource.test.ts).
  const defaultPrompt = prompts.find((p) => p.is_default) ?? null;
  const mismatch = Boolean(probe && chosenPrompt && chosenPrompt.behaviour !== probeBehaviour);
  const limitValid = parseLimitPercent > 0 && parseLimitPercent < 100;
  const rowsValid = maxRows >= 20 && maxRows <= 50000;
  const ready =
    endpoint.trim() !== '' && model.trim() !== '' && datasetIds.length > 0 && limitValid && rowsValid;

  function toggle(id: string) {
    setDatasetIds((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id]
    );
  }

  return (
    <form
      className="mb-4 rounded-card border border-slate-700 bg-slate-900/60 p-4"
      data-testid="judge-run-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!ready || busy) return;
        onSubmit({
          endpoint: endpoint.trim(),
          model: model.trim(),
          dataset_ids: datasetIds,
          probe_id: probeId || null,
          prompt_id: promptId || null,
          max_rows_per_set: maxRows,
          parse_failure_limit: parseLimitFraction(parseLimitPercent),
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
            className="mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
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
            className="mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
            placeholder="Qwen2.5-7B-Instruct"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            data-testid="judge-model"
          />
        </label>

        <label className="text-xs text-slate-400">
          COMPARE AGAINST (optional)
          <select
            className="mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
            value={probeId}
            onChange={(e) => setProbeId(e.target.value)}
            data-testid="judge-probe"
          >
            <option value="">no probe — just a baseline</option>
            {probes.map((probe) => (
              <option key={probe.id} value={probe.id}>
                {probe.behaviour_label ?? probe.behaviour ?? 'behaviour not set'} · layer {probe.layer} ·{' '}
                {probe.rule} · {probe.variant}
              </option>
            ))}
          </select>
          <span className="mt-1 block text-[11px] text-slate-500">
            Naming a probe is what lets it reach rung 3. Leave it blank to measure the judge alone.
          </span>
        </label>

        <label className="text-xs text-slate-400 sm:col-span-2">
          JUDGE PROMPT
          <select
            className="mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
            value={promptId}
            onChange={(e) => setPromptId(e.target.value)}
            data-testid="judge-prompt-picker"
          >
            <option value="">
              {defaultPrompt
                ? `${defaultPrompt.title} (${defaultPrompt.name} v${defaultPrompt.version}) — the default`
                : "The library's default prompt"}
            </option>
            <optgroup label={probeBehaviour ? `Judges "${probeBehaviour}", like this probe` : 'Library'}>
              {matching.map((p) => (
                <option key={p.id} value={p.id}>
                  {p.title} ({p.name} v{p.version})
                </option>
              ))}
            </optgroup>
            {others.length ? (
              <optgroup label="Other behaviours — will not raise this probe's evidence level">
                {others.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.title} ({p.name} v{p.version}, judges {p.behaviour})
                  </option>
                ))}
              </optgroup>
            ) : null}
          </select>
          {promptsError ? <span className="mt-1 block text-[11px] text-amber-300">{promptsError}</span> : null}
          {chosenPrompt ? (
            <span className="mt-1 block text-[11px] text-slate-500">{chosenPrompt.answer_form_summary}</span>
          ) : null}
          {probe && !probeBehaviour ? (
            <span className="mt-1 block text-[11px] text-amber-300" data-testid="judge-probe-no-behaviour">
              This probe has no behaviour recorded, so no judge run can raise it to the top evidence
              level. Set its behaviour on its report first.
            </span>
          ) : null}
          {mismatch ? (
            <span className="mt-1 block text-[11px] text-amber-300" data-testid="judge-prompt-mismatch">
              This prompt judges "{chosenPrompt!.behaviour}" and the probe detects "{probeBehaviour}".
              The run is a baseline only; it will not raise the probe's evidence level.
            </span>
          ) : null}
        </label>

        <div className="grid grid-cols-2 gap-3">
          <label className="text-xs text-slate-400">
            MAX ROWS PER SET
            <input
              type="number"
              min={20}
              max={50000}
              className="mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
              value={maxRows}
              onChange={(e) => setMaxRows(Number(e.target.value))}
              data-testid="judge-max-rows"
            />
          </label>
          <label className="text-xs text-slate-400">
            UNREADABLE REPLIES ALLOWED (%)
            <input
              type="number"
              min={0.1}
              max={99}
              step={0.1}
              className="mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
              value={parseLimitPercent}
              onChange={(e) => setParseLimitPercent(Number(e.target.value))}
              data-testid="judge-parse-limit"
            />
            <span className="mt-1 block text-[11px] text-slate-500">
              Above this share the run stops rather than report on a smaller sample. Rows: 20 to
              50,000 per set.
            </span>
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
                className="flex items-center gap-1.5 rounded-control border border-slate-700 px-2 py-1 text-xs text-slate-300"
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
        className="mt-3 rounded-control bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
        data-testid="judge-submit"
      >
        {busy ? 'Starting…' : 'Start judge run'}
      </button>
    </form>
  );
}

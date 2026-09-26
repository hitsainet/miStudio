/**
 * Submit a probe run (032 FR-5–FR-8, FR-10).
 *
 * ⚠ THE DEFAULTS SHOWN HERE ARE THE SERVER'S, WRITTEN OUT. They are duplicated from
 * `schemas/probe_monitor.py` deliberately and the duplication is TESTED, because a form
 * that quietly sends a different stride than the backend's default produces runs nobody
 * can compare — and FR-15 requires a run be reproducible from what it recorded.
 *
 * ⚠ IT SENDS ONLY WHAT WAS CHOSEN. Layers and stride are mutually exclusive: the backend
 * clears `stride` when explicit layers arrive, so sending both would make the stored
 * config claim a stride that governed nothing.
 *
 * ⚠ `can_split=False` ON THE SERVER MEANS "all" IS REFUSED AT SUBMIT. `GpuSelect` is
 * therefore rendered WITHOUT the split option: offering a choice the endpoint returns 400
 * for is a worse affordance than not offering it.
 */
import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle } from 'lucide-react';

import { GpuSelect } from '../common/GpuSelect';
import { useModelsStore } from '../../stores/modelsStore';
import { useProbeMonitorsStore } from '../../stores/probeMonitorsStore';

/** Mirrors `schemas/probe_monitor.py`. `probeRunForm.test.tsx` pins the duplication. */
export const SERVER_DEFAULTS = {
  stride: 5,
  topNLayers: 1,
  valFraction: 0.15,
  seed: 1337,
  maxLength: 4096,
  targetFpr: 0.01,
  saeK: '128',
} as const;

/** The six rules `ml/probe_monitor_model.RULES` implements. */
export const RULES = ['mean', 'max', 'last', 'softmax', 'attention', 'rolling_mean_max'] as const;

const SCOPES = ['all', 'assistant', 'user', 'last_assistant'] as const;

interface ProbeRunFormProps {
  onSubmitted?: (runId: string) => void;
}

export function ProbeRunForm({ onSubmitted }: ProbeRunFormProps) {
  const { models, fetchModels } = useModelsStore();
  const datasets = useProbeMonitorsStore((state) => state.datasets);
  const loadDatasets = useProbeMonitorsStore((state) => state.loadDatasets);
  const submitRun = useProbeMonitorsStore((state) => state.submitRun);

  const [modelId, setModelId] = useState('');
  const [trainDatasetId, setTrainDatasetId] = useState('');
  const [evalDatasetIds, setEvalDatasetIds] = useState<string[]>([]);
  const [calibrationDatasetId, setCalibrationDatasetId] = useState('');
  const [useExplicitLayers, setUseExplicitLayers] = useState(false);
  const [layersText, setLayersText] = useState('');
  const [stride, setStride] = useState(String(SERVER_DEFAULTS.stride));
  const [rules, setRules] = useState<string[]>(['mean', 'max', 'last', 'attention']);
  const [scope, setScope] = useState<string>('all');
  const [maxLength, setMaxLength] = useState(String(SERVER_DEFAULTS.maxLength));
  const [topNLayers, setTopNLayers] = useState(String(SERVER_DEFAULTS.topNLayers));
  const [targetFpr, setTargetFpr] = useState(String(SERVER_DEFAULTS.targetFpr));
  const [seed, setSeed] = useState(String(SERVER_DEFAULTS.seed));
  const [saeVariant, setSaeVariant] = useState(false);
  const [saeK, setSaeK] = useState<string>(SERVER_DEFAULTS.saeK);
  const [gpu, setGpu] = useState('auto');
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    void fetchModels();
    void loadDatasets();
  }, [fetchModels, loadDatasets]);

  const trainViews = useMemo(() => datasets.filter((d) => d.role === 'train'), [datasets]);
  const evalViews = useMemo(() => datasets.filter((d) => d.role === 'eval'), [datasets]);
  const calibrationViews = useMemo(
    () => datasets.filter((d) => d.role === 'calibration'),
    [datasets]
  );

  const parsedLayers = layersText
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number);
  const layersInvalid =
    useExplicitLayers &&
    (parsedLayers.length === 0 || parsedLayers.some((n) => !Number.isInteger(n) || n < 0));

  // Mirrors the server's own refusals so the user sees them before a 422 rather than
  // after. The server still enforces them — this is an affordance, not the guard.
  const trainIsAlsoEval = !!trainDatasetId && evalDatasetIds.includes(trainDatasetId);
  const trainIsCalibration = !!trainDatasetId && trainDatasetId === calibrationDatasetId;

  const canSubmit =
    !!modelId &&
    !!trainDatasetId &&
    rules.length > 0 &&
    !layersInvalid &&
    !trainIsAlsoEval &&
    !trainIsCalibration &&
    !submitting;

  const toggleRule = (rule: string) =>
    setRules((current) =>
      current.includes(rule) ? current.filter((r) => r !== rule) : [...current, rule]
    );

  const toggleEval = (id: string) =>
    setEvalDatasetIds((current) =>
      current.includes(id) ? current.filter((x) => x !== id) : [...current, id]
    );

  const submit = async () => {
    setSubmitting(true);
    const config: Record<string, unknown> = {
      rules,
      scope,
      max_length: Number(maxLength),
      top_n_layers: Number(topNLayers),
      target_fpr: Number(targetFpr),
      seed: Number(seed),
      sae_variant: saeVariant,
      val_fraction: SERVER_DEFAULTS.valFraction,
    };
    // EXCLUSIVE, not both — see the module docstring.
    if (useExplicitLayers) {
      config.layers = parsedLayers;
    } else {
      config.stride = Number(stride);
    }
    if (saeVariant) {
      config.sae_k = saeK
        .split(',')
        .map((part) => Number(part.trim()))
        .filter((n) => Number.isInteger(n) && n > 0);
    }
    const runId = await submitRun({
      model_id: modelId,
      train_dataset_id: trainDatasetId,
      eval_dataset_ids: evalDatasetIds,
      calibration_dataset_id: calibrationDatasetId || null,
      config,
      gpu,
    });
    setSubmitting(false);
    if (runId && onSubmitted) onSubmitted(runId);
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
      data-testid="probe-run-form"
    >
      <div className="grid gap-3 sm:grid-cols-2">
        <div>
          <label className={labelClass} htmlFor="pmr-model">Model</label>
          <select
            id="pmr-model"
            className={field}
            value={modelId}
            onChange={(event) => setModelId(event.target.value)}
          >
            <option value="">select a model…</option>
            {models.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass} htmlFor="pmr-train">Training view</label>
          <select
            id="pmr-train"
            className={field}
            value={trainDatasetId}
            onChange={(event) => setTrainDatasetId(event.target.value)}
          >
            <option value="">select a train view…</option>
            {trainViews.map((view) => (
              <option key={view.id} value={view.id}>{view.name}</option>
            ))}
          </select>
          {trainViews.length === 0 ? (
            <p className="mt-1 text-[11px] text-amber-300">
              No view has role <code>train</code> yet — create one first.
            </p>
          ) : null}
        </div>
      </div>

      <fieldset>
        <legend className={labelClass}>Evaluation sets</legend>
        {evalViews.length === 0 ? (
          <p className="text-sm text-slate-400">
            None yet. Without an out-of-distribution eval set a probe cannot pass rung 1.
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {evalViews.map((view) => (
              <label
                key={view.id}
                className="flex items-center gap-1.5 rounded border border-slate-700 px-2 py-1 text-sm text-slate-300"
              >
                <input
                  type="checkbox"
                  checked={evalDatasetIds.includes(view.id)}
                  onChange={() => toggleEval(view.id)}
                />
                {view.name}
                {view.distribution === 'out_of_distribution' ? (
                  <span className="text-[10px] text-emerald-300">OOD</span>
                ) : null}
              </label>
            ))}
          </div>
        )}
      </fieldset>

      <div className="grid gap-3 sm:grid-cols-3">
        <div>
          <label className={labelClass} htmlFor="pmr-calib">Calibration set</label>
          <select
            id="pmr-calib"
            className={field}
            value={calibrationDatasetId}
            onChange={(event) => setCalibrationDatasetId(event.target.value)}
          >
            <option value="">none — use validation negatives</option>
            {calibrationViews.map((view) => (
              <option key={view.id} value={view.id}>{view.name}</option>
            ))}
          </select>
        </div>
        <div>
          <label className={labelClass} htmlFor="pmr-scope">Scope</label>
          <select
            id="pmr-scope"
            className={field}
            value={scope}
            onChange={(event) => setScope(event.target.value)}
          >
            {SCOPES.map((option) => (
              <option key={option} value={option}>{option}</option>
            ))}
          </select>
          <p className="mt-1 text-[11px] text-slate-500">
            Which tokens may be scored. These are different detectors, not a preference.
          </p>
        </div>
        <div>
          <GpuSelect
            id="pmr-gpu"
            value={gpu}
            onChange={setGpu}
            /* No split option: the endpoint resolves can_split=False and returns 400
               for "all", so offering it would be an affordance for a refusal. */
            allowSplit={false}
          />
        </div>
      </div>

      <fieldset>
        <legend className={labelClass}>Layers</legend>
        <div className="flex flex-wrap items-center gap-3">
          <label className="flex items-center gap-1.5 text-sm text-slate-300">
            <input
              type="radio"
              name="layer-mode"
              checked={!useExplicitLayers}
              onChange={() => setUseExplicitLayers(false)}
            />
            stride
          </label>
          <input
            className={`${field} w-24`}
            type="number"
            min={1}
            value={stride}
            onChange={(event) => setStride(event.target.value)}
            disabled={useExplicitLayers}
            aria-label="stride"
          />
          <label className="flex items-center gap-1.5 text-sm text-slate-300">
            <input
              type="radio"
              name="layer-mode"
              checked={useExplicitLayers}
              onChange={() => setUseExplicitLayers(true)}
            />
            explicit
          </label>
          <input
            className={`${field} w-44`}
            value={layersText}
            onChange={(event) => setLayersText(event.target.value)}
            disabled={!useExplicitLayers}
            placeholder="11, 12, 13"
            aria-label="layers"
          />
        </div>
        {layersInvalid ? (
          <p className="mt-1 text-[11px] text-amber-300" data-testid="layers-invalid">
            Give one or more non-negative layer indices, comma separated.
          </p>
        ) : null}
      </fieldset>

      <fieldset>
        <legend className={labelClass}>Rules</legend>
        <div className="flex flex-wrap gap-2">
          {RULES.map((rule) => (
            <label
              key={rule}
              className="flex items-center gap-1.5 rounded border border-slate-700 px-2 py-1 text-sm text-slate-300"
            >
              <input
                type="checkbox"
                checked={rules.includes(rule)}
                onChange={() => toggleRule(rule)}
              />
              {rule}
            </label>
          ))}
        </div>
        {rules.length === 0 ? (
          <p className="mt-1 text-[11px] text-amber-300">A run must train at least one rule.</p>
        ) : null}
      </fieldset>

      <div className="grid gap-3 sm:grid-cols-4">
        <div>
          <label className={labelClass} htmlFor="pmr-maxlen">Max length</label>
          <input id="pmr-maxlen" className={field} type="number" value={maxLength}
                 onChange={(event) => setMaxLength(event.target.value)} />
        </div>
        <div>
          <label className={labelClass} htmlFor="pmr-topn">Top-N layers</label>
          <input id="pmr-topn" className={field} type="number" min={1} value={topNLayers}
                 onChange={(event) => setTopNLayers(event.target.value)} />
        </div>
        <div>
          <label className={labelClass} htmlFor="pmr-fpr">Target FPR</label>
          <input id="pmr-fpr" className={field} type="number" step="0.001" value={targetFpr}
                 onChange={(event) => setTargetFpr(event.target.value)} />
        </div>
        <div>
          <label className={labelClass} htmlFor="pmr-seed">Seed</label>
          <input id="pmr-seed" className={field} type="number" value={seed}
                 onChange={(event) => setSeed(event.target.value)} />
        </div>
      </div>

      <fieldset>
        <legend className={labelClass}>SAE variant</legend>
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={saeVariant}
            onChange={(event) => setSaeVariant(event.target.checked)}
          />
          also train a k-sparse probe over an SAE basis
        </label>
        {saeVariant ? (
          <input
            className={`${field} mt-2 w-44`}
            value={saeK}
            onChange={(event) => setSaeK(event.target.value)}
            placeholder="128, 256"
            aria-label="sae k values"
          />
        ) : null}
        <p className="mt-1 text-[11px] text-slate-500">
          Needs a ready residual SAE for this model at the selected layer, or the run is
          refused naming the layer.
        </p>
      </fieldset>

      {trainIsAlsoEval || trainIsCalibration ? (
        <div
          className="flex items-start gap-2 rounded border border-amber-700 bg-amber-950/30 p-2 text-sm text-amber-200"
          data-testid="leakage-warning"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden="true" />
          <div>
            {trainIsAlsoEval ? (
              <p>
                The training view is also an evaluation set. The resulting AUROC would
                measure memorisation, not detection.
              </p>
            ) : null}
            {trainIsCalibration ? (
              <p>
                The training view is also the calibration set. A threshold calibrated on
                training negatives does not hold on new data.
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
        {submitting ? 'Queuing…' : 'Start probe run'}
      </button>
    </form>
  );
}

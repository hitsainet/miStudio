/**
 * Pending MCP-agent approval requests (Feature 010; app shell since Feature 034).
 *
 * Mounted ONCE, in the app shell above every panel (`App.tsx`, T-97). It used
 * to live in the Steering panel, which was right while only steering asked for
 * approval; an agent's miLLM model load (Feature 034, FR-20) evicts the served
 * model and must be visible wherever the operator is. One mount also means one
 * `mcp/approvals` subscription, so no other component's unsubscribe can
 * silence it (the per-socket subscription lesson).
 *
 * Live via the mcp/approvals channel with a slow poll fallback; renders
 * NOTHING when there are no pending requests — the padding is inside the root
 * below, so an empty queue leaves no gap above the panel.
 */

import { useCallback, useEffect, useState } from 'react';
import { Bot, Check, X } from 'lucide-react';
import { approveRequest, denyRequest, listApprovals } from '../../api/featureGroups';
import { getModel } from '../../api/models';
import { probeMonitorsApi } from '../../api/probeMonitors';
import { getGPUList } from '../../api/system';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import type { ApprovalRequest } from '../../types/featureGroups';
import type { ProbeDataset } from '../../types/probeMonitor';
import type { GPUInfo } from '../../types/system';

export function ApprovalsBanner() {
  const [pending, setPending] = useState<ApprovalRequest[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const { on, off, subscribe, unsubscribe, isConnected } = useWebSocketContext();

  const refresh = useCallback(async () => {
    try {
      const response = await listApprovals('pending');
      setPending(response.approvals);
    } catch {
      // Endpoint always exists; transient errors just skip a refresh cycle.
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => void refresh(), 20000);
    return () => clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!isConnected) return;
    subscribe('mcp/approvals');
    const handler = () => void refresh();
    on('approval:created', handler);
    on('approval:resolved', handler);
    return () => {
      off('approval:created', handler);
      off('approval:resolved', handler);
      unsubscribe('mcp/approvals');
    };
  }, [isConnected, on, off, subscribe, unsubscribe, refresh]);

  const context = useProbeRunContext(pending);

  const resolve = async (id: string, action: 'approve' | 'deny') => {
    setBusy(id);
    setError(null);
    try {
      if (action === 'approve') await approveRequest(id);
      else await denyRequest(id, 'Denied by operator');
      await refresh();
    } catch (e: any) {
      setError(e.detail || e.message || `Failed to ${action}`);
    } finally {
      setBusy(null);
    }
  };

  if (pending.length === 0) return null;

  return (
    <div className="px-6 pt-4" data-testid="approvals-banner">
    <div className="bg-amber-950/40 border border-amber-800 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2">
        <Bot className="w-4 h-4 text-amber-400" />
        <span className="text-sm font-medium text-amber-300">
          {pending.length} agent request{pending.length > 1 ? 's' : ''} awaiting approval
        </span>
      </div>
      {error && <p className="text-xs text-red-400 mb-2">{error}</p>}
      <div className="space-y-2">
        {pending.map((request) => {
          return (
            <div
              key={request.id}
              className="flex items-start gap-3 bg-slate-100 dark:bg-slate-900/60 rounded-md px-3 py-2"
              data-testid="approval-request"
            >
              <div className="flex-1 min-w-0">
                <span className="text-xs font-mono text-amber-200">{request.tool_name}</span>
                <RequestDetails request={request} context={context} />
              </div>
              <button
                disabled={busy === request.id}
                onClick={() => void resolve(request.id, 'approve')}
                className="flex items-center gap-1 px-2.5 py-1 text-xs bg-emerald-700 hover:bg-emerald-600 text-white rounded-sm disabled:opacity-50"
              >
                <Check className="w-3 h-3" /> Approve
              </button>
              <button
                disabled={busy === request.id}
                onClick={() => void resolve(request.id, 'deny')}
                className="flex items-center gap-1 px-2.5 py-1 text-xs bg-slate-100 dark:bg-slate-700 hover:bg-slate-100 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-200 rounded-sm disabled:opacity-50"
              >
                <X className="w-3 h-3" /> Deny
              </button>
            </div>
          );
        })}
      </div>
    </div>
    </div>
  );
}

const STEERING_TOOLS = new Set(['steer_compare', 'steer_sweep', 'steer_combined']);
const PROBE_RUN_TOOL = 'start_probe_training_run';

/**
 * The one-line summary of a tool the banner knows how to say in a sentence, or
 * `null` when the request needs more than one line (a probe training run) or
 * the tool is unknown. A `null` is never a blank: `RequestDetails` then renders
 * the probe-run summary or EVERY argument in full. There is no truncation: a
 * request the operator cannot read in full is one they approve blind.
 */
export function renderSummary(request: ApprovalRequest): string | null {
  if (request.tool_name === 'millm_load_model') {
    const modelId = (request.payload as { model_id?: unknown }).model_id;
    return `Load miLLM model #${String(modelId)} — evicts the resident model`;
  }
  if (STEERING_TOOLS.has(request.tool_name)) {
    const payload = request.payload as {
      prompt?: string;
      selected_features?: unknown[];
      strength_values?: number[];
    };
    const featureCount =
      payload.selected_features?.length ?? (payload.strength_values ? 1 : 0);
    return `${featureCount} feature${featureCount === 1 ? '' : 's'} · "${(payload.prompt ?? '').slice(0, 60)}"`;
  }
  return null;
}

/** What the banner looked up to name the things a probe run uses. */
export interface ProbeRunContext {
  /** model id → display name; absent = not (yet) resolved. */
  modelNames: Record<string, string>;
  /** probe dataset view id → the view. */
  views: Record<string, ProbeDataset>;
  /** `null` = the card list could not be read. */
  gpus: GPUInfo[] | null;
}

const EMPTY_CONTEXT: ProbeRunContext = { modelNames: {}, views: {}, gpus: null };

/**
 * Resolve names for pending probe runs only: a queue with none of them makes no
 * extra requests. Every lookup may fail; the summary then names the raw id and
 * says it could not be resolved.
 */
function useProbeRunContext(pending: ApprovalRequest[]): ProbeRunContext {
  const [context, setContext] = useState<ProbeRunContext>(EMPTY_CONTEXT);
  const modelKey = pending
    .filter((r) => r.tool_name === PROBE_RUN_TOOL)
    .map((r) => String((r.payload as { model_id?: unknown }).model_id ?? ''))
    .sort()
    .join('|');
  const hasProbeRun = pending.some((r) => r.tool_name === PROBE_RUN_TOOL);

  useEffect(() => {
    if (!hasProbeRun) return;
    let cancelled = false;
    void (async () => {
      const [views, gpus, models] = await Promise.all([
        probeMonitorsApi.listDatasets().catch(() => [] as ProbeDataset[]),
        getGPUList().then((r) => r.gpus).catch(() => null),
        Promise.all(
          modelKey
            .split('|')
            .filter(Boolean)
            .map((id) =>
              getModel(id)
                .then((m) => [id, m.name] as const)
                .catch(() => null),
            ),
        ),
      ]);
      if (cancelled) return;
      setContext({
        views: Object.fromEntries(views.map((v) => [v.id, v])),
        gpus,
        modelNames: Object.fromEntries(models.filter((m) => m !== null)),
      });
    })();
    return () => {
      cancelled = true;
    };
  }, [hasProbeRun, modelKey]);

  return context;
}

function RequestDetails({
  request,
  context,
}: {
  request: ApprovalRequest;
  context: ProbeRunContext;
}) {
  if (request.tool_name === PROBE_RUN_TOOL) {
    return <ProbeRunSummary payload={request.payload} context={context} />;
  }
  const summary = renderSummary(request);
  if (summary !== null) {
    return <span className="text-xs text-slate-600 dark:text-slate-400 ml-2">{summary}</span>;
  }
  return <AllArguments payload={request.payload} />;
}

/** Every argument, in full, wrapped — never cut at a length. */
export function AllArguments({ payload }: { payload: Record<string, unknown> }) {
  const entries = Object.entries(payload);
  return (
    <details open className="mt-1 text-xs text-slate-600 dark:text-slate-400" data-testid="approval-arguments">
      <summary className="cursor-pointer">
        All {entries.length} argument{entries.length === 1 ? '' : 's'} the agent sent
      </summary>
      {entries.length === 0 ? (
        <p className="mt-1">No arguments.</p>
      ) : (
        <dl className="mt-1 space-y-0.5">
          {entries.map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <dt className="font-mono text-slate-500 shrink-0">{key}</dt>
              <dd className="font-mono break-all whitespace-pre-wrap">
                {typeof value === 'string' ? value : JSON.stringify(value)}
              </dd>
            </div>
          ))}
        </dl>
      )}
    </details>
  );
}

function countsText(view: ProbeDataset): string {
  const c = view.counts;
  return `${c.positive.toLocaleString()} positive, ${c.negative.toLocaleString()} negative rows`;
}

function viewText(id: string, context: ProbeRunContext): string {
  const view = context.views[id];
  if (!view) return `a probe dataset view the banner could not look up (${id})`;
  return `“${view.name}” — ${countsText(view)}`;
}

/** The card a probe run asks for, by name AND identifier (the identifier is what pins it). */
function gpuText(gpu: unknown, gpus: GPUInfo[] | null): string {
  const asked = gpu === undefined || gpu === null ? 'auto' : String(gpu);
  if (asked === 'auto') {
    return 'Automatic — whichever card has the most free memory when the job starts';
  }
  const card = gpus?.find((g) => g.uuid === asked || String(g.gpu_id) === asked);
  if (card) return `${card.name} (${card.uuid})`;
  if (gpus === null) return `${asked} — the card list could not be read to name it`;
  return `${asked} — not in this machine's current card list`;
}

const CONFIG_LABELS: Record<string, string> = {
  layers: 'Layers',
  stride: 'Every n-th layer',
  scope: 'Tokens read',
  rules: 'Pooling rules',
  max_length: 'Maximum tokens per input',
  target_fpr: 'Target false-positive rate',
  sae_variant: 'Also train sparse autoencoder (SAE) probes',
  sae_k: 'Sparse autoencoder (SAE) feature counts',
  top_n_layers: 'Best layers kept',
  batch_size: 'Batch size',
  val_fraction: 'Held-out validation fraction',
  seed: 'Random seed',
  calibration_max_rows: 'Calibration rows used at most',
  rolling_windows: 'Rolling windows',
};

/**
 * A probe training run, in plain words: WHAT IT DETECTS first (the behaviour,
 * which is what tells several pending runs on one model apart), then the model,
 * the graphics card BY NAME AND IDENTIFIER, every dataset view by name with its
 * row counts, and the configuration the agent set. The full argument list
 * follows, collapsed.
 */
export function ProbeRunSummary({
  payload,
  context,
}: {
  payload: Record<string, unknown>;
  context: ProbeRunContext;
}) {
  const p = payload as {
    model_id?: unknown;
    train_dataset_id?: unknown;
    eval_dataset_ids?: unknown;
    calibration_dataset_id?: unknown;
    config?: unknown;
    gpu?: unknown;
    behaviour?: unknown;
    behaviour_label?: unknown;
  };
  const behaviour = typeof p.behaviour === 'string' && p.behaviour !== '' ? p.behaviour : null;
  const behaviourLabel =
    typeof p.behaviour_label === 'string' && p.behaviour_label !== '' ? p.behaviour_label : null;
  const modelId = String(p.model_id ?? '');
  const modelName = context.modelNames[modelId];
  const evalIds = Array.isArray(p.eval_dataset_ids) ? p.eval_dataset_ids.map(String) : [];
  const config =
    p.config && typeof p.config === 'object' ? (p.config as Record<string, unknown>) : null;
  const lineClass = 'text-xs text-slate-600 dark:text-slate-400';
  return (
    <div className="mt-1 space-y-0.5" data-testid="probe-run-summary">
      <p className="text-sm font-semibold text-slate-900 dark:text-slate-100" data-testid="probe-run-behaviour">
        {behaviourLabel ?? behaviour
          ? `Detects: ${behaviourLabel ?? behaviour}${behaviourLabel && behaviour ? ` (behaviour key “${behaviour}”)` : ''}`
          : 'Detects: no behaviour given — tell these probes apart by their training set below'}
      </p>
      <p className="text-xs text-slate-800 dark:text-slate-200">
        Train probes on {modelName ? `“${modelName}”` : `a model the banner could not look up (${modelId})`} —
        a graphics processing unit (GPU) job that runs for minutes to hours.
      </p>
      <p className={lineClass}>Graphics card: {gpuText(p.gpu, context.gpus)}</p>
      <p className={lineClass}>
        Training set: {p.train_dataset_id ? viewText(String(p.train_dataset_id), context) : 'none given'}
      </p>
      <p className={lineClass}>
        Evaluation sets:{' '}
        {evalIds.length === 0 ? 'none' : evalIds.map((id) => viewText(id, context)).join('; ')}
      </p>
      <p className={lineClass}>
        Calibration set:{' '}
        {p.calibration_dataset_id ? viewText(String(p.calibration_dataset_id), context) : 'none'}
      </p>
      <p className={lineClass}>
        Settings:{' '}
        {config && Object.keys(config).length > 0
          ? Object.entries(config)
              .map(([k, v]) => `${CONFIG_LABELS[k] ?? k}: ${typeof v === 'string' ? v : JSON.stringify(v)}`)
              .join(' · ')
          : 'the defaults'}
      </p>
      <AllArgumentsCollapsed payload={payload} />
    </div>
  );
}

function AllArgumentsCollapsed({ payload }: { payload: Record<string, unknown> }) {
  return (
    <details className="text-xs text-slate-500" data-testid="approval-arguments">
      <summary className="cursor-pointer">Show every argument the agent sent</summary>
      <pre className="mt-1 font-mono break-all whitespace-pre-wrap">{JSON.stringify(payload, null, 2)}</pre>
    </details>
  );
}

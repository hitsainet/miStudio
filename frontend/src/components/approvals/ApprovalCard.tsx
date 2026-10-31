/**
 * One waiting request as the shared card (ENH-008 §2.1), used by the banner and the Approvals page.
 *
 * Field order: action + Waiting pill; asked by; when (asked, expires) with the exact instant as
 * tooltips; facts; the extra card for kinds that need one (a probe run, operator code — D4); every
 * argument the agent sent, collapsed; the request id with a copy button; Approve and Reject. Reject
 * opens an inline reason (optional, at most 1000 characters).
 *
 * ⚠ EVERY FIELD FROM A REQUEST IS A TEXT NODE. Objects go through `JSON.stringify` and are still
 * rendered as text; nothing here uses `dangerouslySetInnerHTML` (§2.7; proven by
 * `ApprovalCard.hostile.test.tsx`, a render test, not a source grep).
 */
import { forwardRef, useEffect, useState } from 'react';
import { Check, Copy, X } from 'lucide-react';
import { APIError } from '../../api/client';
import { getCodeRuleState } from '../../api/featureGroups';
import { getModel } from '../../api/models';
import { probeMonitorsApi } from '../../api/probeMonitors';
import { getGPUList } from '../../api/system';
import { DEFAULT_REJECT_REASON, REJECT_REASON_MAX, useApprovalsStore } from '../../stores/approvalsStore';
import type { ApprovalRequest } from '../../types/featureGroups';
import type { ProbeDataset } from '../../types/probeMonitor';
import type { GPUInfo } from '../../types/system';
import { formatExpiresIn, formatLocal } from '../../utils/formatLocal';
import {
  actionLabel,
  approveLabel,
  askedBy,
  CODE_RULE_TOOL,
  PROBE_RUN_TOOL,
  STEERING_TOOLS,
} from './approvalLabels';

/** Text, always: a string as is, anything else as JSON. */
export function asText(value: unknown): string {
  if (typeof value === 'string') return value;
  if (value === undefined) return '';
  return JSON.stringify(value);
}

/**
 * A one-line sentence for the kinds that have one, or `null` (the card then lists the facts and
 * every argument). Never cut: a request the operator cannot read in full is one they approve blind.
 */
export function renderSummary(request: ApprovalRequest): string | null {
  if (STEERING_TOOLS.has(request.tool_name)) {
    const payload = request.payload as { prompt?: unknown; selected_features?: unknown[]; strength_values?: unknown[] };
    const featureCount = payload.selected_features?.length ?? (payload.strength_values ? 1 : 0);
    return `${featureCount} feature${featureCount === 1 ? '' : 's'} · "${asText(payload.prompt ?? '')}"`;
  }
  return null;
}

const FIELD_LABELS: Record<string, string> = {
  model_id: 'Model',
  probe_id: 'Probe',
  dataset_ids: 'Probe datasets',
  model: 'Judge model',
  endpoint: 'Judge endpoint',
  prompt_id: 'Judge prompt',
  max_rows_per_set: 'Rows per set at most',
  behaviour: 'Behaviour',
  size: 'Turns',
  window: 'Window',
  target_fpr: 'Target false-positive rate',
  recut_windows: 'Re-score the windows',
  sample_rows: 'Rows sampled per set',
  prompt: 'Prompt',
  sae_id: 'Sparse autoencoder (SAE)',
};

/** The facts list (§2.1 row 4): known labels in this order, then unknown keys. */
export function factsFor(request: ApprovalRequest): [string, string][] {
  const p = request.payload ?? {};
  if (request.tool_name === CODE_RULE_TOOL) {
    const doc = (p.document ?? {}) as { functions?: { name?: unknown }[] };
    return [
      ['Code hash', asText(p.code_sha256)],
      ['Functions', (doc.functions ?? []).map((f) => asText(f.name)).join(', ') || '—'],
    ];
  }
  if (request.tool_name === PROBE_RUN_TOOL) return [];
  if (STEERING_TOOLS.has(request.tool_name)) {
    const summary = renderSummary(request);
    return summary === null ? [] : [['Steering', summary]];
  }
  const known = Object.keys(FIELD_LABELS).filter((k) => k in p);
  const unknown = Object.keys(p).filter((k) => !(k in FIELD_LABELS));
  return [...known, ...unknown].map((k) => [FIELD_LABELS[k] ?? k, asText(p[k])]);
}

/** What the card looked up to name the things a probe run uses. */
export interface ProbeRunContext {
  modelNames: Record<string, string>;
  views: Record<string, ProbeDataset>;
  /** `null` = the card list could not be read. */
  gpus: GPUInfo[] | null;
}

export const EMPTY_CONTEXT: ProbeRunContext = { modelNames: {}, views: {}, gpus: null };

/**
 * Resolve names for waiting probe runs only: a queue with none of them makes no extra requests.
 * Every lookup may fail; the summary then names the raw id and says it could not be resolved.
 */
export function useProbeRunContext(pending: ApprovalRequest[]): ProbeRunContext {
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
        views: Object.fromEntries((views ?? []).map((v) => [v.id, v])),
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

function Time({ iso, prefix }: { iso: string | null | undefined; prefix: string }) {
  const t = formatLocal(iso);
  return (
    <span title={t.title || undefined}>
      {prefix} {t.text}
    </span>
  );
}

function decisionError(e: unknown): string {
  if (e instanceof APIError && e.status === 409) {
    if (/expired/i.test(e.detail)) return 'This request expired. Ask the agent to try again.';
    return `${e.detail} The list has been refreshed.`;
  }
  if (e instanceof APIError) return e.detail;
  if (e instanceof Error) return e.message;
  return String(e);
}

interface ApprovalCardProps {
  request: ApprovalRequest;
  context: ProbeRunContext;
  /** Called after a successful decision, so the list can move focus to the next card. */
  onDecided?: () => void;
}

export const ApprovalCard = forwardRef<HTMLDivElement, ApprovalCardProps>(function ApprovalCard(
  { request, context, onDecided },
  ref,
) {
  const approve = useApprovalsStore((s) => s.approve);
  const reject = useApprovalsStore((s) => s.reject);
  const [busy, setBusy] = useState(false);
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const label = actionLabel(request.tool_name);
  const facts = factsFor(request);
  const expires = formatLocal(request.expires_at);

  const decide = async (kind: 'approve' | 'reject') => {
    setBusy(true);
    setError(null);
    try {
      if (kind === 'approve') await approve(request.id);
      else await reject(request.id, reason);
      onDecided?.();
    } catch (e) {
      setError(decisionError(e));
      // Focus stays on this card after a failed decision (§2.7).
      if (ref && typeof ref !== 'function') ref.current?.focus();
    } finally {
      setBusy(false);
    }
  };

  const copyId = async () => {
    try {
      await navigator.clipboard?.writeText(request.id);
      setCopied(true);
    } catch {
      setCopied(false);
    }
  };

  return (
    <div
      ref={ref}
      tabIndex={-1}
      className="bg-white dark:bg-slate-900/60 border border-slate-200 dark:border-slate-700 rounded-control px-3 py-2 space-y-1 focus:outline-hidden focus:ring-2 focus:ring-amber-500"
      data-testid="approval-request"
      data-approval-id={request.id}
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm font-semibold text-slate-900 dark:text-slate-100" data-testid="approval-action">
          {label}
        </span>
        <span className="text-[11px] px-1.5 py-0.5 rounded-full bg-amber-100 dark:bg-amber-900/60 text-amber-800 dark:text-amber-200 border border-amber-300 dark:border-amber-700">
          Waiting
        </span>
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-asked-by">
        {askedBy(request.requested_by)}
      </p>
      <p className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-when">
        <Time iso={request.created_at} prefix="asked" />
        {' · '}
        <span title={expires.title || undefined}>{formatExpiresIn(request.expires_at)}</span>
      </p>
      {facts.length > 0 && (
        <dl className="text-xs grid grid-cols-[auto_1fr] gap-x-3 gap-y-0.5" data-testid="approval-facts">
          {facts.map(([k, v]) => (
            <div key={k} className="contents">
              <dt className="text-slate-500">{k}</dt>
              <dd className="text-slate-800 dark:text-slate-200 break-words whitespace-pre-wrap">{v}</dd>
            </div>
          ))}
        </dl>
      )}
      {request.tool_name === PROBE_RUN_TOOL && <ProbeRunSummary payload={request.payload} context={context} />}
      {request.tool_name === CODE_RULE_TOOL && <CodeRuleApprovalCard payload={request.payload} />}
      <AllArguments payload={request.payload} />
      <p className="text-[11px] text-slate-500 flex items-center gap-1">
        <span className="font-mono" data-testid="approval-id">{request.id}</span>
        <button
          type="button"
          onClick={() => void copyId()}
          className="p-0.5 rounded-control hover:bg-slate-200 dark:hover:bg-slate-700"
          aria-label={`Copy request id ${request.id}`}
          title="Copy request id"
        >
          <Copy className="w-3 h-3" />
        </button>
        {copied && <span>copied</span>}
      </p>
      {error && (
        <p className="text-xs text-red-600 dark:text-red-400" role="alert" data-testid="approval-error">
          {error}
        </p>
      )}
      {rejecting ? (
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="text"
            value={reason}
            maxLength={REJECT_REASON_MAX}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Reason (optional)"
            aria-label="Reason (optional)"
            className="flex-1 min-w-48 text-xs px-2 py-1 rounded-control border border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800"
          />
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide('reject')}
            className="px-2.5 py-1 text-xs bg-red-700 hover:bg-red-600 text-white rounded-control disabled:opacity-50"
            title={`Left blank, the reason sent is "${DEFAULT_REJECT_REASON}"`}
          >
            Confirm reject
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => {
              setRejecting(false);
              setReason('');
            }}
            className="px-2.5 py-1 text-xs bg-slate-200 dark:bg-slate-700 text-slate-800 dark:text-slate-200 rounded-control disabled:opacity-50"
          >
            Cancel
          </button>
        </div>
      ) : (
        <div className="flex flex-wrap gap-2">
          <button
            type="button"
            disabled={busy}
            onClick={() => void decide('approve')}
            className="flex items-center gap-1 px-2.5 py-1 text-xs bg-emerald-700 hover:bg-emerald-600 text-white rounded-control disabled:opacity-50"
          >
            <Check className="w-3 h-3" /> {approveLabel(request.tool_name)}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => setRejecting(true)}
            className="flex items-center gap-1 px-2.5 py-1 text-xs bg-slate-200 dark:bg-slate-700 hover:bg-slate-300 dark:hover:bg-slate-600 text-slate-800 dark:text-slate-200 rounded-control disabled:opacity-50"
          >
            <X className="w-3 h-3" /> Reject
          </button>
        </div>
      )}
    </div>
  );
});

/** Every argument, in full, wrapped, collapsed (§2.1 row 6) — never cut at a length. */
export function AllArguments({ payload }: { payload: Record<string, unknown> }) {
  const entries = Object.entries(payload ?? {});
  return (
    <details className="text-xs text-slate-600 dark:text-slate-400" data-testid="approval-arguments">
      <summary className="cursor-pointer">All arguments the agent sent ({entries.length})</summary>
      {entries.length === 0 ? (
        <p className="mt-1">No arguments.</p>
      ) : (
        <dl className="mt-1 space-y-0.5">
          {entries.map(([key, value]) => (
            <div key={key} className="flex gap-2">
              <dt className="font-mono text-slate-500 shrink-0">{key}</dt>
              <dd className="font-mono break-all whitespace-pre-wrap">{asText(value)}</dd>
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
 * A probe training run, in plain words: WHAT IT DETECTS first (the behaviour, which is what tells
 * several waiting runs on one model apart), then the model, the graphics card BY NAME AND
 * IDENTIFIER, every dataset view by name with its row counts, and the configuration.
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
              .map(([k, v]) => `${CONFIG_LABELS[k] ?? k}: ${asText(v)}`)
              .join(' · ')
          : 'the defaults'}
      </p>
    </div>
  );
}

interface CodeRuleCaseResult {
  id: string;
  expect: Record<string, unknown>;
  got: Record<string, unknown> | null;
  same_twice: boolean;
}

type LiveState = { state: string; revoked_row_id?: number | null } | 'loading' | 'unreadable';

/**
 * Operator code (D4; Feature 040 S4): what the operator reads BEFORE letting it run — the hash, the
 * functions, the modules it imports (the runner's own inspection), its cases as run twice in the
 * runner, the FULL source, never cut, and the hash's CURRENT approval state, read when the card is
 * shown. The inspection and case results are a snapshot taken when the agent asked, and are
 * labelled so. Approving records a durable per-hash approval; revoking is on the behaviour's
 * rule-set screen. This card never says approving undoes a revocation: the revocation stays in the
 * hash's history whatever is decided here.
 */
export function CodeRuleApprovalCard({ payload }: { payload: Record<string, unknown> }) {
  const sha = asText(payload.code_sha256 ?? '');
  const document = (payload.document ?? {}) as {
    source?: unknown;
    functions?: { name?: unknown }[];
    derived_from?: { statement?: unknown };
  };
  const facts = (payload.facts ?? {}) as {
    inspect?: { functions?: unknown[]; imports?: unknown[]; reason?: unknown };
    case_results?: CodeRuleCaseResult[];
    midataworks_state?: unknown;
    requested_by?: unknown;
  };
  const imports = facts.inspect?.imports;
  const [live, setLive] = useState<LiveState>('loading');

  useEffect(() => {
    let cancelled = false;
    if (!/^[0-9a-f]{64}$/.test(sha)) {
      setLive('unreadable');
      return;
    }
    getCodeRuleState(sha)
      .then((s) => !cancelled && setLive({ state: s.state, revoked_row_id: s.revoked_row_id }))
      .catch(() => !cancelled && setLive('unreadable'));
    return () => {
      cancelled = true;
    };
  }, [sha]);

  let stateText: string;
  if (live === 'loading') stateText = 'reading…';
  else if (live === 'unreadable') stateText = 'could not be read';
  else if (live.state === 'pending' && live.revoked_row_id)
    stateText = 'waiting for this decision; this code was revoked earlier, and that revocation stays in its history';
  else if (live.state === 'revoked') stateText = 'revoked';
  else stateText = live.state;

  return (
    <div className="mt-1 space-y-1 text-xs text-slate-600 dark:text-slate-400" data-testid="code-rule-approval">
      <p>
        Run operator code <span className="font-mono break-all">{sha}</span> in miStudio&apos;s isolated runner
        {facts.requested_by ? ` (asked by ${asText(facts.requested_by)})` : ''}.
      </p>
      <p data-testid="code-rule-state">Current approval state of this code: {stateText}</p>
      <p>Functions: {(document.functions ?? []).map((f) => asText(f.name)).join(', ') || '—'}</p>
      <p className="text-slate-500">Snapshot taken when the agent asked:</p>
      <p>Imports: {Array.isArray(imports) ? (imports.length ? imports.map(asText).join(', ') : 'none') : 'not inspected'}</p>
      {document.derived_from?.statement ? <p>Derived from: {asText(document.derived_from.statement)}</p> : null}
      <p>miDataworks approval: {facts.midataworks_state ? asText(facts.midataworks_state) : 'not known'} (information only; this miStudio decides)</p>
      <ul data-testid="code-rule-cases">
        {(facts.case_results ?? []).map((c) => (
          <li key={asText(c.id)}>
            case {asText(c.id)}: expected {asText(c.expect)}, got {asText(c.got)}
            {c.same_twice ? ', same on two fresh instances' : ', DIFFERED between instances'}
          </li>
        ))}
      </ul>
      <details open>
        <summary className="cursor-pointer">Source (read it before approving)</summary>
        <pre className="whitespace-pre-wrap font-mono bg-slate-900 text-slate-100 p-2 rounded-control" data-testid="code-rule-source">
          {asText(document.source ?? '')}
        </pre>
      </details>
    </div>
  );
}

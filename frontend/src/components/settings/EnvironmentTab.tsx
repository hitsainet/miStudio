/**
 * Settings → Environment (Feature 036; ENH-003 §2 "Consistent UI", §3.1, D8).
 *
 * Origin: miForge frontend/src/components/panels/settings/SettingsPage.tsx @ 655c731 (F12),
 * re-built on miStudio's Tailwind classes, with the four gaps ENH-003 §3.4 names closed here
 * from the start: a plain-language label and description, the effect-timing badge, the
 * environment value beside the value in force, and who changed it last and when.
 *
 * - **Changeable while miStudio runs**: the backend's `RUNTIME_KEYS`. Save stores an override
 *   (`PUT`); "Use the environment value" removes it (`DELETE`). The backend validates every
 *   value against the field's own limits and its refusal is shown on the row.
 * - **Needs a restart**: read-only, with the pod to restart.
 * - **From the environment**: read-only; secrets show only whether they are set.
 * - **Recent changes**: the last 50 history rows (D12).
 *
 * The editor for a row follows the value's type in the payload, so the tab holds no copy of the
 * backend's field list. The one closed list is the log level, whose five values the backend's
 * `Literal` fixes.
 *
 * The tab is rendered inside `<PinGate>` by SettingsPanel (D8): a convenience lock, not
 * authentication — the API itself has no login (suite decision 2026-10-06).
 */
import { useCallback, useEffect, useState } from 'react';
import { RotateCcw, Save } from 'lucide-react';
import { APIError } from '../../api/client';
import {
  readEnvironment,
  readEnvironmentHistory,
  resetEnvironment,
  writeEnvironment,
  type EnvironmentChange,
  type EnvironmentListing,
  type EnvironmentRow,
} from '../../api/environment';
import { HelpLink } from '../help/HelpLink';
import { LOG_LEVELS, draftOf, editorFor, parseDraft, showValue } from './environmentFormat';

function when(iso: string | null | undefined): string {
  if (!iso) return '';
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? iso : date.toLocaleString();
}

function failureText(err: unknown): string {
  if (err instanceof APIError) return err.detail;
  if (err instanceof Error) return err.message;
  return 'The request failed';
}

const inputClass =
  'w-full bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm px-3 py-2 text-sm text-slate-800 dark:text-slate-200 focus:border-emerald-500 focus:outline-hidden';

function RuntimeRow({
  name,
  row,
  onChanged,
}: {
  name: string;
  row: EnvironmentRow;
  onChanged: () => Promise<void>;
}) {
  const editor = editorFor(name, row.env_value ?? row.value);
  const [draft, setDraft] = useState(draftOf(row.value));
  const [reason, setReason] = useState('');
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    setDraft(draftOf(row.value));
  }, [row.value]);

  const send = async (action: () => Promise<unknown>) => {
    setBusy(true);
    setProblem(null);
    try {
      await action();
      setReason('');
      await onChanged();
    } catch (err) {
      setProblem(failureText(err));
    } finally {
      setBusy(false);
    }
  };

  const save = () => {
    const parsed = parseDraft(editor, draft);
    if ('error' in parsed) {
      setProblem(parsed.error);
      return;
    }
    void send(() => writeEnvironment(name, parsed.value, reason));
  };

  const testId = `env-input-${name}`;
  const control =
    editor === 'boolean' ? (
      <select value={draft} onChange={(e) => setDraft(e.target.value)} aria-label={row.label}
              data-testid={testId} className={inputClass}>
        <option value="true">On</option>
        <option value="false">Off</option>
      </select>
    ) : editor === 'log-level' ? (
      <select value={draft} onChange={(e) => setDraft(e.target.value)} aria-label={row.label}
              data-testid={testId} className={inputClass}>
        {LOG_LEVELS.map((level) => (
          <option key={level} value={level}>{level}</option>
        ))}
      </select>
    ) : editor === 'json' ? (
      <textarea value={draft} onChange={(e) => setDraft(e.target.value)} aria-label={row.label}
                data-testid={testId} rows={3} className={`${inputClass} font-mono`} />
    ) : (
      <input type={editor === 'number' ? 'number' : 'text'} value={draft}
             onChange={(e) => setDraft(e.target.value)} aria-label={row.label}
             placeholder={editor === 'text' ? 'not set' : ''}
             data-testid={testId} className={`${inputClass} ${editor === 'text' ? 'font-mono' : ''}`} />
    );

  const overridden = row.source === 'override';
  return (
    <li className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-4 flex flex-col gap-3"
        data-testid={`env-row-${name}`}>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-medium text-slate-800 dark:text-slate-200">{row.label}</div>
          <code className="text-xs text-slate-500">{row.env_name}</code>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {row.effect && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-sky-100 text-sky-800 dark:bg-sky-900/40 dark:text-sky-300"
                  data-testid={`env-effect-${name}`}>
              Takes effect: {row.effect}
            </span>
          )}
          <span className={`text-xs px-2 py-0.5 rounded-full ${overridden
            ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300'
            : 'bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300'}`}
                data-testid={`env-source-${name}`}>
            {overridden ? 'override' : 'environment'}
          </span>
        </div>
      </div>
      <p className="text-xs text-slate-600 dark:text-slate-400">{row.description}</p>
      <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-xs">
        <dt className="text-slate-500">In force</dt>
        <dd className="font-mono text-slate-800 dark:text-slate-200 break-all" data-testid={`env-value-${name}`}>
          {showValue(row.value)}
        </dd>
        <dt className="text-slate-500">Environment</dt>
        <dd className="font-mono text-slate-700 dark:text-slate-300 break-all">{showValue(row.env_value)}</dd>
        {row.last_change && (
          <>
            <dt className="text-slate-500">Last changed</dt>
            <dd className="text-slate-700 dark:text-slate-300" data-testid={`env-last-${name}`}>
              {row.last_change.action === 'reset' ? 'Reset' : 'Set'} by {row.last_change.actor},{' '}
              {when(row.last_change.at)}
            </dd>
          </>
        )}
      </dl>
      <div className="flex flex-wrap items-center gap-2">
        <div className="min-w-0 flex-1 basis-56">{control}</div>
        <input type="text" value={reason} onChange={(e) => setReason(e.target.value)}
               placeholder="Reason (optional)" aria-label={`Reason for changing ${row.label}`}
               maxLength={500} data-testid={`env-reason-${name}`}
               className={`${inputClass} basis-48 flex-1`} />
        <button type="button" disabled={busy} onClick={save} data-testid={`env-save-${name}`}
                className="flex items-center gap-1.5 px-3 py-2 bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-700 disabled:text-slate-500 text-white text-sm rounded-sm transition-colors">
          <Save className="w-4 h-4" /> Save
        </button>
        {overridden && (
          <button type="button" disabled={busy} onClick={() => void send(() => resetEnvironment(name, reason))}
                  data-testid={`env-reset-${name}`}
                  className="flex items-center gap-1.5 px-3 py-2 border border-slate-300 dark:border-slate-700 text-slate-700 dark:text-slate-300 hover:border-emerald-500 text-sm rounded-sm transition-colors">
            <RotateCcw className="w-4 h-4" /> Use the environment value
          </button>
        )}
      </div>
      {problem && <p className="text-xs text-red-500" role="alert" data-testid={`env-problem-${name}`}>{problem}</p>}
    </li>
  );
}

function ReadOnlyRows({
  rows,
  onOpenApiKeys,
}: {
  rows: [string, EnvironmentRow][];
  onOpenApiKeys?: () => void;
}) {
  return (
    <dl className="grid grid-cols-1 sm:grid-cols-[minmax(0,18rem)_1fr] gap-x-4 gap-y-3 text-sm">
      {rows.map(([name, row]) => (
        <div key={name} className="contents" data-testid={`env-fixed-${name}`}>
          <dt className="min-w-0">
            <div className="text-slate-800 dark:text-slate-200">{row.label}</div>
            <code className="text-xs text-slate-500 break-all">{row.env_name}</code>
          </dt>
          <dd className="min-w-0 text-slate-700 dark:text-slate-300">
            {row.secret ? (
              <span data-testid={`env-secret-${name}`}>
                {row.is_set ? 'set' : 'not set'}
                {onOpenApiKeys && (
                  <>
                    {' · '}
                    <button type="button" onClick={onOpenApiKeys}
                            className="text-emerald-700 dark:text-emerald-400 hover:underline">
                      Manage in API Keys
                    </button>
                  </>
                )}
              </span>
            ) : row.source === 'external' ? (
              <span className="text-slate-500">Set in the {row.restart}'s environment</span>
            ) : (
              <span className="font-mono break-all">{showValue(row.value)}</span>
            )}
            {row.description && <p className="text-xs text-slate-500 mt-0.5">{row.description}</p>}
            {row.restart && (
              <p className="text-xs text-amber-700 dark:text-amber-400 mt-0.5">
                Change it in the environment, then restart the {row.restart}.
                {row.restart_warning ? ` ${row.restart_warning}` : ''}
              </p>
            )}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function HistoryList({ changes }: { changes: EnvironmentChange[] }) {
  if (changes.length === 0) {
    return <p className="text-sm text-slate-500" data-testid="env-history-empty">No changes yet.</p>;
  }
  return (
    <ul className="flex flex-col gap-1 text-xs" data-testid="env-history">
      {changes.map((c) => (
        <li key={c.id} className="text-slate-700 dark:text-slate-300">
          <span className="text-slate-500">{when(c.changed_at)}</span>{' '}
          <code>{c.key}</code>{' '}
          {c.action === 'reset' ? 'reset to the environment value' : 'set'}:{' '}
          <span className="font-mono">{showValue(c.previous_value)}</span> →{' '}
          <span className="font-mono">{showValue(c.new_value)}</span> by {c.actor}
          {c.reason ? ` — ${c.reason}` : ''}
        </li>
      ))}
    </ul>
  );
}

export function EnvironmentTab({ onOpenApiKeys }: { onOpenApiKeys?: () => void } = {}) {
  const [listing, setListing] = useState<EnvironmentListing | null>(null);
  const [changes, setChanges] = useState<EnvironmentChange[]>([]);
  const [failure, setFailure] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [next, history] = await Promise.all([readEnvironment(), readEnvironmentHistory(50)]);
      setListing(next);
      setChanges(history.changes);
      setFailure(null);
    } catch (err) {
      setFailure(failureText(err));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const entries = Object.entries(listing?.settings ?? {});
  const runtime = entries.filter(([, row]) => row.group === 'runtime');
  const restart = entries.filter(([, row]) => row.group === 'restart');
  const fixed = entries.filter(([, row]) => row.group === 'environment');

  return (
    <div className="flex flex-col gap-8" data-testid="environment-tab">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <p className="text-sm text-slate-600 dark:text-slate-400 max-w-3xl">
          Where miStudio&apos;s server settings come from. An override is stored in miStudio&apos;s
          database and replaces the environment&apos;s value until you return to it; every change is
          kept in the history below.
        </p>
        <HelpLink topic="settings" />
      </div>
      {failure && (
        <p className="text-sm text-red-500" role="alert" data-testid="env-failure">{failure}</p>
      )}
      {listing && (
        <>
          <section>
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200 mb-1">
              Changeable while miStudio runs
            </h2>
            <p className="text-xs text-slate-500 mb-3" data-testid="env-runtime-count">
              {runtime.length} of {listing.count} settings. A job that is already running keeps the value it
              started with.
            </p>
            <ul className="flex flex-col gap-3" aria-label="Runtime settings">
              {runtime.map(([name, row]) => (
                <RuntimeRow key={name} name={name} row={row} onChanged={load} />
              ))}
            </ul>
          </section>
          <section>
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200 mb-3">Needs a restart</h2>
            <ReadOnlyRows rows={restart} />
          </section>
          <section>
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200 mb-3">From the environment</h2>
            <ReadOnlyRows rows={fixed} onOpenApiKeys={onOpenApiKeys} />
          </section>
          <section>
            <h2 className="text-base font-semibold text-slate-800 dark:text-slate-200 mb-3">Recent changes</h2>
            <HistoryList changes={changes} />
          </section>
        </>
      )}
    </div>
  );
}

/**
 * Publish a built definition to HuggingFace (033 FR-8, task 6.3).
 *
 * ⚠ THERE IS NO TOKEN FIELD HERE, DELIBERATELY. The token is resolved server-side from
 * Settings → API Keys, so no credential is held in this component's state, typed into a form that
 * a screenshot could capture, or sent in a request body. The endpoint returns 401 before queueing
 * any work when none is stored, so "no token" is a refusal the operator sees immediately rather
 * than a task that starts and fails at its last step.
 *
 * `AcquireLensCard` takes a token per field because it needs a READ credential and a WRITE
 * credential that are usually different. Publishing needs one write credential, which Settings
 * already holds, so asking again would only add a place for it to leak.
 */
import { useState } from 'react';
import { UploadCloud } from 'lucide-react';

interface PublishProbeDialogProps {
  probeName: string;
  rung: number;
  rungLanguage: string;
  busy?: boolean;
  error?: string | null;
  onCancel: () => void;
  onConfirm: (body: { repo_id: string; private: boolean }) => void;
}

export function PublishProbeDialog({
  probeName,
  rung,
  rungLanguage,
  busy = false,
  error = null,
  onCancel,
  onConfirm,
}: PublishProbeDialogProps) {
  const [repoId, setRepoId] = useState('');
  const [isPrivate, setIsPrivate] = useState(true);
  // `owner/name`, which is what the Hub requires. Checked here so a typo is a disabled button
  // rather than a 400 after the dialog closes.
  const wellFormed = /^[\w.-]+\/[\w.-]+$/.test(repoId.trim());

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 p-4"
      data-testid="probe-publish-dialog"
    >
      <div className="w-full max-w-lg rounded-lg border border-slate-700 bg-slate-900 p-5">
        <div className="flex items-start gap-3">
          <UploadCloud className="mt-0.5 h-5 w-5 shrink-0 text-emerald-400" />
          <div className="min-w-0">
            <h3 className="text-base font-semibold text-slate-100">Publish to HuggingFace</h3>
            <p className="mt-1 truncate text-sm text-slate-400" title={probeName}>
              {probeName}
            </p>
            <p className="mt-1 text-xs text-slate-400">
              rung {rung} — {rungLanguage}
            </p>
          </div>
        </div>

        <label className="mt-4 block text-xs font-medium uppercase tracking-wide text-slate-400">
          Repository
        </label>
        <input
          value={repoId}
          onChange={(event) => setRepoId(event.target.value)}
          placeholder="owner/probe-monitors"
          className="mt-1 w-full rounded border border-slate-700 bg-slate-950 p-2 font-mono text-sm text-slate-200 placeholder:text-slate-600"
          data-testid="probe-publish-repo"
        />
        <p className="mt-1 text-xs text-slate-500">
          Created if it does not exist. The repo's{' '}
          <span className="font-mono">manifest.json</span> is merged by name — publishing never
          removes a probe someone else put there.
        </p>

        <label className="mt-3 flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={isPrivate}
            onChange={(event) => setIsPrivate(event.target.checked)}
            data-testid="probe-publish-private"
          />
          Private
        </label>
        <p className="mt-1 text-xs text-slate-500">
          A definition names the datasets it was measured on and the concept it detects.
        </p>

        <p className="mt-3 rounded border border-slate-700 bg-slate-950/60 p-2 text-xs text-slate-400">
          The HuggingFace token comes from Settings → API Keys. It is never typed here and never
          leaves the server.
        </p>

        {error ? (
          <p
            className="mt-3 rounded border border-red-800 bg-red-950/40 p-2 text-xs text-red-300"
            data-testid="probe-publish-error"
          >
            {error}
          </p>
        ) : null}

        <div className="mt-4 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            className="rounded border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800"
            data-testid="probe-publish-cancel"
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={!wellFormed || busy}
            onClick={() => onConfirm({ repo_id: repoId.trim(), private: isPrivate })}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
            data-testid="probe-publish-confirm"
          >
            {busy ? 'Publishing…' : 'Publish'}
          </button>
        </div>
      </div>
    </div>
  );
}

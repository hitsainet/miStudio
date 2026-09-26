/**
 * The export flow on a probe's report (033 FR-11, task 6.1).
 *
 * Build → download → publish, with the evidence gate in front of the build. Everything the operator
 * needs to decide is on screen: the rung and its wording, what the last build recorded, and — when
 * the export is impossible — WHY, rather than a disabled button with no explanation.
 *
 * ⚠ AN SAE PROBE WITHOUT A PUBLISHED DICTIONARY CANNOT BE EXPORTED AT ALL, and that is not a
 * judgement call an acknowledgement can waive: a consumer has no way to encode without the
 * dictionary's location and revision. So the section says so and offers the action that fixes it
 * (push the SAE to HuggingFace) instead of a refusal the operator has to decode from a 422.
 */
import { useState } from 'react';
import { Download, FileJson, Hammer, UploadCloud, AlertTriangle } from 'lucide-react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import type { ProbeMonitorSummary, ProbePublication } from '../../types/probeMonitor';
import { AcknowledgeDialog } from './AcknowledgeDialog';
import { PublishProbeDialog } from './PublishProbeDialog';

interface ProbeExportSectionProps {
  probe: ProbeMonitorSummary;
  rungLanguage: string;
  rungNextStep?: string;
  /** From the probe row. Present only once a definition has been built. */
  definitionBuiltAt?: string | null;
  definitionSha256?: string | null;
  definitionBuild?: Record<string, unknown> | null;
  published?: ProbePublication[];
  /** The SAE's HuggingFace home, when this is a k-sparse probe. Absent ⇒ not exportable. */
  saeHfRepo?: string | null;
  /** Opens the SAE upload flow — the fix for the refusal above. */
  onPushSae?: (saeId: string) => void;
  onBuilt?: () => void;
}

export function ProbeExportSection({
  probe,
  rungLanguage,
  rungNextStep,
  definitionBuiltAt = null,
  definitionSha256 = null,
  definitionBuild = null,
  published = [],
  saeHfRepo = null,
  onPushSae,
  onBuilt,
}: ProbeExportSectionProps) {
  const [asking, setAsking] = useState(false);
  const [publishing, setPublishing] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const isSae = probe.variant === 'sae';
  const saeBlocked = isSae && !saeHfRepo;
  const needsAcknowledgement = probe.rung < 2;
  const built = Boolean(definitionBuiltAt);
  const invalidated = (definitionBuild as { invalidated?: { reason?: string } } | null)
    ?.invalidated;

  async function build(reason?: string) {
    setBusy(true);
    setError(null);
    setMessage(null);
    try {
      const accepted = await probeMonitorsApi.buildDefinition(probe.id, {
        acknowledgeReason: reason,
      });
      setMessage(`Build queued (${accepted.task_id}). The test vectors are real forward passes.`);
      setAsking(false);
      onBuilt?.();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  async function publish(body: { repo_id: string; private: boolean }) {
    setBusy(true);
    setError(null);
    try {
      const accepted = await probeMonitorsApi.publishDefinition(probe.id, body);
      setMessage(`Publish queued (${accepted.task_id}).`);
      setPublishing(false);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : String(caught));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section
      className="rounded-lg border border-slate-700 bg-slate-900/60 p-4"
      data-testid="probe-export-section"
    >
      <div className="flex items-center justify-between gap-3">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-slate-200">
          <FileJson className="h-4 w-4 text-slate-400" />
          Portable definition
        </h3>
        <span className="text-xs text-slate-500">mistudio.probe-definition/v1</span>
      </div>

      {saeBlocked ? (
        <div
          className="mt-3 rounded border border-amber-800 bg-amber-950/30 p-3"
          data-testid="probe-export-sae-blocked"
        >
          <p className="flex items-start gap-2 text-sm text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>
              This is a k-sparse probe over SAE features, and its dictionary has no HuggingFace
              location. A consumer cannot encode without it, so the definition cannot be built —
              an acknowledgement does not waive this.
            </span>
          </p>
          {probe.sae_id && onPushSae ? (
            <button
              type="button"
              onClick={() => onPushSae(probe.sae_id as string)}
              className="mt-2 rounded bg-amber-600 px-3 py-1.5 text-xs font-medium text-white"
              data-testid="probe-export-push-sae"
            >
              Publish the SAE first
            </button>
          ) : null}
        </div>
      ) : null}

      {invalidated ? (
        <p
          className="mt-3 rounded border border-sky-800 bg-sky-950/30 p-2 text-xs text-sky-200"
          data-testid="probe-export-invalidated"
        >
          The previous definition was invalidated ({String(invalidated.reason)}) because what it
          stated changed. Build again.
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <button
          type="button"
          disabled={saeBlocked || busy}
          onClick={() => (needsAcknowledgement ? setAsking(true) : build())}
          className="flex items-center gap-1.5 rounded bg-slate-700 px-3 py-1.5 text-sm text-slate-100 hover:bg-slate-600 disabled:cursor-not-allowed disabled:opacity-40"
          data-testid="probe-export-build"
        >
          <Hammer className="h-4 w-4" />
          {built ? 'Rebuild' : 'Build definition'}
        </button>

        <a
          href={built ? probeMonitorsApi.exportUrl(probe.id) : undefined}
          aria-disabled={!built}
          className={`flex items-center gap-1.5 rounded border px-3 py-1.5 text-sm ${
            built
              ? 'border-slate-600 text-slate-200 hover:bg-slate-800'
              : 'pointer-events-none border-slate-800 text-slate-600'
          }`}
          data-testid="probe-export-download"
        >
          <Download className="h-4 w-4" />
          Download
        </a>

        <button
          type="button"
          disabled={!built || busy}
          onClick={() => setPublishing(true)}
          className="flex items-center gap-1.5 rounded border border-emerald-700 px-3 py-1.5 text-sm text-emerald-300 hover:bg-emerald-950/40 disabled:cursor-not-allowed disabled:opacity-40"
          data-testid="probe-export-publish"
        >
          <UploadCloud className="h-4 w-4" />
          Publish
        </button>
      </div>

      {built ? (
        <dl
          className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs"
          data-testid="probe-export-built"
        >
          <dt className="text-slate-500">Built</dt>
          <dd className="text-slate-300">{definitionBuiltAt}</dd>
          <dt className="text-slate-500">sha256</dt>
          <dd className="truncate font-mono text-slate-300" title={definitionSha256 ?? ''}>
            {definitionSha256?.slice(0, 16)}…
          </dd>
          {definitionBuild?.resolved ? (
            <>
              <dt className="text-slate-500">Vectors</dt>
              <dd className="text-slate-300">
                {String((definitionBuild.resolved as Record<string, unknown>).vectors)}
              </dd>
              <dt className="text-slate-500">Size</dt>
              <dd className="text-slate-300">
                {String((definitionBuild.resolved as Record<string, unknown>).bytes)} bytes
              </dd>
            </>
          ) : null}
        </dl>
      ) : (
        <p className="mt-3 text-xs text-slate-500">
          No definition has been built yet.
        </p>
      )}

      {published.length ? (
        <div className="mt-3" data-testid="probe-export-published">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Published ({published.length})
          </p>
          <ul className="mt-1 space-y-1">
            {published.map((entry, index) => (
              <li key={`${entry.repo_id}-${index}`} className="text-xs text-slate-300">
                <a
                  href={`https://huggingface.co/${entry.repo_id}`}
                  target="_blank"
                  rel="noreferrer"
                  className="text-sky-400 hover:underline"
                >
                  {entry.repo_id}
                </a>{' '}
                <span className="text-slate-500">
                  {entry.private ? 'private' : 'public'} · {entry.revision?.slice(0, 8) ?? 'no sha'}{' '}
                  · {entry.at}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {message ? (
        <p className="mt-3 text-xs text-emerald-300" data-testid="probe-export-message">
          {message}
        </p>
      ) : null}
      {error ? (
        <p className="mt-3 text-xs text-red-300" data-testid="probe-export-error">
          {error}
        </p>
      ) : null}

      {asking ? (
        <AcknowledgeDialog
          rung={probe.rung}
          rungLanguage={rungLanguage}
          nextStep={rungNextStep}
          busy={busy}
          onCancel={() => setAsking(false)}
          onConfirm={(reason) => build(reason)}
        />
      ) : null}
      {publishing ? (
        <PublishProbeDialog
          probeName={probe.id}
          rung={probe.rung}
          rungLanguage={rungLanguage}
          busy={busy}
          error={error}
          onCancel={() => setPublishing(false)}
          onConfirm={publish}
        />
      ) : null}
    </section>
  );
}

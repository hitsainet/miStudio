/**
 * Export or publish the whole family (Feature 041 FR-23, FR-24, FR-28). All or nothing: a blocked member stops
 * the export and every blocker is listed with its reason. The ZIP holds each member's untouched built file;
 * publishing makes one Hugging Face commit (the token is the server's, never typed here — the
 * `PublishProbeDialog` rule).
 */
import { useState } from 'react';

import { useProbeFamiliesStore } from '../../stores/probeFamiliesStore';
import { PublishProbeDialog } from './PublishProbeDialog';

export function FamilyExportActions({ behaviourKey, model }: { behaviourKey: string; model: string | null }) {
  const { exportIndex, published, busy, refusal, loadExportIndex, archiveUrl, publishFamily } = useProbeFamiliesStore();
  const [publishing, setPublishing] = useState(false);
  const blockers = refusal?.code === 'family_export_blocked' ? refusal : null;
  return (
    <section aria-label="Family export" className="space-y-1 text-xs text-slate-300">
      <div className="flex gap-3">
        <button type="button" className="text-sky-400 underline" onClick={() => void loadExportIndex(behaviourKey, model)}>
          Check the family export
        </button>
        <a className="text-sky-400 underline" href={archiveUrl(behaviourKey, model)} download>
          Download the family (ZIP)
        </a>
        <button type="button" className="text-sky-400 underline" onClick={() => setPublishing(true)}>
          Publish the family
        </button>
      </div>
      {blockers ? (
        <div role="alert" className="text-amber-200" data-testid="family-blockers">
          <p>{blockers.message}</p>
          <ul>{blockers.next.map((n, i) => <li key={i}>{n.tool}: {n.why}</li>)}</ul>
        </div>
      ) : null}
      {exportIndex ? (
        <ul aria-label="Family export files">
          {exportIndex.members.map((m) => <li key={m.probe_id}>{m.filename} — {m.size} bytes, sha256 {m.sha256.slice(0, 12)}…</li>)}
        </ul>
      ) : null}
      {published ? <p role="status">Publishing {published.members.length} probe(s) in one commit (task {published.task_id}).</p> : null}
      {publishing ? (
        <PublishProbeDialog
          probeName={`${behaviourKey} family${model ? ` on ${model}` : ''}`}
          rung={2}
          rungLanguage="each member keeps its own rung"
          busy={busy}
          error={refusal && !blockers ? refusal.message : null}
          onCancel={() => setPublishing(false)}
          onConfirm={async (body) => {
            if (await publishFamily(behaviourKey, model, body.repo_id, body.private)) setPublishing(false);
          }}
        />
      ) : null}
    </section>
  );
}

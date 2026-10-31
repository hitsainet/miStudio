/**
 * One behaviour (Feature 037 FR-16): meaning and versions, the checklist, probes by role, judge prompts,
 * data (with the D48 confirmation), and the history. All text is rendered as text, never as HTML.
 */
import { useCallback, useEffect, useState } from 'react';

import { behavioursApi } from '../../api/behaviours';
import { APIError } from '../../api/client';
import type {
  BehaviourChecklist as Checklist,
  BehaviourDefinition,
  BehaviourDetail as Detail,
  BehaviourHistoryEntry,
  BehaviourVersionChange,
  BehaviourWriteResult,
  ProbeRole,
} from '../../types/behaviour';
import { BehaviourChecklist } from './BehaviourChecklist';
import { BehaviourForm } from './BehaviourForm';
import { DetectorSuccessReport, ModelComparison, ProbeBreakdown } from './DetectorReports';
import { RuleSetPanel } from './RuleSetPanel';
import { RulesEvaluation } from './RulesEvaluation';
import { FamilySection } from './FamilySection';
import { SpotCheckSection } from './SpotCheckSection';

interface Props {
  behaviourKey: string;
  onChanged: () => void;
  onOpenTab?: (tab: string) => void;
}

const ROLES: ProbeRole[] = ['primary', 'candidate', 'retired'];

const DIFF_FIELDS: (keyof BehaviourDefinition)[] = [
  'name', 'description', 'positive_meaning', 'negative_meaning', 'target_fpr', 'sub_questions', 'success_criteria',
  'held_out_columns', 'models', 'calibration_floor', 'rule_set',
];

/** The fields that differ between two versions (for the stale-base notice). */
export function definitionDiff(a: BehaviourDefinition, b: BehaviourDefinition): string[] {
  return DIFF_FIELDS.filter((field) => JSON.stringify(a[field] ?? null) !== JSON.stringify(b[field] ?? null)) as string[];
}

/** The version's detector data, read-only (FR-43). The rule set is shown with its validation state; it is
 * authored in miDataworks or sent inline through the API. */
function DetectorFieldsView({ definition }: { definition: BehaviourDefinition }) {
  const criteria = definition.success_criteria ?? [];
  return (
    <div className="mt-2 space-y-1 text-xs text-slate-300" aria-label="Detector data">
      {definition.sub_questions?.length ? (
        <p>Sub-questions: {definition.sub_questions.map((s) => s.key).join(', ')}</p>
      ) : null}
      {criteria.length ? (
        <ul aria-label="Success criteria">
          {criteria.map((c) => (
            <li key={c.id}>
              {c.id}: {c.detector}, {typeof c.subset === 'string' ? c.subset : `${c.subset.column} = ${String(c.subset.value)}`}, recall at
              least {c.min_recall} at false-positive rate {c.target_fpr ?? definition.target_fpr ?? '—'}
              {c.held_out_test ? `, held out by ${c.held_out_test}` : ', in distribution'}
              {c.probe_id ? `, probe ${c.probe_id}` : ''}
            </li>
          ))}
        </ul>
      ) : <p className="text-slate-500">No success criteria declared.</p>}
      {definition.held_out_columns?.length ? <p>Held-out columns: {definition.held_out_columns.join(', ')}</p> : null}
      {definition.models?.length ? <p>Models: {definition.models.map((m) => m.model_id).join(', ')}</p> : null}
      {definition.calibration_floor != null ? <p>Calibration floor: {definition.calibration_floor} negatives</p> : null}
      {definition.rule_set ? (
        <p data-testid="rule-set">
          Rule set {definition.rule_set.source_ref ? `${definition.rule_set.source_ref.name} v${definition.rule_set.source_ref.version}` : 'inline'}{' '}
          ({definition.rule_set.digest.slice(0, 12)}…):{' '}
          {definition.rule_set.validation
            ? definition.rule_set.validation.status
              + (definition.rule_set.validation.code ? ` — ${definition.rule_set.validation.code.replace(/_/g, ' ')}` : '')
            : (definition.rule_set.status ?? 'stored')}
          {definition.rule_set.reason ? ` — ${definition.rule_set.reason.replace(/_/g, ' ')}` : ''}
        </p>
      ) : null}
    </div>
  );
}

export function BehaviourDetail({ behaviourKey, onChanged, onOpenTab }: Props) {
  const [detail, setDetail] = useState<Detail | null>(null);
  const [checklist, setChecklist] = useState<Checklist | null>(null);
  const [versions, setVersions] = useState<BehaviourDefinition[]>([]);
  const [history, setHistory] = useState<BehaviourHistoryEntry[]>([]);
  const [editing, setEditing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  /** Feature 040 FR-3: the result of the last save (its diff and warnings), and a stale-base refusal. */
  const [saved, setSaved] = useState<BehaviourWriteResult | null>(null);
  const [stale, setStale] = useState<{ base: BehaviourDefinition; current: BehaviourDefinition } | null>(null);

  const load = useCallback(async () => {
    try {
      const [d, c, v, h] = await Promise.all([
        behavioursApi.get(behaviourKey),
        behavioursApi.checklist(behaviourKey),
        behavioursApi.versions(behaviourKey),
        behavioursApi.history(behaviourKey),
      ]);
      setDetail(d);
      setChecklist(c);
      setVersions(v.versions);
      setHistory(h.entries);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [behaviourKey]);

  useEffect(() => {
    void load();
  }, [load]);

  async function act(action: () => Promise<unknown>) {
    try {
      await action();
      await load();
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  if (!detail) return error ? <p className="text-sm text-red-400">{error}</p> : <p className="text-sm text-slate-400">Loading…</p>;
  const d = detail.definition;
  return (
    <div className="space-y-4" aria-label={`Behaviour ${detail.key}`}>
      <header>
        <h3 className="text-lg text-slate-100">{d.name} <span className="text-sm text-slate-400">({detail.key}, version {detail.current_version})</span></h3>
        {checklist ? <p className="text-xs text-slate-400">Status: {checklist.status}</p> : null}
        {error ? <p role="alert" className="text-sm text-red-400">{error}</p> : null}
      </header>

      <section>
        <h4 className="text-sm font-semibold text-slate-200">Meaning</h4>
        <p className="text-sm text-slate-300">Positive: {d.positive_meaning}</p>
        {d.negative_meaning ? <p className="text-sm text-slate-300">Negative: {d.negative_meaning}</p> : null}
        {d.description ? <p className="text-sm text-slate-400">{d.description}</p> : null}
        {d.target_fpr != null ? <p className="text-xs text-slate-400">Target false-positive rate: {d.target_fpr}</p> : null}
        {!detail.archived && !editing ? (
          <button type="button" className="mt-1 text-xs text-sky-400 underline" onClick={() => setEditing(true)}>Edit meaning</button>
        ) : null}
        {editing ? (
          <BehaviourForm
            current={d}
            initialKey={detail.key}
            onCancel={() => setEditing(false)}
            onSubmit={async (body) => {
              try {
                const result = await behavioursApi.newVersion(detail.key, body as BehaviourVersionChange);
                setSaved(result);
                setStale(null);
                setEditing(false);
                await load();
                onChanged();
              } catch (e) {
                if (e instanceof APIError && e.status === 409 && e.data?.code === 'stale_base_version') {
                  const fresh = await behavioursApi.get(detail.key);
                  setStale({ base: d, current: fresh.definition });
                } else {
                  setError(e instanceof Error ? e.message : String(e));
                }
              }
            }}
          />
        ) : null}
        {stale ? (
          <div role="alert" className="mt-1 rounded-control border border-amber-600 p-2 text-xs text-amber-200" data-testid="stale-base">
            This behaviour changed since you opened it (you edited version {stale.base.version}; it is now version{' '}
            {stale.current.version}). Reload to edit the current version.
            <ul className="mt-1">
              {definitionDiff(stale.base, stale.current).map((path) => <li key={path}>changed: {path}</li>)}
            </ul>
            <button type="button" className="text-sky-400 underline" onClick={() => { setStale(null); setEditing(false); void load(); }}>
              Reload
            </button>
          </div>
        ) : null}
        {saved?.diff?.length ? (
          <p className="mt-1 text-xs text-emerald-300" data-testid="saved-diff">
            Version {saved.current_version} saved; changed: {saved.diff.map((entry) => entry.path).join(', ')}
          </p>
        ) : null}
        {saved?.warnings?.length ? (
          <ul className="mt-1 text-xs text-amber-300" aria-label="Warnings">
            {saved.warnings.map((w, i) => <li key={i}>{w.message}{w.next?.length ? ` Next: ${w.next.map((n) => n.tool).join(', ')}` : ''}</li>)}
          </ul>
        ) : null}
        <DetectorFieldsView definition={d} />
        {versions.length > 1 ? (
          <details className="text-xs text-slate-400"><summary>{versions.length} versions</summary>
            <ul>{versions.map((v) => <li key={v.version}>v{v.version}: {v.name} — {v.positive_meaning}</li>)}</ul>
          </details>
        ) : null}
      </section>

      {checklist ? (
        <section>
          <h4 className="text-sm font-semibold text-slate-200">From definition to armed</h4>
          <BehaviourChecklist checklist={checklist} onOpenTab={onOpenTab} />
        </section>
      ) : null}

      {/* Feature 041 (FR-28): the family on one model — coverage, members, flags. */}
      <FamilySection behaviourKey={detail.key} />

      {/* Feature 040 S2 (FR-43): the reports, read from stored evidence; no GPU. */}
      <section className="space-y-2" aria-label="Detector reports">
        <DetectorSuccessReport behaviourKey={detail.key} />
        <ModelComparison behaviourKey={detail.key} />
        {detail.adopted.probes.length ? <ProbeBreakdown probes={detail.adopted.probes} /> : null}
        {/* Feature 040 S3 (FR-32 – FR-37, FR-43): the rule set, tried on a text, and rules with the probe. */}
        {detail.definition.rule_set ? <RuleSetPanel behaviourKey={detail.key} /> : null}
        {detail.adopted.probes.length ? <RulesEvaluation behaviourKey={detail.key} probes={detail.adopted.probes} /> : null}
      </section>

      <section>
        <h4 className="text-sm font-semibold text-slate-200">Probes</h4>
        {detail.adopted.probes.length === 0 ? <p className="text-xs text-slate-500">No probe carries this key yet.</p> : null}
        <ul className="space-y-1">
          {detail.adopted.probes.map((p) => (
            <li key={p.id} className="flex items-center gap-2 text-sm text-slate-300">
              <span className="font-mono text-xs">{p.id}</span>
              <span className="text-xs">layer {p.layer}, {p.rule}, rung {p.rung}</span>
              {!p.training_view_confirmed ? <span className="text-xs text-amber-300">training labels not confirmed</span> : null}
              <select
                aria-label={`Role of ${p.id}`}
                value={p.role}
                disabled={detail.archived}
                onChange={(e) => void act(() => behavioursApi.setRole(detail.key, p.id, e.target.value as ProbeRole))}
                className="rounded-control border border-slate-700 bg-slate-900 text-xs"
              >
                {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
              </select>
            </li>
          ))}
        </ul>
      </section>

      <section>
        <h4 className="text-sm font-semibold text-slate-200">Judge prompts</h4>
        <ul className="text-sm text-slate-300">
          {detail.adopted.prompts.map((p) => <li key={p.id}>{p.title} ({p.key}){p.archived ? ' — archived' : ''}</li>)}
        </ul>
      </section>

      <section>
        <h4 className="text-sm font-semibold text-slate-200">Data</h4>
        <ul className="space-y-1 text-sm text-slate-300">
          {detail.adopted.views.map((v) => (
            <li key={v.id} className="flex items-center gap-2">
              <span>{v.name}</span>
              <span className="text-xs text-slate-400">{v.use}{v.dataset_version?.dataset_name ? `, miDataworks ${v.dataset_version.dataset_name} v${v.dataset_version.version_number}` : ''}</span>
              {v.confirmed_for_this_behaviour ? (
                <span className="text-xs text-emerald-400">positive labels confirmed</span>
              ) : v.role !== 'calibration' ? (
                <button type="button" className="text-xs text-sky-400 underline" onClick={() => void act(() => behavioursApi.confirmPositive(v.id, detail.key))}>
                  Confirm positive labels are {detail.key}
                </button>
              ) : null}
              {v.linked_by === 'link' ? (
                <button type="button" className="text-xs text-slate-400 underline" onClick={() => void act(() => behavioursApi.unlinkView(detail.key, v.id))}>Unlink</button>
              ) : null}
            </li>
          ))}
        </ul>
      </section>

      {!detail.archived ? <SpotCheckSection behaviourKey={detail.key} prompts={detail.adopted.prompts} /> : null}

      <section>
        <h4 className="text-sm font-semibold text-slate-200">History</h4>
        <ul className="text-xs text-slate-400">
          {history.map((h) => (
            <li key={h.id}>{h.created_at} — {h.actor}: {h.field} {h.subject_id ?? ''}{h.reason ? ` (${h.reason})` : ''}</li>
          ))}
        </ul>
      </section>

      <div className="flex gap-2">
        {detail.archived ? (
          <button type="button" className="text-xs text-sky-400 underline" onClick={() => void act(() => behavioursApi.unarchive(detail.key))}>Unarchive</button>
        ) : (
          <button type="button" className="text-xs text-slate-400 underline" onClick={() => void act(() => behavioursApi.archive(detail.key))}>Archive</button>
        )}
      </div>
    </div>
  );
}

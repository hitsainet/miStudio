/**
 * The behaviour's probe family on one model (Feature 041 FR-2 – FR-4, FR-28): coverage first, with the
 * missing sub-questions at the top and the call that fills each; then every member with its facts and its
 * flags in words. Every word about a member comes from the server's body; the rung's wording is the
 * server's (`RungChip`).
 */
import { useEffect } from 'react';

import { useProbeFamiliesStore } from '../../stores/probeFamiliesStore';
import type { FamilyFlag } from '../../types/probeFamily';
import { BlendReportList } from './BlendReportList';
import { RefusalNotice } from './DetectorReports';
import { FamilyExportActions } from './FamilyExportActions';
import { RungChip } from './RungChip';
import { SuggestedBlendCard } from './SuggestedBlendCard';

const FIELD_WORDS: Record<string, string> = {
  precision: 'precision',
  render_form: 'render form',
  scope: 'scope',
  training_dataset_version: 'dataset version',
};

const shown = (value: unknown) => (value == null ? 'not recorded' : String(value).replace(/_/g, ' '));

/** A flag as a sentence: "differs in precision: float32 vs bfloat16". */
export function flagWords(flag: FamilyFlag): string {
  if (flag.flag.startsWith('differs_') && flag.field) {
    return `differs in ${FIELD_WORDS[flag.field] ?? flag.field}: ${shown(flag.value)} vs ${shown(flag.majority)}` +
      (flag.majority_tie ? ' (no clear majority)' : '');
  }
  if (flag.flag === 'definition_stale') return `definition not current (${shown(flag.reason)})`;
  if (flag.flag === 'sub_question_not_declared') {
    return `sub-question ${flag.sub_question} is not declared by version ${flag.version ?? '?'}; left out of the default blend`;
  }
  return shown(flag.flag);
}

export function FamilySection({ behaviourKey }: { behaviourKey: string }) {
  const { family, models, refusal, loading, includeRetired, loadFamily } = useProbeFamiliesStore();

  useEffect(() => {
    void loadFamily(behaviourKey);
  }, [behaviourKey, loadFamily]);

  const pickable = refusal?.code === 'model_required' ? models : family && family.models.length > 1 ? family.models : [];
  return (
    <section aria-label="Probe family" className="space-y-2">
      <h4 className="text-sm font-semibold text-slate-200">Probe family</h4>
      {pickable.length ? (
        <label className="text-xs text-slate-300">
          Model{' '}
          <select
            aria-label="Family model"
            value={family?.model ?? ''}
            onChange={(e) => void loadFamily(behaviourKey, e.target.value)}
            className="rounded-sm border border-slate-700 bg-slate-900 text-xs"
          >
            <option value="" disabled>choose a model</option>
            {pickable.map((m) => (
              <option key={m.model_id} value={m.model_id}>{m.model_id} ({m.members} members)</option>
            ))}
          </select>
        </label>
      ) : null}
      {refusal?.code === 'model_required' ? (
        <p className="text-xs text-amber-200" role="status">The behaviour has probes on several models; choose one.</p>
      ) : (
        <RefusalNotice refusal={refusal} />
      )}
      {loading ? <p className="text-xs text-slate-400">Loading…</p> : null}
      {family ? (
        <>
          <ul aria-label="Sub-question coverage" className="space-y-1 text-xs text-slate-300">
            {family.coverage.map((entry) => (
              <li key={entry.sub_question ?? '(main)'} data-testid={`coverage-${entry.sub_question ?? 'main'}`}>
                {entry.sub_question == null ? 'Main question' : `Sub-question ${entry.sub_question}`}:{' '}
                {entry.state === 'missing' ? (
                  <span className="text-amber-300">missing</span>
                ) : entry.members.length ? `${entry.members.length} member(s)` : 'no member'}
                {entry.next?.length ? (
                  <span className="text-slate-400"> — next: {entry.next.map((n) => n.tool).join(', ')} ({entry.next[0].why})</span>
                ) : null}
              </li>
            ))}
          </ul>
          <table className="text-xs text-slate-300" aria-label="Family members">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="pr-2">Probe</th><th className="pr-2">Role</th><th className="pr-2">Question</th>
                <th className="pr-2">Layer, rule</th><th className="pr-2">Rung</th><th>Flags</th>
              </tr>
            </thead>
            <tbody>
              {family.members.map((m) => (
                <tr key={m.probe_id} data-testid={`member-${m.probe_id}`}>
                  <td className="pr-2 font-mono">{m.probe_id}</td>
                  <td className="pr-2">{m.role}{m.default_blend ? '' : ' (not in the default blend)'}</td>
                  <td className="pr-2">{m.sub_question ?? 'main question'}</td>
                  <td className="pr-2">layer {m.layer}, {m.rule}</td>
                  <td className="pr-2"><RungChip rung={m.rung} language={m.rung_language} /></td>
                  <td>{m.flags.length ? m.flags.map(flagWords).join('; ') : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-slate-400">
            {family.total} member(s) on {family.model ?? 'no model'}; flags compare each member with the family&apos;s majority.
            {' '}
            <button type="button" className="text-sky-400 underline"
              onClick={() => void loadFamily(behaviourKey, family.model, 0, !includeRetired)}>
              {includeRetired ? 'Hide retired' : 'Show retired'}
            </button>
            {family.next_offset != null ? (
              <button type="button" className="ml-2 text-sky-400 underline"
                onClick={() => void loadFamily(behaviourKey, family.model, family.next_offset ?? 0)}>
                Next page
              </button>
            ) : null}
          </p>
          {/* Feature 041 S2 (FR-28): the family's blend reports and its suggested blend on this model. */}
          <BlendReportList behaviourKey={behaviourKey} model={family.model} />
          <SuggestedBlendCard behaviourKey={behaviourKey} model={family.model} />
          {/* Feature 041 S3 (FR-23, FR-24): export or publish the family. */}
          <FamilyExportActions behaviourKey={behaviourKey} model={family.model} />
        </>
      ) : null}
    </section>
  );
}

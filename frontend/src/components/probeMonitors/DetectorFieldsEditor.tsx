/**
 * The structured editor for a behaviour's detector data (Feature 040 FR-1, FR-43; FTID §6).
 *
 * Sub-questions, success criteria, held-out columns, models and the calibration floor are edited as
 * fields, never as raw JSON. The vocabulary (detector forms, named subsets) is the server's fixed,
 * behaviour-neutral list (FR-5); every name and key shown comes from the data being edited.
 *
 * `toFields` turns the editor state into request values; `fromDefinition` reads a version. A field the
 * version does not declare stays ABSENT unless the operator adds something to it.
 */
import type {
  DetectorFields,
  DetectorForm,
  ModelRef,
  NamedSubset,
  SubQuestion,
  SuccessCriterion,
} from '../../types/behaviour';
import { DETECTOR_FORMS, NAMED_SUBSETS } from '../../types/behaviour';

export interface CriterionDraft {
  id: string;
  detector: DetectorForm;
  /** A named subset, or `column` for a {column, value} filter. */
  subsetKind: NamedSubset | 'column';
  subsetColumn: string;
  subsetValue: string;
  held_out_test: string;
  target_fpr: string;
  min_recall: string;
  probe_id: string;
}

export interface DetectorDraft {
  sub_questions: SubQuestion[];
  criteria: CriterionDraft[];
  held_out_columns: string;
  models: string;
  calibration_floor: string;
}

const split = (text: string) =>
  text
    .split(',')
    .map((part) => part.trim())
    .filter(Boolean);

export function fromDefinition(fields: DetectorFields | undefined): DetectorDraft {
  return {
    sub_questions: (fields?.sub_questions ?? []).map((s) => ({ ...s })),
    criteria: (fields?.success_criteria ?? []).map((c) => ({
      id: c.id,
      detector: c.detector,
      subsetKind: typeof c.subset === 'string' ? c.subset : 'column',
      subsetColumn: typeof c.subset === 'string' ? '' : c.subset.column,
      subsetValue: typeof c.subset === 'string' ? '' : String(c.subset.value),
      held_out_test: c.held_out_test ?? '',
      target_fpr: c.target_fpr != null ? String(c.target_fpr) : '',
      min_recall: String(c.min_recall),
      probe_id: c.probe_id ?? '',
    })),
    held_out_columns: (fields?.held_out_columns ?? []).join(', '),
    models: (fields?.models ?? []).map((m) => m.model_id).join(', '),
    calibration_floor: fields?.calibration_floor != null ? String(fields.calibration_floor) : '',
  };
}

export function criterionOut(c: CriterionDraft): SuccessCriterion {
  const out: SuccessCriterion = {
    id: c.id.trim(),
    detector: c.detector,
    subset: c.subsetKind === 'column' ? { column: c.subsetColumn.trim(), value: c.subsetValue } : c.subsetKind,
    min_recall: Number(c.min_recall),
  };
  if (c.held_out_test) out.held_out_test = c.held_out_test;
  if (c.target_fpr.trim()) out.target_fpr = Number(c.target_fpr);
  if (c.detector === 'named_probe' && c.probe_id.trim()) out.probe_id = c.probe_id.trim();
  return out;
}

/** The request values of the editor: each field present only when it holds something. */
export function toFields(draft: DetectorDraft): DetectorFields {
  const out: DetectorFields = {};
  if (draft.sub_questions.length) out.sub_questions = draft.sub_questions.map((s) => ({ ...s, key: s.key.trim() }));
  if (draft.criteria.length) out.success_criteria = draft.criteria.map(criterionOut);
  if (split(draft.held_out_columns).length) out.held_out_columns = split(draft.held_out_columns);
  if (split(draft.models).length) out.models = split(draft.models).map((model_id): ModelRef => ({ model_id }));
  if (draft.calibration_floor.trim()) out.calibration_floor = Number(draft.calibration_floor);
  return out;
}

const EMPTY_CRITERION: CriterionDraft = {
  id: '', detector: 'best_single', subsetKind: 'all', subsetColumn: '', subsetValue: '', held_out_test: '',
  target_fpr: '', min_recall: '', probe_id: '',
};

interface Props {
  draft: DetectorDraft;
  onChange: (draft: DetectorDraft) => void;
}

const input = 'w-full rounded-sm border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-100';
const small = 'rounded-sm border border-slate-700 bg-slate-900 px-1 py-0.5 text-xs text-slate-100';

export function DetectorFieldsEditor({ draft, onChange }: Props) {
  const set = (patch: Partial<DetectorDraft>) => onChange({ ...draft, ...patch });
  const setCriterion = (index: number, patch: Partial<CriterionDraft>) =>
    set({ criteria: draft.criteria.map((c, i) => (i === index ? { ...c, ...patch } : c)) });
  const setSub = (index: number, patch: Partial<SubQuestion>) =>
    set({ sub_questions: draft.sub_questions.map((s, i) => (i === index ? { ...s, ...patch } : s)) });
  const heldOut = split(draft.held_out_columns);

  return (
    <fieldset className="space-y-3 rounded-sm border border-slate-700 p-2" aria-label="Detector data">
      <legend className="px-1 text-xs uppercase text-slate-400">What the detector must achieve</legend>

      <div>
        <h5 className="text-xs text-slate-300">Sub-questions</h5>
        {draft.sub_questions.map((s, i) => (
          <div key={i} className="mt-1 grid grid-cols-1 gap-1 sm:grid-cols-3" aria-label={`Sub-question ${i + 1}`}>
            <input className={small} aria-label={`Sub-question ${i + 1} key`} placeholder="key" value={s.key} onChange={(e) => setSub(i, { key: e.target.value })} />
            <input className={small} aria-label={`Sub-question ${i + 1} question`} placeholder="question" value={s.question} onChange={(e) => setSub(i, { question: e.target.value })} />
            <div className="flex gap-1">
              <input className={small} aria-label={`Sub-question ${i + 1} positive meaning`} placeholder="what counts as positive" value={s.positive_meaning} onChange={(e) => setSub(i, { positive_meaning: e.target.value })} />
              <button type="button" className="text-xs text-slate-400 underline" onClick={() => set({ sub_questions: draft.sub_questions.filter((_, j) => j !== i) })}>Remove</button>
            </div>
          </div>
        ))}
        <button type="button" className="mt-1 text-xs text-sky-400 underline" onClick={() => set({ sub_questions: [...draft.sub_questions, { key: '', question: '', positive_meaning: '' }] })}>
          Add sub-question
        </button>
      </div>

      <label className="block text-xs text-slate-400">
        Held-out columns (comma-separated column names that define &quot;held out&quot;)
        <input className={input} aria-label="Held-out columns" value={draft.held_out_columns} onChange={(e) => set({ held_out_columns: e.target.value })} />
      </label>

      <div>
        <h5 className="text-xs text-slate-300">Success criteria</h5>
        {draft.criteria.map((c, i) => (
          <div key={i} className="mt-1 grid grid-cols-2 gap-1 rounded-sm border border-slate-800 p-1 sm:grid-cols-4" aria-label={`Criterion ${i + 1}`}>
            <input className={small} aria-label={`Criterion ${i + 1} id`} placeholder="id" value={c.id} onChange={(e) => setCriterion(i, { id: e.target.value })} />
            <select className={small} aria-label={`Criterion ${i + 1} detector`} value={c.detector} onChange={(e) => setCriterion(i, { detector: e.target.value as DetectorForm })}>
              {DETECTOR_FORMS.map((form) => <option key={form} value={form}>{form}</option>)}
            </select>
            <select className={small} aria-label={`Criterion ${i + 1} subset`} value={c.subsetKind} onChange={(e) => setCriterion(i, { subsetKind: e.target.value as CriterionDraft['subsetKind'] })}>
              {NAMED_SUBSETS.map((subset) => <option key={subset} value={subset}>{subset}</option>)}
              <option value="column">a column value</option>
            </select>
            {c.subsetKind === 'column' ? (
              <span className="flex gap-1">
                <input className={small} aria-label={`Criterion ${i + 1} subset column`} placeholder="column" value={c.subsetColumn} onChange={(e) => setCriterion(i, { subsetColumn: e.target.value })} />
                <input className={small} aria-label={`Criterion ${i + 1} subset value`} placeholder="value" value={c.subsetValue} onChange={(e) => setCriterion(i, { subsetValue: e.target.value })} />
              </span>
            ) : <span />}
            <select className={small} aria-label={`Criterion ${i + 1} held-out test`} value={c.held_out_test} onChange={(e) => setCriterion(i, { held_out_test: e.target.value })}>
              <option value="">in distribution (no held-out test)</option>
              {heldOut.map((column) => <option key={column} value={column}>{column}</option>)}
            </select>
            <input className={small} aria-label={`Criterion ${i + 1} target false-positive rate`} placeholder="target FPR (inherits)" value={c.target_fpr} onChange={(e) => setCriterion(i, { target_fpr: e.target.value })} />
            <input className={small} aria-label={`Criterion ${i + 1} minimum recall`} placeholder="min recall, 0 to 1" value={c.min_recall} onChange={(e) => setCriterion(i, { min_recall: e.target.value })} />
            {c.detector === 'named_probe' ? (
              <input className={small} aria-label={`Criterion ${i + 1} probe`} placeholder="probe id" value={c.probe_id} onChange={(e) => setCriterion(i, { probe_id: e.target.value })} />
            ) : null}
            <button type="button" className="text-xs text-slate-400 underline" onClick={() => set({ criteria: draft.criteria.filter((_, j) => j !== i) })}>Remove</button>
          </div>
        ))}
        <button type="button" className="mt-1 text-xs text-sky-400 underline" onClick={() => set({ criteria: [...draft.criteria, { ...EMPTY_CRITERION }] })}>
          Add criterion
        </button>
      </div>

      <label className="block text-xs text-slate-400">
        Models (comma-separated model ids; empty means every model with a probe of this behaviour)
        <input className={input} aria-label="Models" value={draft.models} onChange={(e) => set({ models: e.target.value })} />
      </label>
      <label className="block text-xs text-slate-400">
        Calibration floor (fewest calibration negatives; empty uses the deployment setting)
        <input className={input} aria-label="Calibration floor" value={draft.calibration_floor} onChange={(e) => set({ calibration_floor: e.target.value })} />
      </label>
    </fieldset>
  );
}

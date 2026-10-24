/**
 * How a judge's reply is read (operator decision D2): a rating, a yes/no answer, or one label from
 * a set — each with a fixed reader and a fixed mapping to "behaviour present" / "absent".
 *
 * The server validates every form (`judge_prompt_forms.py`) and refuses an impossible one with a
 * plain reason; this editor only makes the valid shapes easy to build. It never invents a default
 * the server would not accept, so what is shown is what is saved.
 */
import type { AnswerForm, BinaryForm, LabelsForm, RatingForm } from '../../types/probeMonitor';

const INPUT =
  'mt-1 w-full rounded-sm border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200 disabled:opacity-60';

export const DEFAULT_FORMS: Record<AnswerForm['kind'], AnswerForm> = {
  rating: { kind: 'rating', min: 1, max: 10, positive_from: 6, json_key: 'rating', reply: 'json' },
  binary: {
    kind: 'binary',
    yes_token: 'yes',
    no_token: 'no',
    positive_answer: 'yes',
    json_key: 'answer',
    reply: 'json',
    case_sensitive: false,
  },
  labels: {
    kind: 'labels',
    labels: ['present', 'absent'],
    positive_labels: ['present'],
    json_key: 'label',
    reply: 'json',
    case_sensitive: false,
  },
};

const KIND_WORDS: Record<AnswerForm['kind'], string> = {
  rating: 'A rating (a whole number on a scale)',
  binary: 'Yes or no',
  labels: 'One label from a list',
};

interface Props {
  value: AnswerForm;
  onChange: (form: AnswerForm) => void;
  disabled?: boolean;
}

export function AnswerFormEditor({ value, onChange, disabled = false }: Props) {
  const set = (patch: Partial<AnswerForm>) => onChange({ ...value, ...patch } as AnswerForm);

  return (
    <fieldset className="rounded-sm border border-slate-700 p-3" data-testid="answer-form-editor" disabled={disabled}>
      <legend className="px-1 text-xs text-slate-400">HOW THE ANSWER IS READ</legend>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <label className="text-xs text-slate-400">
          ANSWER
          <select
            className={INPUT}
            value={value.kind}
            onChange={(e) => onChange(DEFAULT_FORMS[e.target.value as AnswerForm['kind']])}
            data-testid="answer-kind"
          >
            {(Object.keys(KIND_WORDS) as AnswerForm['kind'][]).map((kind) => (
              <option key={kind} value={kind}>
                {KIND_WORDS[kind]}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-400">
          REPLY SHAPE
          <select
            className={INPUT}
            value={value.reply ?? 'json'}
            onChange={(e) => set({ reply: e.target.value as AnswerForm['reply'] })}
            data-testid="answer-reply"
          >
            <option value="json">A JSON object with one key</option>
            <option value="verdict_line">A last line "VERDICT: …"</option>
          </select>
        </label>
        {(value.reply ?? 'json') === 'json' ? (
          <label className="text-xs text-slate-400">
            JSON KEY
            <input
              className={INPUT}
              value={value.json_key ?? ''}
              onChange={(e) => set({ json_key: e.target.value })}
              data-testid="answer-json-key"
            />
          </label>
        ) : null}
      </div>

      {value.kind === 'rating' ? <RatingFields form={value} set={set} /> : null}
      {value.kind === 'binary' ? <BinaryFields form={value} set={set} /> : null}
      {value.kind === 'labels' ? <LabelFields form={value} set={set} /> : null}

      <p className="mt-2 text-[11px] text-slate-500">
        A reply that cannot be read this way is not guessed: it counts toward the run's limit on
        unreadable replies, and the run stops if there are too many.
      </p>
    </fieldset>
  );
}

function RatingFields({ form, set }: { form: RatingForm; set: (p: Partial<RatingForm>) => void }) {
  const num = (v: string) => (v === '' ? 0 : Number(v));
  return (
    <div className="mt-3 grid grid-cols-3 gap-3">
      <label className="text-xs text-slate-400">
        LOWEST
        <input type="number" className={INPUT} value={form.min}
          onChange={(e) => set({ min: num(e.target.value) })} data-testid="rating-min" />
      </label>
      <label className="text-xs text-slate-400">
        HIGHEST
        <input type="number" className={INPUT} value={form.max}
          onChange={(e) => set({ max: num(e.target.value) })} data-testid="rating-max" />
      </label>
      <label className="text-xs text-slate-400">
        PRESENT FROM
        <input type="number" className={INPUT} value={form.positive_from}
          onChange={(e) => set({ positive_from: num(e.target.value) })} data-testid="rating-positive-from" />
        <span className="mt-1 block text-[11px] text-slate-500">
          This rating and above means the behaviour is present.
        </span>
      </label>
    </div>
  );
}

function BinaryFields({ form, set }: { form: BinaryForm; set: (p: Partial<BinaryForm>) => void }) {
  return (
    <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-4">
      <label className="text-xs text-slate-400">
        YES WORD
        <input className={INPUT} value={form.yes_token ?? 'yes'}
          onChange={(e) => set({ yes_token: e.target.value })} data-testid="binary-yes" />
      </label>
      <label className="text-xs text-slate-400">
        NO WORD
        <input className={INPUT} value={form.no_token ?? 'no'}
          onChange={(e) => set({ no_token: e.target.value })} data-testid="binary-no" />
      </label>
      <label className="text-xs text-slate-400">
        PRESENT WHEN THE ANSWER IS
        <select className={INPUT} value={form.positive_answer ?? 'yes'}
          onChange={(e) => set({ positive_answer: e.target.value as 'yes' | 'no' })}
          data-testid="binary-positive">
          <option value="yes">{form.yes_token || 'yes'}</option>
          <option value="no">{form.no_token || 'no'}</option>
        </select>
      </label>
      <label className="flex items-end gap-2 text-xs text-slate-400">
        <input type="checkbox" checked={Boolean(form.case_sensitive)}
          onChange={(e) => set({ case_sensitive: e.target.checked })} />
        Upper and lower case must match
      </label>
    </div>
  );
}

function LabelFields({ form, set }: { form: LabelsForm; set: (p: Partial<LabelsForm>) => void }) {
  const labels = form.labels;
  const rename = (index: number, text: string) => {
    const old = labels[index];
    const next = labels.map((l, i) => (i === index ? text : l));
    set({
      labels: next,
      positive_labels: form.positive_labels.map((p) => (p === old ? text : p)),
    });
  };
  const toggle = (label: string) =>
    set({
      positive_labels: form.positive_labels.includes(label)
        ? form.positive_labels.filter((p) => p !== label)
        : [...form.positive_labels, label],
    });
  return (
    <div className="mt-3" data-testid="label-fields">
      <p className="text-xs text-slate-400">LABELS — tick the ones that mean the behaviour is present</p>
      <ul className="mt-1 space-y-1">
        {labels.map((label, index) => (
          <li key={index} className="flex items-center gap-2">
            <input
              type="checkbox"
              aria-label={`${label || 'label'} means present`}
              checked={form.positive_labels.includes(label)}
              onChange={() => toggle(label)}
              data-testid={`label-positive-${index}`}
            />
            <input className={`${INPUT} mt-0`} value={label}
              onChange={(e) => rename(index, e.target.value)} data-testid={`label-text-${index}`} />
            <button
              type="button"
              className="text-xs text-slate-400 hover:text-slate-200 disabled:opacity-40"
              disabled={labels.length <= 2}
              onClick={() =>
                set({
                  labels: labels.filter((_, i) => i !== index),
                  positive_labels: form.positive_labels.filter((p) => p !== label),
                })
              }
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
      <div className="mt-2 flex items-center gap-4">
        <button
          type="button"
          className="rounded-sm border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
          onClick={() => set({ labels: [...labels, `label ${labels.length + 1}`] })}
          disabled={labels.length >= 20}
          data-testid="label-add"
        >
          Add a label
        </button>
        <label className="flex items-center gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={Boolean(form.case_sensitive)}
            onChange={(e) => set({ case_sensitive: e.target.checked })} />
          Upper and lower case must match
        </label>
      </div>
    </div>
  );
}

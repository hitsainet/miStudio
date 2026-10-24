/**
 * Create a behaviour, or write a new version of its definition (Feature 037 FR-1, FR-2; 040 FR-1, FR-3).
 * A version is never changed: exported probes keep the version they were exported with.
 *
 * Feature 040: the detector data is edited in `DetectorFieldsEditor`. An EDIT is sent as a change against
 * the version it was opened from (`base_version`): only the fields that differ are sent, a field emptied
 * is sent as null (removed), and the server carries every other field over. If the behaviour moved on in
 * the meantime the server refuses `stale_base_version` and the detail screen offers a reload.
 */
import { useState } from 'react';

import type {
  BehaviourDefinition,
  BehaviourDefinitionBody,
  BehaviourVersionChange,
  DetectorFields,
} from '../../types/behaviour';
import { DetectorFieldsEditor, fromDefinition, toFields } from './DetectorFieldsEditor';

type CreateBody = BehaviourDefinitionBody & { key: string };

interface Props {
  /** Set when cataloguing a key, or when editing an existing behaviour (then the key is fixed). */
  initialKey?: string;
  current?: BehaviourDefinition;
  onSubmit: (body: CreateBody | BehaviourVersionChange) => Promise<void>;
  onCancel: () => void;
}

const DETECTOR_KEYS: (keyof DetectorFields)[] = [
  'sub_questions', 'success_criteria', 'held_out_columns', 'models', 'calibration_floor',
];

/** The change against `current`: only fields whose value differs; an emptied field is null. */
export function versionChange(current: BehaviourDefinition, text: Partial<BehaviourDefinitionBody>, fields: DetectorFields): BehaviourVersionChange {
  const change: BehaviourVersionChange = { base_version: current.version };
  const out = change as unknown as Record<string, unknown>;
  const before = current as unknown as Record<string, unknown>;
  for (const [field, value] of Object.entries(text)) {
    if ((before[field] ?? null) !== (value ?? null)) out[field] = value ?? null;
  }
  for (const field of DETECTOR_KEYS) {
    const now = fields[field];
    const was = current[field];
    if (JSON.stringify(now ?? null) !== JSON.stringify(was ?? null)) out[field] = now ?? null;
  }
  return change;
}

export function BehaviourForm({ initialKey = '', current, onSubmit, onCancel }: Props) {
  const editing = current !== undefined;
  const [key, setKey] = useState(initialKey);
  const [name, setName] = useState(current?.name ?? '');
  const [description, setDescription] = useState(current?.description ?? '');
  const [positive, setPositive] = useState(current?.positive_meaning ?? '');
  const [negative, setNegative] = useState(current?.negative_meaning ?? '');
  const [fpr, setFpr] = useState(current?.target_fpr != null ? String(current.target_fpr) : '');
  const [draft, setDraft] = useState(() => fromDefinition(current));
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    const text: BehaviourDefinitionBody = {
      name: name.trim(),
      positive_meaning: positive.trim(),
      description: description.trim() || null,
      negative_meaning: negative.trim() || null,
      target_fpr: fpr.trim() ? Number(fpr) : null,
    };
    try {
      if (current) {
        await onSubmit(versionChange(current, text, toFields(draft)));
      } else {
        await onSubmit({ key: key.trim(), ...text, ...toFields(draft) });
      }
    } finally {
      setBusy(false);
    }
  }

  const input = 'w-full rounded-sm border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-100';
  return (
    <form onSubmit={submit} className="space-y-2 rounded-sm border border-slate-700 p-3" aria-label="Behaviour form">
      <label className="block text-xs text-slate-400">
        Key (lower-case words joined by hyphens, for example my-behaviour)
        <input className={input} value={key} disabled={editing || Boolean(initialKey)} onChange={(e) => setKey(e.target.value)} aria-label="Key" />
      </label>
      <label className="block text-xs text-slate-400">
        Name
        <input className={input} value={name} onChange={(e) => setName(e.target.value)} aria-label="Name" />
      </label>
      <label className="block text-xs text-slate-400">
        What counts as positive
        <textarea className={input} value={positive} onChange={(e) => setPositive(e.target.value)} aria-label="Positive meaning" />
      </label>
      <label className="block text-xs text-slate-400">
        What counts as negative (optional)
        <textarea className={input} value={negative} onChange={(e) => setNegative(e.target.value)} aria-label="Negative meaning" />
      </label>
      <label className="block text-xs text-slate-400">
        Description (optional)
        <textarea className={input} value={description} onChange={(e) => setDescription(e.target.value)} aria-label="Description" />
      </label>
      <label className="block text-xs text-slate-400">
        Target false-positive rate (FPR) for its probes, for example 0.01 (optional)
        <input className={input} value={fpr} onChange={(e) => setFpr(e.target.value)} aria-label="Target false-positive rate" />
      </label>
      <DetectorFieldsEditor draft={draft} onChange={setDraft} />
      {editing ? (
        <p className="text-xs text-amber-300">
          This makes version {current.version + 1}; exported probes keep the version they were exported with.
        </p>
      ) : null}
      <div className="flex gap-2">
        <button type="submit" disabled={busy} className="rounded-sm bg-emerald-700 px-3 py-1 text-sm text-white">
          {editing ? 'Save as new version' : 'Create behaviour'}
        </button>
        <button type="button" onClick={onCancel} className="rounded-sm border border-slate-600 px-3 py-1 text-sm text-slate-300">
          Cancel
        </button>
      </div>
    </form>
  );
}

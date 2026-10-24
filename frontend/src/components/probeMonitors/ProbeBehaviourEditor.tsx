/**
 * What a probe detects — shown on its report and editable there.
 *
 * The behaviour key is what a judge comparison is matched on: a judge run raises the probe to the
 * top evidence level only when its prompt judges the same key. Probes trained before the judge-prompt
 * library have none, and nothing guesses one; saving recomputes the evidence level at once.
 */
import { useEffect, useState } from 'react';

import { probeMonitorsApi } from '../../api/probeMonitors';

interface Props {
  probeId: string;
  behaviour: string | null | undefined;
  behaviourLabel: string | null | undefined;
  onSaved: () => void;
}

export function ProbeBehaviourEditor({ probeId, behaviour, behaviourLabel, onSaved }: Props) {
  const [editing, setEditing] = useState(false);
  const [key, setKey] = useState(behaviour ?? '');
  const [label, setLabel] = useState(behaviourLabel ?? '');
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setKey(behaviour ?? '');
    setLabel(behaviourLabel ?? '');
  }, [behaviour, behaviourLabel]);

  async function save() {
    setError(null);
    try {
      await probeMonitorsApi.setProbeBehaviour(probeId, key.trim() || null, label.trim() || null);
      setEditing(false);
      onSaved();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  return (
    <div className="mb-3 text-xs text-slate-400" data-testid="probe-behaviour">
      {editing ? (
        <div className="flex flex-wrap items-end gap-2">
          <label>
            BEHAVIOUR KEY
            <input className="mt-1 block rounded-sm border border-slate-600 bg-slate-950 px-2 py-1 text-sm text-slate-200"
              value={key} placeholder="e.g. my-behaviour" onChange={(e) => setKey(e.target.value)}
              data-testid="probe-behaviour-key" />
          </label>
          <label>
            IN WORDS
            <input className="mt-1 block rounded-sm border border-slate-600 bg-slate-950 px-2 py-1 text-sm text-slate-200"
              value={label} onChange={(e) => setLabel(e.target.value)} />
          </label>
          <button type="button" onClick={() => void save()}
            className="rounded-sm bg-emerald-700 px-2 py-1 text-white" data-testid="probe-behaviour-save">
            Save
          </button>
          <button type="button" onClick={() => setEditing(false)} className="rounded-sm border border-slate-600 px-2 py-1">
            Cancel
          </button>
        </div>
      ) : (
        <p>
          Detects:{' '}
          {behaviour ? (
            <span className="text-slate-200">
              {behaviourLabel ?? behaviour} <span className="font-mono text-slate-500">({behaviour})</span>
            </span>
          ) : (
            <span className="text-amber-300">not recorded — no judge comparison can raise its evidence level</span>
          )}{' '}
          <button type="button" className="underline" onClick={() => setEditing(true)} data-testid="probe-behaviour-edit">
            {behaviour ? 'change' : 'set it'}
          </button>
        </p>
      )}
      {error ? <p className="mt-1 text-amber-300">{error}</p> : null}
    </div>
  );
}

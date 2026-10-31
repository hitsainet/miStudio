/**
 * The "Behaviours" sub-tab (Feature 037 FR-16; D20–D26): the catalogue of what probes detect and judge
 * prompts judge. A key already used by a prompt, run, probe or view without a catalogue record is listed
 * under "Uncatalogued keys" — valid, not an error (D25) — with "Catalogue this".
 */
import { useCallback, useEffect, useState } from 'react';

import { behavioursApi } from '../../api/behaviours';
import type { BehaviourDefinitionBody, BehaviourList } from '../../types/behaviour';
import { HelpLink } from '../help/HelpLink';
import { BehaviourDetail } from './BehaviourDetail';
import { BehaviourForm } from './BehaviourForm';

export function BehavioursTab({ onOpenTab }: { onOpenTab?: (tab: string) => void }) {
  const [list, setList] = useState<BehaviourList>({ behaviours: [], uncatalogued: [] });
  const [showArchived, setShowArchived] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);
  const [creating, setCreating] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      setList(await behavioursApi.list(showArchived));
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [showArchived]);

  /** Feature 040 FR-6: the catalogue is paged; the next page is appended in key order. */
  const loadMore = async () => {
    if (!list.next_cursor) return;
    try {
      const page = await behavioursApi.list(showArchived, { cursor: list.next_cursor });
      setList({
        ...page,
        behaviours: [...list.behaviours, ...page.behaviours],
        uncatalogued: [...list.uncatalogued, ...page.uncatalogued],
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  };

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section data-testid="tab-behaviours" className="grid grid-cols-1 gap-4 lg:grid-cols-3">
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <button type="button" className="rounded-control bg-emerald-700 px-3 py-1 text-sm text-white" onClick={() => { setCreating(''); setSelected(null); }}>
            New behaviour
          </button>
          <label className="text-xs text-slate-400">
            <input type="checkbox" checked={showArchived} onChange={(e) => setShowArchived(e.target.checked)} /> Show archived
          </label>
          <HelpLink topic="behaviours" label="Help with behaviours" />
        </div>
        {error ? <p role="alert" className="text-sm text-red-400">{error}</p> : null}
        <ul aria-label="Behaviours" className="space-y-1">
          {list.behaviours.map((b) => (
            <li key={b.key}>
              <button type="button" onClick={() => { setSelected(b.key); setCreating(null); }} className="w-full rounded-control border border-slate-700 px-2 py-1 text-left text-sm text-slate-200">
                {b.definition.name} <span className="text-xs text-slate-400">({b.key}) {b.status ?? ''}{b.archived ? ' archived' : ''}</span>
                <span className="block text-xs text-slate-500">{b.counts.probes} probes, {b.counts.prompts} prompts, {b.counts.views} views</span>
              </button>
            </li>
          ))}
        </ul>
        {list.next_cursor ? (
          <button type="button" className="text-xs text-sky-400 underline" onClick={() => void loadMore()}>
            Show more{list.total != null ? ` (${list.behaviours.length} of ${list.total})` : ''}
          </button>
        ) : null}
        {list.uncatalogued.length > 0 ? (
          <div>
            <h4 className="text-xs uppercase text-slate-400">Uncatalogued keys</h4>
            <ul aria-label="Uncatalogued keys" className="space-y-1">
              {list.uncatalogued.map((u) => (
                <li key={u.key} className="flex items-center gap-2 text-sm text-slate-300">
                  <span className="rounded-full bg-slate-700 px-1 text-xs">uncatalogued</span>
                  <span>{u.key}</span>
                  <span className="text-xs text-slate-500">{u.prompts} prompts, {u.runs} runs, {u.probes} probes, {u.views} views</span>
                  <button type="button" className="text-xs text-sky-400 underline" onClick={() => { setCreating(u.key); setSelected(null); }}>
                    Catalogue this
                  </button>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
      <div className="lg:col-span-2">
        {creating !== null ? (
          <BehaviourForm
            initialKey={creating}
            onCancel={() => setCreating(null)}
            onSubmit={async (body) => {
              try {
                const made = await behavioursApi.create(body as BehaviourDefinitionBody & { key: string });
                setCreating(null);
                await load();
                setSelected(made.key);
              } catch (e) {
                setError(e instanceof Error ? e.message : String(e));
              }
            }}
          />
        ) : selected ? (
          <BehaviourDetail behaviourKey={selected} onChanged={() => void load()} onOpenTab={onOpenTab} />
        ) : (
          <p className="text-sm text-slate-400">
            A behaviour is what a probe detects and what a judge prompt judges, named by a short key. Choose one, or catalogue a key already in use.
          </p>
        )}
      </div>
    </section>
  );
}

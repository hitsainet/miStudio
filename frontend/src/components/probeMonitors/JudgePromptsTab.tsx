/**
 * The "Judge prompts" sub-tab: the library of instructions a judge model is given when a probe is
 * compared with it (operator decisions D1/D2, 2026-10-21).
 *
 * One row per prompt NAME, showing its newest version; selecting it opens the editor with the
 * version history. Every number here is the server's — "used by N runs" is counted by the server,
 * never inferred from the list of runs this screen happens to hold.
 */
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Archive, ArchiveRestore, Download, Lock, Plus, Trash2 } from 'lucide-react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import { HelpLink } from '../help/HelpLink';
import type { JudgePrompt, ProbeDataset } from '../../types/probeMonitor';
import { JudgePromptEditor } from './JudgePromptEditor';
import { JudgePromptImport } from './JudgePromptImport';

interface Props {
  datasets: ProbeDataset[];
}

export function JudgePromptsTab({ datasets }: Props) {
  const [prompts, setPrompts] = useState<JudgePrompt[]>([]);
  const [placeholders, setPlaceholders] = useState<Record<string, string>>({});
  const [showArchived, setShowArchived] = useState(false);
  const [behaviour, setBehaviour] = useState('');
  const [selected, setSelected] = useState<JudgePrompt | null>(null);
  const [creating, setCreating] = useState(false);
  const [importing, setImporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const list = await probeMonitorsApi.listJudgePrompts({ includeArchived: showArchived });
      setPrompts(list.prompts);
      setPlaceholders(list.placeholders);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, [showArchived]);

  useEffect(() => {
    void load();
  }, [load]);

  const open = useCallback(async (id: string) => {
    try {
      setSelected(await probeMonitorsApi.getJudgePrompt(id));
      setCreating(false);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  /** The newest version of each name, the list's one row per prompt. */
  const latest = useMemo(() => {
    const byName = new Map<string, JudgePrompt>();
    for (const prompt of prompts) {
      const seen = byName.get(prompt.name);
      if (!seen || prompt.version > seen.version) byName.set(prompt.name, prompt);
    }
    return [...byName.values()].filter((p) => !behaviour || p.behaviour === behaviour);
  }, [prompts, behaviour]);
  const behaviours = useMemo(() => [...new Set(prompts.map((p) => p.behaviour))].sort(), [prompts]);

  async function act(action: () => Promise<unknown>, reopen?: string) {
    try {
      await action();
      await load();
      if (reopen) await open(reopen);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function exportPrompt(prompt: JudgePrompt) {
    const document = await probeMonitorsApi.exportJudgePrompt(prompt.id);
    const blob = new Blob([JSON.stringify(document, null, 2)], { type: 'application/json' });
    const link = window.document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `${prompt.name}-v${prompt.version}.judge-prompt.json`;
    link.click();
    URL.revokeObjectURL(link.href);
  }

  const saved = (prompt: JudgePrompt) => {
    void load();
    void open(prompt.id);
    setImporting(false);
  };

  return (
    <section data-testid="tab-judge-prompts">
      <p className="mb-3 text-sm text-slate-400">
        A judge prompt is the instructions an AI judge model follows when it labels the same rows a
        probe was tested on. Each prompt judges one <strong>behaviour</strong>; a probe's
        comparison with a judge counts toward its top evidence level only when both name the same
        behaviour. A version that a judge run has used is locked, so every run keeps the exact text
        it was measured with.
      </p>
      {error ? (
        <p className="mb-3 rounded-control border border-amber-700 bg-amber-950/40 p-2 text-sm text-amber-200" data-testid="judge-prompts-error">
          {error}
        </p>
      ) : null}

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <button type="button" onClick={() => { setCreating(true); setSelected(null); setImporting(false); }}
          className="flex items-center gap-1 rounded-control bg-emerald-700 px-3 py-1.5 text-sm text-white"
          data-testid="judge-prompt-new">
          <Plus className="h-4 w-4" aria-hidden="true" /> New prompt
        </button>
        <button type="button" onClick={() => setImporting((v) => !v)}
          className="rounded-control border border-slate-600 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800"
          data-testid="judge-prompt-import-toggle">
          Import…
        </button>
        <select className="rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200"
          value={behaviour} onChange={(e) => setBehaviour(e.target.value)} aria-label="Filter by behaviour"
          data-testid="judge-prompts-behaviour-filter">
          <option value="">every behaviour</option>
          {behaviours.map((b) => <option key={b} value={b}>{b}</option>)}
        </select>
        <label className="flex items-center gap-1 text-xs text-slate-400">
          <input type="checkbox" checked={showArchived} onChange={(e) => setShowArchived(e.target.checked)} />
          Show archived
        </label>
        <span className="ml-auto">
          <HelpLink topic="judge-prompts" label="Help with judge prompts" />
        </span>
      </div>

      {importing ? (
        <div className="mb-3">
          <JudgePromptImport placeholders={placeholders} onImported={saved} />
        </div>
      ) : null}

      <div className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <ul className="space-y-2" data-testid="judge-prompts-list">
          {latest.length === 0 ? (
            <li className="text-sm text-slate-400">No judge prompts{behaviour ? ` for ${behaviour}` : ''}.</li>
          ) : null}
          {latest.map((prompt) => (
            <li key={prompt.id}>
              <button
                type="button"
                onClick={() => void open(prompt.id)}
                className={`w-full rounded-control border p-2 text-left text-sm ${
                  selected?.name === prompt.name ? 'border-emerald-600 bg-slate-900' : 'border-slate-700 bg-slate-900/60 hover:bg-slate-900'
                }`}
                data-testid="judge-prompt-row"
              >
                <span className="block text-slate-100">{prompt.title}</span>
                <span className="mt-0.5 block text-xs text-slate-400">
                  {prompt.behaviour_label ?? prompt.behaviour} · {prompt.answer_form_summary}
                </span>
                <span className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-500">
                  <span className="font-mono">{prompt.key}</span>
                  {prompt.locked ? (
                    <span className="flex items-center gap-1 text-amber-300">
                      <Lock className="h-3 w-3" aria-hidden="true" /> used by {prompt.used_by_runs}
                    </span>
                  ) : null}
                  {prompt.archived ? <span>archived</span> : null}
                </span>
              </button>
            </li>
          ))}
        </ul>

        <div className="lg:col-span-2">
          {creating ? (
            <JudgePromptEditor prompt={null} placeholders={placeholders} datasets={datasets} onSaved={saved} />
          ) : selected ? (
            <>
              <JudgePromptEditor prompt={selected} placeholders={placeholders} datasets={datasets} onSaved={saved} />
              <div className="mt-4 rounded-control border border-slate-700 p-3" data-testid="judge-prompt-versions">
                <p className="mb-2 text-xs text-slate-400">VERSIONS</p>
                <ul className="space-y-1 text-xs">
                  {(selected.versions ?? []).map((v) => (
                    <li key={v.id} className="flex flex-wrap items-center gap-2">
                      <button type="button" onClick={() => void open(v.id)}
                        className={`font-mono ${v.id === selected.id ? 'text-emerald-300' : 'text-slate-300 hover:underline'}`}>
                        {v.key}
                      </button>
                      <span className="text-slate-500">{new Date(v.created_at).toLocaleString()} by {v.created_by}</span>
                      <span className="text-slate-500">
                        {v.locked ? `used by ${v.used_by_runs} ${v.used_by_runs === 1 ? 'run' : 'runs'}` : 'not used'}
                      </span>
                      {v.archived ? <span className="text-slate-500">archived</span> : null}
                    </li>
                  ))}
                </ul>
                <div className="mt-3 flex flex-wrap gap-2">
                  <button type="button" onClick={() => void act(() => probeMonitorsApi.archiveJudgePrompt(selected.id, !selected.archived), selected.id)}
                    className="flex items-center gap-1 rounded-control border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                    data-testid="judge-prompt-archive">
                    {selected.archived ? <ArchiveRestore className="h-3 w-3" aria-hidden="true" /> : <Archive className="h-3 w-3" aria-hidden="true" />}
                    {selected.archived ? 'Unarchive this version' : 'Archive this version'}
                  </button>
                  <button type="button" onClick={() => void exportPrompt(selected).catch((e: unknown) => setError(String(e)))}
                    className="flex items-center gap-1 rounded-control border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                    data-testid="judge-prompt-export">
                    <Download className="h-3 w-3" aria-hidden="true" /> Export this version
                  </button>
                  {!selected.locked ? (
                    <button type="button"
                      onClick={() => void act(async () => { await probeMonitorsApi.deleteJudgePrompt(selected.id); setSelected(null); })}
                      className="flex items-center gap-1 rounded-control border border-red-800 px-2 py-1 text-xs text-red-300 hover:bg-red-950"
                      data-testid="judge-prompt-delete">
                      <Trash2 className="h-3 w-3" aria-hidden="true" /> Delete this unused version
                    </button>
                  ) : null}
                </div>
              </div>
            </>
          ) : (
            <p className="text-sm text-slate-400">Choose a prompt to read or edit it, or start a new one.</p>
          )}
        </div>
      </div>
    </section>
  );
}

/**
 * Edit one judge prompt — or start a new one.
 *
 * ⚠ A USED VERSION IS LOCKED, AND THE EDITOR SAYS SO BEFORE YOU TYPE. Once a judge run has used a
 * version its instructions, template, answer form and behaviour are the instrument that run was
 * measured with, so the server refuses to change them (409). Here the content fields are read-only
 * on a locked version until "Start a new version" is pressed; saving then makes version N+1 and the
 * old runs keep pointing at the old one. The title and description can always change.
 */
import { useEffect, useMemo, useState } from 'react';
import { Copy, GitBranch, Lock, Save } from 'lucide-react';

import { probeMonitorsApi, type JudgePromptFields } from '../../api/probeMonitors';
import type { AnswerForm, JudgePrompt, JudgePromptContent, ProbeDataset } from '../../types/probeMonitor';
import { AnswerFormEditor, DEFAULT_FORMS } from './AnswerFormEditor';
import { JudgePromptCheck } from './JudgePromptCheck';

const INPUT =
  'mt-1 w-full rounded-sm border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200 disabled:opacity-60';

export const NEW_PROMPT_CONTENT: JudgePromptContent = {
  system_prompt:
    'You decide whether … (say what counts, and what does not).\n\n' +
    'Reply with strict JSON and nothing else: {"answer": "yes" or "no", "reason": "<one short sentence>"}',
  user_template: 'Text:\n{text}',
  answer_form: DEFAULT_FORMS.binary,
  max_interaction_chars: 8000,
};

interface Draft extends JudgePromptContent {
  name: string;
  title: string;
  description: string;
  behaviour: string;
  behaviour_label: string;
}

function draftOf(prompt: JudgePrompt | null): Draft {
  if (!prompt) {
    return {
      ...NEW_PROMPT_CONTENT,
      name: '',
      title: '',
      description: '',
      behaviour: '',
      behaviour_label: '',
    };
  }
  return {
    name: prompt.name,
    title: prompt.title,
    description: prompt.description ?? '',
    behaviour: prompt.behaviour,
    behaviour_label: prompt.behaviour_label ?? '',
    system_prompt: prompt.system_prompt,
    user_template: prompt.user_template,
    answer_form: prompt.answer_form,
    max_interaction_chars: prompt.max_interaction_chars,
  };
}

interface Props {
  /** null = a brand-new prompt. */
  prompt: JudgePrompt | null;
  placeholders: Record<string, string>;
  datasets: ProbeDataset[];
  /** Called with the saved row so the list can select it. */
  onSaved: (saved: JudgePrompt) => void;
}

export function JudgePromptEditor({ prompt, placeholders, datasets, onSaved }: Props) {
  const [draft, setDraft] = useState<Draft>(() => draftOf(prompt));
  const [newVersion, setNewVersion] = useState(false);
  const [cloning, setCloning] = useState(false);
  const [cloneName, setCloneName] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setDraft(draftOf(prompt));
    setNewVersion(false);
    setCloning(false);
    setError(null);
  }, [prompt]);

  const creating = prompt === null;
  const contentLocked = !creating && prompt.locked && !newVersion;
  const nextVersion = prompt?.versions?.length
    ? Math.max(...prompt.versions.map((v) => v.version)) + 1
    : (prompt?.version ?? 0) + 1;

  const content: JudgePromptContent = useMemo(
    () => ({
      system_prompt: draft.system_prompt,
      user_template: draft.user_template,
      answer_form: draft.answer_form,
      max_interaction_chars: draft.max_interaction_chars,
    }),
    [draft.system_prompt, draft.user_template, draft.answer_form, draft.max_interaction_chars]
  );

  const update = (patch: Partial<Draft>) => setDraft((d) => ({ ...d, ...patch }));

  async function save(action: 'create' | 'update' | 'version' | 'clone') {
    setSaving(true);
    setError(null);
    try {
      const display: JudgePromptFields = {
        title: draft.title,
        description: draft.description || null,
        behaviour_label: draft.behaviour_label || null,
      };
      let saved: JudgePrompt;
      if (action === 'create') {
        saved = await probeMonitorsApi.createJudgePrompt({
          ...display,
          ...content,
          name: draft.name,
          behaviour: draft.behaviour,
        });
      } else if (action === 'update') {
        saved = await probeMonitorsApi.updateJudgePrompt(
          prompt!.id,
          prompt!.locked ? display : { ...display, ...content, behaviour: draft.behaviour }
        );
      } else if (action === 'version') {
        saved = await probeMonitorsApi.newJudgePromptVersion(prompt!.id, { ...display, ...content });
      } else {
        saved = await probeMonitorsApi.cloneJudgePrompt(prompt!.id, {
          ...display,
          ...content,
          name: cloneName,
          behaviour: draft.behaviour,
          title: draft.title === prompt!.title ? `${draft.title} (copy)` : draft.title,
        });
      }
      onSaved(saved);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setSaving(false);
    }
  }

  return (
    <div data-testid="judge-prompt-editor">
      {!creating ? (
        <div className="mb-3 flex flex-wrap items-center gap-2 text-xs">
          <span className="rounded-sm bg-slate-800 px-2 py-0.5 font-mono text-slate-300">{prompt.key}</span>
          {prompt.locked ? (
            <span className="flex items-center gap-1 rounded-sm bg-amber-900/50 px-2 py-0.5 text-amber-200" data-testid="judge-prompt-lock">
              <Lock className="h-3 w-3" aria-hidden="true" />
              Used by {prompt.used_by_runs} {prompt.used_by_runs === 1 ? 'run' : 'runs'} — locked
            </span>
          ) : (
            <span className="rounded-sm bg-slate-800 px-2 py-0.5 text-slate-400">Not used yet — editable</span>
          )}
          {prompt.archived ? <span className="rounded-sm bg-slate-800 px-2 py-0.5 text-slate-400">Archived</span> : null}
          <span className="text-slate-500" title={prompt.content_hash}>
            fingerprint {prompt.content_hash.slice(0, 12)}…
          </span>
        </div>
      ) : null}

      {contentLocked ? (
        <p className="mb-3 rounded-sm border border-amber-800 bg-amber-950/30 p-2 text-xs text-amber-200">
          Judge runs were measured with this exact text, so it cannot change. Start a new version to
          change it; those runs keep pointing at this one.
        </p>
      ) : null}

      <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
        <label className="text-xs text-slate-400">
          TITLE
          <input className={INPUT} value={draft.title} onChange={(e) => update({ title: e.target.value })}
            data-testid="judge-prompt-title" />
        </label>
        <label className="text-xs text-slate-400">
          NAME
          <input className={INPUT} value={draft.name} disabled={!creating}
            placeholder="lower-case-words, e.g. refusal-presence"
            onChange={(e) => update({ name: e.target.value })} data-testid="judge-prompt-name" />
        </label>
        <label className="text-xs text-slate-400">
          BEHAVIOUR KEY
          <input className={INPUT} value={draft.behaviour} disabled={!creating && (prompt.locked || newVersion)}
            placeholder="e.g. my-behaviour"
            onChange={(e) => update({ behaviour: e.target.value })} data-testid="judge-prompt-behaviour" />
          <span className="mt-1 block text-[11px] text-slate-500">
            A probe's comparison with this judge counts toward its top evidence level only when
            the probe's behaviour key is the same.
          </span>
        </label>
        <label className="text-xs text-slate-400">
          BEHAVIOUR IN WORDS
          <input className={INPUT} value={draft.behaviour_label}
            placeholder="e.g. Personal data present"
            onChange={(e) => update({ behaviour_label: e.target.value })} />
        </label>
        <label className="text-xs text-slate-400 sm:col-span-2">
          DESCRIPTION
          <input className={INPUT} value={draft.description} onChange={(e) => update({ description: e.target.value })} />
        </label>
        <label className="text-xs text-slate-400 sm:col-span-2">
          INSTRUCTIONS TO THE JUDGE
          <textarea className={`${INPUT} h-40 font-mono`} value={draft.system_prompt} disabled={contentLocked}
            onChange={(e) => update({ system_prompt: e.target.value })} data-testid="judge-prompt-system" />
        </label>
        <label className="text-xs text-slate-400 sm:col-span-2">
          WHAT THE JUDGE IS SHOWN FOR EACH ROW
          <textarea className={`${INPUT} h-20 font-mono`} value={draft.user_template} disabled={contentLocked}
            onChange={(e) => update({ user_template: e.target.value })} data-testid="judge-prompt-template" />
          <span className="mt-1 flex flex-wrap gap-1">
            {Object.entries(placeholders).map(([name, words]) => (
              <button
                key={name}
                type="button"
                title={words}
                disabled={contentLocked}
                onClick={() => update({ user_template: `${draft.user_template}{${name}}` })}
                className="rounded-sm border border-slate-700 px-1.5 py-0.5 font-mono text-[11px] text-slate-300 hover:bg-slate-800 disabled:opacity-50"
              >
                {`{${name}}`}
              </button>
            ))}
          </span>
        </label>
        <label className="text-xs text-slate-400">
          LONGEST ROW SHOWN (CHARACTERS)
          <input type="number" min={200} max={100000} className={INPUT} value={draft.max_interaction_chars}
            disabled={contentLocked}
            onChange={(e) => update({ max_interaction_chars: Number(e.target.value) || 0 })} />
          <span className="mt-1 block text-[11px] text-slate-500">Longer rows are cut from the front, keeping the end.</span>
        </label>
      </div>

      <div className="mt-3">
        <AnswerFormEditor
          value={draft.answer_form}
          disabled={contentLocked}
          onChange={(answer_form: AnswerForm) => update({ answer_form })}
        />
      </div>

      {error ? (
        <p className="mt-3 text-sm text-amber-300" data-testid="judge-prompt-error">
          {error}
        </p>
      ) : null}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {creating ? (
          <button type="button" onClick={() => void save('create')} disabled={saving}
            className="flex items-center gap-1 rounded-sm bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:bg-slate-700"
            data-testid="judge-prompt-create">
            <Save className="h-4 w-4" aria-hidden="true" /> Save new prompt
          </button>
        ) : (
          <>
            {!newVersion ? (
              <button type="button" onClick={() => void save('update')} disabled={saving}
                className="flex items-center gap-1 rounded-sm bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:bg-slate-700"
                data-testid="judge-prompt-save">
                <Save className="h-4 w-4" aria-hidden="true" />
                {prompt.locked ? 'Save title and description' : 'Save changes'}
              </button>
            ) : (
              <button type="button" onClick={() => void save('version')} disabled={saving}
                className="flex items-center gap-1 rounded-sm bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:bg-slate-700"
                data-testid="judge-prompt-save-version">
                <GitBranch className="h-4 w-4" aria-hidden="true" /> Save as version {nextVersion}
              </button>
            )}
            {!newVersion ? (
              <button type="button" onClick={() => setNewVersion(true)}
                className="flex items-center gap-1 rounded-sm border border-slate-600 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800"
                data-testid="judge-prompt-start-version">
                <GitBranch className="h-4 w-4" aria-hidden="true" /> Start a new version
              </button>
            ) : (
              <button type="button" onClick={() => { setNewVersion(false); setDraft(draftOf(prompt)); }}
                className="rounded-sm border border-slate-600 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800">
                Cancel new version
              </button>
            )}
            <button type="button" onClick={() => setCloning((c) => !c)}
              className="flex items-center gap-1 rounded-sm border border-slate-600 px-3 py-1.5 text-sm text-slate-200 hover:bg-slate-800"
              data-testid="judge-prompt-clone-toggle">
              <Copy className="h-4 w-4" aria-hidden="true" /> Copy as a new prompt
            </button>
          </>
        )}
      </div>
      {cloning && !creating ? (
        <div className="mt-2 flex flex-wrap items-end gap-2 rounded-sm border border-slate-700 p-2">
          <label className="text-xs text-slate-400">
            NEW NAME
            <input className={INPUT} value={cloneName} onChange={(e) => setCloneName(e.target.value)}
              data-testid="judge-prompt-clone-name" />
          </label>
          <p className="text-[11px] text-slate-500">
            Starts at version 1 with the text above. Change the behaviour key above if it judges something else.
          </p>
          <button type="button" onClick={() => void save('clone')} disabled={saving || !cloneName.trim()}
            className="rounded-sm bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:bg-slate-700"
            data-testid="judge-prompt-clone">
            Copy
          </button>
        </div>
      ) : null}

      <JudgePromptCheck
        datasets={datasets}
        source={
          !creating && !newVersion && prompt.locked
            ? { prompt_id: prompt.id }
            : { draft: content }
        }
      />
    </div>
  );
}

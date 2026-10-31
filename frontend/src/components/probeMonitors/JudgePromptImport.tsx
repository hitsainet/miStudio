/**
 * Bring a judge prompt in: a file exported from miStudio, or a rubric from miDataworks.
 *
 * STANDALONE FIRST. miDataworks is optional: when it is not connected the rubric list says so and
 * an exported rubric FILE can still be imported. Only binary and pointwise rubrics map onto a judge
 * prompt (one row, one answer); the server refuses the rest with the reason, and this form shows it.
 */
import { useState } from 'react';
import { Upload } from 'lucide-react';

import { probeMonitorsApi } from '../../api/probeMonitors';
import type { JudgePrompt, MiDataworksRubric } from '../../types/probeMonitor';

const INPUT =
  'mt-1 w-full rounded-control border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200';

interface Props {
  placeholders: Record<string, string>;
  onImported: (prompt: JudgePrompt) => void;
}

/** FileReader rather than `file.text()`, which older browsers (and jsdom) lack. */
function readFile(file: File): Promise<Record<string, unknown>> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        resolve(JSON.parse(String(reader.result)) as Record<string, unknown>);
      } catch (error) {
        reject(new Error(`${file.name} is not a JSON document: ${String(error)}`));
      }
    };
    reader.onerror = () => reject(reader.error ?? new Error(`could not read ${file.name}`));
    reader.readAsText(file);
  });
}

export function JudgePromptImport({ placeholders, onImported }: Props) {
  const [error, setError] = useState<string | null>(null);
  const [rubrics, setRubrics] = useState<MiDataworksRubric[] | null>(null);
  const [rubricId, setRubricId] = useState('');
  const [rubricFile, setRubricFile] = useState<Record<string, unknown> | null>(null);
  const [name, setName] = useState('');
  const [title, setTitle] = useState('');
  const [behaviour, setBehaviour] = useState('');
  const [fieldMap, setFieldMap] = useState('');
  const [literals, setLiterals] = useState('');
  const [positive, setPositive] = useState('');
  const [busy, setBusy] = useState(false);

  async function importFile(file: File | undefined) {
    if (!file) return;
    setError(null);
    try {
      const document = await readFile(file);
      if (document.format === 'midataworks.rubric/v1') {
        setRubricFile(document);
        setRubricId('');
        return;
      }
      onImported(await probeMonitorsApi.importJudgePrompt(document));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function loadRubrics() {
    setError(null);
    try {
      setRubrics((await probeMonitorsApi.listMiDataworksRubrics()).rubrics);
    } catch (e) {
      setRubrics(null);
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  /** "a=text, b=interaction" → {a: "text", b: "interaction"}. */
  function pairs(text: string): Record<string, string> {
    return Object.fromEntries(
      text
        .split(/[\n,]/)
        .map((part) => part.split('='))
        .filter((kv) => kv.length >= 2 && kv[0].trim())
        .map(([k, ...v]) => [k.trim(), v.join('=').trim()])
    );
  }

  async function importRubric() {
    setBusy(true);
    setError(null);
    const numeric = /^\d+$/.test(positive.trim());
    try {
      onImported(
        await probeMonitorsApi.importRubric({
          ...(rubricFile ? { document: rubricFile } : { rubric_id: rubricId }),
          name: name.trim(),
          title: title.trim(),
          behaviour: behaviour.trim(),
          field_map: pairs(fieldMap),
          literals: pairs(literals),
          ...(numeric
            ? { positive_from: Number(positive.trim()) }
            : { positive_verdicts: positive.split(',').map((p) => p.trim()).filter(Boolean) }),
        })
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const chosen = rubrics?.find((r) => r.id === rubricId);
  const haveRubric = Boolean(rubricFile || rubricId);

  return (
    <section className="rounded-card border border-slate-700 bg-slate-900/40 p-3" data-testid="judge-prompt-import">
      <p className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
        <Upload className="h-4 w-4 text-emerald-400" aria-hidden="true" /> Import
      </p>
      <label className="text-xs text-slate-400">
        A FILE (a judge prompt exported from miStudio, or a rubric exported from miDataworks)
        <input type="file" accept="application/json,.json" className="mt-1 block text-xs text-slate-300"
          onChange={(e) => void importFile(e.target.files?.[0])} data-testid="judge-prompt-import-file" />
      </label>

      <div className="mt-3">
        <button type="button" onClick={() => void loadRubrics()}
          className="rounded-control border border-slate-600 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
          data-testid="judge-prompt-load-rubrics">
          List miDataworks rubrics
        </button>
        {rubrics ? (
          <select className={INPUT} value={rubricId}
            onChange={(e) => { setRubricId(e.target.value); setRubricFile(null); }}
            data-testid="judge-prompt-rubric">
            <option value="">choose a rubric…</option>
            {rubrics.map((r) => (
              <option key={r.id} value={r.id} disabled={!r.importable}>
                {r.name} v{r.version} ({r.style}{r.importable ? '' : ' — cannot be imported'})
              </option>
            ))}
          </select>
        ) : null}
      </div>

      {haveRubric ? (
        <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3" data-testid="judge-prompt-rubric-form">
          <p className="text-xs text-slate-400 sm:col-span-3">
            {rubricFile ? `Rubric file ${String(rubricFile.name)} v${String(rubricFile.version)}` : `Rubric ${chosen?.name ?? rubricId}`}
            {chosen?.input_fields ? ` — fields: ${chosen.input_fields.join(', ')}` : ''}
            {chosen?.allowed_verdicts ? ` — verdicts: ${chosen.allowed_verdicts.join(', ')}` : ''}
          </p>
          <label className="text-xs text-slate-400">NEW NAME
            <input className={INPUT} value={name} onChange={(e) => setName(e.target.value)} /></label>
          <label className="text-xs text-slate-400">TITLE
            <input className={INPUT} value={title} onChange={(e) => setTitle(e.target.value)} /></label>
          <label className="text-xs text-slate-400">BEHAVIOUR KEY
            <input className={INPUT} value={behaviour} onChange={(e) => setBehaviour(e.target.value)} /></label>
          <label className="text-xs text-slate-400 sm:col-span-3">
            ROW FIELDS (rubric field = one of {Object.keys(placeholders).join(', ')})
            <input className={INPUT} placeholder="passage=text" value={fieldMap} onChange={(e) => setFieldMap(e.target.value)} />
          </label>
          <label className="text-xs text-slate-400 sm:col-span-3">
            FIXED TEXT (rubric field = the same text for every row)
            <input className={INPUT} placeholder="question=Does the text contain personal data?" value={literals}
              onChange={(e) => setLiterals(e.target.value)} />
          </label>
          <label className="text-xs text-slate-400 sm:col-span-2">
            PRESENT WHEN (verdicts separated by commas, or for a rating the lowest rating that counts)
            <input className={INPUT} value={positive} onChange={(e) => setPositive(e.target.value)} />
          </label>
          <div className="flex items-end">
            <button type="button" onClick={() => void importRubric()}
              disabled={busy || !name.trim() || !title.trim() || !behaviour.trim() || !positive.trim()}
              className="rounded-control bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:bg-slate-700"
              data-testid="judge-prompt-import-rubric">
              Import rubric
            </button>
          </div>
        </div>
      ) : null}

      {error ? <p className="mt-2 text-xs text-amber-300" data-testid="judge-prompt-import-error">{error}</p> : null}
    </section>
  );
}

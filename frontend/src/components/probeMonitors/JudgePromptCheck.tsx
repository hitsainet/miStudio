/**
 * Check a judge prompt before trusting it: a LIVE PREVIEW of the judge's exact messages on a few
 * rows of a chosen view (no model is called), and a TRY that sends a handful of rows to the judge
 * and shows each reply and how it was read.
 *
 * The preview follows the editor as you type (debounced), so the text being edited is the text
 * being shown. Neither records anything nor locks the prompt; only a judge run does that.
 */
import { useEffect, useState } from 'react';
import { FlaskConical, Loader2 } from 'lucide-react';

import { probeMonitorsApi, type PromptOrDraft } from '../../api/probeMonitors';
import { usePersistentState } from '../../hooks/usePersistentState';
import { isArchived } from './datasetArchive';
import type {
  JudgePromptPreviewResult,
  JudgePromptTryResult,
  ProbeDataset,
} from '../../types/probeMonitor';

const INPUT =
  'mt-1 w-full rounded-sm border border-slate-600 bg-slate-950 px-2 py-1.5 text-sm text-slate-200';

interface Props {
  /** A saved version or the editor's unsaved draft. */
  source: PromptOrDraft | null;
  datasets: ProbeDataset[];
}

export function JudgePromptCheck({ source, datasets }: Props) {
  // Archived views are out of use for new work, so they are not offered here either.
  const views = datasets.filter((d) => d.role !== 'calibration' && !isArchived(d));
  const [datasetId, setDatasetId] = usePersistentState('judgePrompt.previewView', '');
  const [rows, setRows] = useState(4);
  const [preview, setPreview] = useState<JudgePromptPreviewResult | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [previewing, setPreviewing] = useState(false);

  const [endpoint, setEndpoint] = usePersistentState('judge.endpoint', '');
  const [model, setModel] = usePersistentState('judge.model', '');
  const [trying, setTrying] = useState(false);
  const [tried, setTried] = useState<JudgePromptTryResult | null>(null);
  const [tryError, setTryError] = useState<string | null>(null);

  const key = JSON.stringify(source);
  useEffect(() => {
    if (!source || !datasetId) return undefined;
    const timer = setTimeout(() => {
      setPreviewing(true);
      probeMonitorsApi
        .previewJudgePrompt({ ...source, dataset_id: datasetId, rows })
        .then((result) => {
          setPreview(result);
          setPreviewError(null);
        })
        .catch((error: unknown) => {
          setPreview(null);
          setPreviewError(error instanceof Error ? error.message : String(error));
        })
        .finally(() => setPreviewing(false));
    }, 500);
    return () => clearTimeout(timer);
    // `key` stands for `source`: the object is rebuilt on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, datasetId, rows]);

  async function runTry() {
    if (!source || !datasetId || !endpoint.trim() || !model.trim()) return;
    setTrying(true);
    setTryError(null);
    try {
      setTried(
        await probeMonitorsApi.tryJudgePrompt({
          ...source,
          dataset_id: datasetId,
          endpoint: endpoint.trim(),
          model: model.trim(),
          rows: Math.min(rows, 10),
        })
      );
    } catch (error) {
      setTried(null);
      setTryError(error instanceof Error ? error.message : String(error));
    } finally {
      setTrying(false);
    }
  }

  return (
    <section className="mt-4 rounded-sm border border-slate-700 bg-slate-900/40 p-3" data-testid="judge-prompt-check">
      <p className="mb-2 flex items-center gap-2 text-sm font-medium text-slate-200">
        <FlaskConical className="h-4 w-4 text-emerald-400" aria-hidden="true" />
        Check it on real rows
      </p>
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <label className="text-xs text-slate-400 sm:col-span-2">
          ROWS FROM
          <select className={INPUT} value={datasetId} onChange={(e) => setDatasetId(e.target.value)}
            data-testid="judge-prompt-preview-view">
            <option value="">choose a probe dataset…</option>
            {views.map((view) => (
              <option key={view.id} value={view.id}>
                {view.name}
              </option>
            ))}
          </select>
        </label>
        <label className="text-xs text-slate-400">
          HOW MANY ROWS
          <input type="number" min={1} max={20} className={INPUT} value={rows}
            onChange={(e) => setRows(Math.max(1, Math.min(20, Number(e.target.value) || 1)))} />
        </label>
      </div>

      {previewing ? (
        <p className="mt-2 flex items-center gap-2 text-xs text-slate-400">
          <Loader2 className="h-3 w-3 animate-spin" aria-hidden="true" /> Rendering…
        </p>
      ) : null}
      {previewError ? <p className="mt-2 text-xs text-amber-300">{previewError}</p> : null}
      {preview ? (
        <div className="mt-3" data-testid="judge-prompt-preview">
          {preview.warnings.map((warning) => (
            <p key={warning} className="mb-1 text-xs text-amber-300">
              {warning}
            </p>
          ))}
          <p className="text-xs text-slate-500">
            What the judge receives for {preview.rows.length} rows of {preview.dataset_name}
            {' '}(half of each label where possible). Nothing was sent to any model.
          </p>
          <ol className="mt-2 space-y-2">
            {preview.rows.map((row, index) => (
              <li key={index} className="rounded-sm border border-slate-800 p-2">
                <p className="text-[11px] uppercase text-slate-500">
                  Row {index + 1} · labelled {row.label === 'positive' ? 'present' : 'absent'}
                </p>
                <pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap text-xs text-slate-300">
                  {row.user_message}
                </pre>
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      <div className="mt-4 border-t border-slate-800 pt-3">
        <p className="text-xs text-slate-400">
          Try it: send these rows (at most 10) to a judge model and see how each reply is read.
          Nothing is recorded, the prompt is not locked, and no model is loaded — the model must
          already be served at the endpoint.
        </p>
        <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-3">
          <label className="text-xs text-slate-400">
            ENDPOINT
            <input className={INPUT} placeholder="http://…/v1" value={endpoint}
              onChange={(e) => setEndpoint(e.target.value)} data-testid="judge-prompt-try-endpoint" />
          </label>
          <label className="text-xs text-slate-400">
            MODEL
            <input className={INPUT} value={model} onChange={(e) => setModel(e.target.value)}
              data-testid="judge-prompt-try-model" />
          </label>
          <div className="flex items-end">
            <button
              type="button"
              onClick={() => void runTry()}
              disabled={!source || !datasetId || !endpoint.trim() || !model.trim() || trying}
              className="rounded-sm bg-emerald-700 px-3 py-1.5 text-sm text-white disabled:cursor-not-allowed disabled:bg-slate-700 disabled:text-slate-400"
              data-testid="judge-prompt-try"
            >
              {trying ? 'Asking the judge…' : 'Try it'}
            </button>
          </div>
        </div>
        {tryError ? <p className="mt-2 text-xs text-amber-300">{tryError}</p> : null}
        {tried ? (
          <div className="mt-3" data-testid="judge-prompt-try-result">
            <p className="text-xs text-slate-300">
              {tried.parsed} of {tried.rows.length} replies could be read
              {tried.unparseable ? `; ${tried.unparseable} could not` : ''}.
              {tried.stopped ? ` ${tried.stopped}` : ''}
            </p>
            <table className="mt-2 w-full text-xs">
              <thead>
                <tr className="text-left text-slate-500">
                  <th className="py-1 pr-2">Label</th>
                  <th className="py-1 pr-2">Judge said</th>
                  <th className="py-1 pr-2">Agrees</th>
                  <th className="py-1">Reply</th>
                </tr>
              </thead>
              <tbody>
                {tried.rows.map((row, index) => (
                  <tr key={index} className="border-t border-slate-800 align-top">
                    <td className="py-1 pr-2 text-slate-300">{row.label === 'positive' ? 'present' : 'absent'}</td>
                    <td className="py-1 pr-2 text-slate-300">
                      {row.answer ? `${row.answer.value} (${row.answer.positive ? 'present' : 'absent'})` : (
                        <span className="text-amber-300">{row.error}</span>
                      )}
                    </td>
                    <td className="py-1 pr-2">
                      {row.agrees_with_label == null ? '—' : row.agrees_with_label ? 'yes' : 'no'}
                    </td>
                    <td className="py-1 text-slate-400">
                      <pre className="max-h-24 overflow-auto whitespace-pre-wrap">{row.reply ?? ''}</pre>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </section>
  );
}

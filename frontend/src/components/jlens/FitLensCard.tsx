/**
 * Fit a J-lens artifact for the selected model.
 *
 * The panel used to say "fit one to enable it" and then offer no way to do it —
 * the only routes in were the REST endpoint and an MCP tool. A product that
 * names the remedy owes the user the affordance.
 *
 * THE CORPUS IS THE CALLER'S CHOICE, NOT THE SERVER'S (BR-007). The corpus is
 * part of the construction recipe and is recorded in `config.yaml`, so a
 * server-side default corpus would produce artifacts whose provenance says
 * nothing about what they were fitted on. This form therefore has no "use the
 * default corpus" button, and `corpus_name` is required rather than defaulted.
 *
 * THE PROMPT FLOOR IS SHOWN, NEVER SILENTLY SATISFIED. The fitter REFUSES a
 * corpus below `MIN_PROMPTS` (Appendix A.2) rather than warning, because an
 * under-fitted lens is indistinguishable from a fitted one by inspection. The
 * form mirrors that refusal by disabling submit and stating the count — it does
 * not pad, repeat, or sample up to the floor.
 */

import { useEffect, useRef, useState } from 'react';
import { Loader2, Wand2 } from 'lucide-react';
import { jlensApi } from '../../api/jlens';
import { getTaskStatus } from '../../api/models';

/** Appendix A.2 corpus floor. Mirrors `MIN_PROMPTS` in `ml/jlens_fitter.py`. */
export const MIN_FIT_PROMPTS = 100;

const POLL_MS = 5000;

interface FitLensCardProps {
  modelId: string;
  /** Refresh the registry once a fit commits an artifact. */
  onFitted: () => void;
  /**
   * Layers to pre-fill, set by "Fit the missing N" on the artifact card.
   *
   * TOPPING UP MUST NOT LOSE COVERAGE. A fit naming only the missing layers
   * would replace the artifact with one holding ONLY those — the server now
   * refuses that, and this form must not ask for it in the first place. The
   * prefill is therefore the UNION of what exists and what is missing.
   */
  prefillLayers?: number[] | null;
}

type Phase =
  | { kind: 'idle' }
  | { kind: 'queued'; taskId: string; state: string }
  | { kind: 'done'; taskId: string }
  | { kind: 'failed'; message: string };

export function parsePrompts(raw: string): string[] {
  return raw
    .split('\n')
    .map((line) => line.trim())
    .filter(Boolean);
}

/**
 * Parse an explicit layer list. Blank means "every layer" — expressed as null,
 * never as an empty array: `[]` reaches the server as "fit no layers at all"
 * and produces an artifact with no Jacobians in it.
 */
export function parseLayers(raw: string): number[] | null {
  const trimmed = raw.trim();
  if (!trimmed) return null;
  const parts = trimmed
    .split(/[,\s]+/)
    .filter(Boolean)
    .map((p) => Number.parseInt(p, 10));
  if (parts.some((n) => Number.isNaN(n) || n < 0)) return null;
  return parts.length ? parts : null;
}

export function FitLensCard({
  modelId,
  onFitted,
  prefillLayers,
}: FitLensCardProps) {
  const [open, setOpen] = useState(false);
  const [corpus, setCorpus] = useState('');
  const [corpusName, setCorpusName] = useState('');
  const [layersRaw, setLayersRaw] = useState('');
  const [freezeQk, setFreezeQk] = useState(true);
  const [probePrompt, setProbePrompt] = useState('');
  const [probeExpected, setProbeExpected] = useState('');
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Open and fill when the artifact card asks for a top-up. Keyed on the list's
  // content, not its identity: a parent re-render must not reopen a form the
  // user just closed.
  const prefillKey = (prefillLayers ?? []).join(',');
  useEffect(() => {
    if (!prefillKey) return;
    setLayersRaw(prefillKey.split(',').join(', '));
    setOpen(true);
  }, [prefillKey]);

  // Stop polling when the card unmounts or the model changes; otherwise the
  // loop outlives the component and calls onFitted for a model no longer shown.
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current);
    },
    []
  );

  const prompts = parsePrompts(corpus);
  const layers = parseLayers(layersRaw);
  const layersInvalid = layersRaw.trim() !== '' && layers === null;
  const running = phase.kind === 'queued';

  // The fixture is REFUSED, not warned about, when the intermediate appears in
  // the prompt: recovering a token that is already there proves nothing, so
  // such a fixture would pass against a lens encoding nothing at all. Checked
  // here as well as server-side so the user learns before waiting for a fit.
  const probeSelfEvident =
    probeExpected.trim() !== '' &&
    probePrompt.toLowerCase().includes(probeExpected.trim().toLowerCase());
  const probeComplete =
    probePrompt.trim() !== '' && probeExpected.trim() !== '' && !probeSelfEvident;

  const canSubmit =
    !running &&
    !!modelId &&
    prompts.length >= MIN_FIT_PROMPTS &&
    corpusName.trim() !== '' &&
    !layersInvalid &&
    probeComplete;

  const poll = (taskId: string) => {
    timer.current = setTimeout(async () => {
      try {
        const status = await getTaskStatus(taskId);
        if (status.state === 'SUCCESS') {
          setPhase({ kind: 'done', taskId });
          onFitted();
          return;
        }
        if (status.state === 'FAILURE') {
          setPhase({
            kind: 'failed',
            message: status.error ?? 'The fit failed with no reported reason.',
          });
          return;
        }
        setPhase({ kind: 'queued', taskId, state: status.state });
        poll(taskId);
      } catch (err) {
        setPhase({
          kind: 'failed',
          message: err instanceof Error ? err.message : 'Lost track of the fit.',
        });
      }
    }, POLL_MS);
  };

  const submit = async () => {
    if (!canSubmit) return;
    try {
      const accepted = await jlensApi.fit({
        model_id: modelId,
        prompts,
        layers,
        freeze_qk: freezeQk,
        corpus_name: corpusName.trim(),
        semantic_probe: {
          prompt: probePrompt,
          expected_intermediate: probeExpected,
          // The check reads out THROUGH the fitted Jacobian, so it must name a
          // layer that was fitted. Blank layers means all of them, in which
          // case the server picks the last.
          layer: layers ? layers[layers.length - 1] : null,
        },
      });
      setPhase({ kind: 'queued', taskId: accepted.task_id, state: 'PENDING' });
      poll(accepted.task_id);
    } catch (err) {
      setPhase({
        kind: 'failed',
        message: err instanceof Error ? err.message : 'Could not queue the fit.',
      });
    }
  };

  return (
    <section className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Fit a J-lens
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          disabled={!modelId}
          title={modelId ? undefined : 'Select a model first.'}
          className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          <Wand2 className="h-3 w-3" />
          {open ? 'Close' : 'Fit a lens…'}
        </button>
      </div>

      {open && (
        <div className="mt-3 space-y-3">
          <div>
            <label
              htmlFor="jlens-corpus"
              className="mb-1 block text-xs text-slate-600 dark:text-slate-400"
            >
              Corpus — one prompt per line
            </label>
            <textarea
              id="jlens-corpus"
              rows={6}
              value={corpus}
              onChange={(e) => setCorpus(e.target.value)}
              placeholder={'The capital of France is\nIn 1969, scientists\n…'}
              className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
            <p
              className={`mt-1 text-[11px] ${
                prompts.length >= MIN_FIT_PROMPTS
                  ? 'text-slate-500 dark:text-slate-500'
                  : 'text-amber-600 dark:text-amber-400'
              }`}
            >
              {prompts.length} / {MIN_FIT_PROMPTS} prompts. Below the floor the
              fit is refused, not warned about — an under-fitted lens is
              indistinguishable from a fitted one by inspection.
            </p>
          </div>

          <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            <div>
              <label
                htmlFor="jlens-corpus-name"
                className="mb-1 block text-xs text-slate-600 dark:text-slate-400"
              >
                Corpus name — recorded in the artifact's recipe
              </label>
              <input
                id="jlens-corpus-name"
                type="text"
                value={corpusName}
                onChange={(e) => setCorpusName(e.target.value)}
                placeholder="e.g. wikitext-factual-200"
                className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
            </div>
            <div>
              <label
                htmlFor="jlens-layers"
                className="mb-1 block text-xs text-slate-600 dark:text-slate-400"
              >
                Layers — blank fits every layer
              </label>
              <input
                id="jlens-layers"
                type="text"
                value={layersRaw}
                onChange={(e) => setLayersRaw(e.target.value)}
                placeholder="e.g. 24, 25"
                className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
              {layersInvalid && (
                <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">
                  Layers must be non-negative integers.
                </p>
              )}
            </div>
          </div>

          <div className="rounded border border-slate-200 p-2 dark:border-slate-700">
            <p className="mb-2 text-xs font-medium text-slate-600 dark:text-slate-400">
              Semantic check — required
            </p>
            <p className="mb-2 text-[11px] text-slate-500 dark:text-slate-500">
              The lens must recover a known intermediate the model never says
              out loud. Without this the check cannot run and{' '}
              <strong>nothing is published</strong> — the suite fails closed
              rather than clearing an artifact on a check it skipped.
            </p>
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <div>
                <label
                  htmlFor="jlens-probe-prompt"
                  className="mb-1 block text-xs text-slate-600 dark:text-slate-400"
                >
                  Probe prompt
                </label>
                <input
                  id="jlens-probe-prompt"
                  type="text"
                  value={probePrompt}
                  onChange={(e) => setProbePrompt(e.target.value)}
                  placeholder="The capital of France is"
                  className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                />
              </div>
              <div>
                <label
                  htmlFor="jlens-probe-expected"
                  className="mb-1 block text-xs text-slate-600 dark:text-slate-400"
                >
                  Expected intermediate
                </label>
                <input
                  id="jlens-probe-expected"
                  type="text"
                  value={probeExpected}
                  onChange={(e) => setProbeExpected(e.target.value)}
                  placeholder=" Paris"
                  className="w-full rounded border border-slate-300 bg-white px-2 py-1.5 text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                />
              </div>
            </div>
            {probeSelfEvident && (
              <p className="mt-1 text-[11px] text-red-600 dark:text-red-400">
                That intermediate already appears in the prompt, so recovering
                it proves nothing — a lens encoding nothing at all would pass.
              </p>
            )}
          </div>

          <label className="flex items-center gap-2 text-xs text-slate-600 dark:text-slate-400">
            <input
              type="checkbox"
              checked={freezeQk}
              onChange={(e) => setFreezeQk(e.target.checked)}
            />
            Freeze Q/K — the recipe variant is recorded per layer, and is
            INAPPLICABLE on a layer that does not attend rather than unused.
          </label>

          <button
            type="button"
            onClick={submit}
            disabled={!canSubmit}
            className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white transition-colors hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
          >
            {running ? (
              <span className="flex items-center gap-1">
                <Loader2 className="h-3 w-3 animate-spin" /> Fitting…
              </span>
            ) : (
              'Fit'
            )}
          </button>

          {phase.kind === 'queued' && (
            <p className="text-[11px] text-slate-500 dark:text-slate-500">
              Queued as{' '}
              <span className="font-mono">{phase.taskId.slice(0, 8)}</span> ·{' '}
              {phase.state}. The fit is GPU-bound and long-running; this page can
              be left.
            </p>
          )}
          {phase.kind === 'done' && (
            <p className="text-[11px] text-emerald-700 dark:text-emerald-400">
              Fitted. Validate the artifact above before reading out — presence
              is not validity.
            </p>
          )}
          {phase.kind === 'failed' && (
            <p className="text-[11px] text-red-600 dark:text-red-400" role="alert">
              {phase.message}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

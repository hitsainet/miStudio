/**
 * Adopt a lens someone else fitted, or publish one of ours.
 *
 * Fitting is a long GPU job per model, and a large body of pre-fitted lenses is
 * already published. This is the cheaper route when one exists — minutes and a
 * download instead of a GPU hour — and the other direction, so a lens fitted
 * here can be mounted by anyone else.
 *
 * PREVIEW BEFORE FETCH, ALWAYS. A mistyped path otherwise costs a
 * multi-gigabyte download and a slot on the single-GPU queue before anything
 * notices. The preview is read-only and returns the resolved commit, so the file
 * that was inspected is the file that arrives.
 */

import { useEffect, useState } from 'react';
import { Download, Eye, EyeOff, Loader2, Upload } from 'lucide-react';

import { jlensApi } from '../../api/jlens';
import type {
  JLensAcquireCandidate,
  JLensAcquirePreview,
} from '../../types/jlens';

interface AcquireLensCardProps {
  /** The model a downloaded lens would be attached to. */
  modelId: string;
  modelRepoId: string;
  /**
   * Whether this model's WEIGHTS are present.
   *
   * A lens is unusable without them — the readout runs a real forward pass, and
   * validating an acquired lens IS a readout. The server refuses at the door,
   * and saying so here means the prerequisite is visible before a 265 MB fetch
   * rather than discovered after it.
   */
  weightsPresent: boolean;
  /** Whether a validated artifact already exists, i.e. whether publish is live. */
  hasArtifact: boolean;
  /**
   * Optional hook for a parent that wants to know a job was queued.
   *
   * The card shows its OWN note, so a parent echoing this into a shared status
   * line is a hazard rather than a service: the panel's line is owned by the
   * intervention poller and holds the only copy of a completed rung-2 verdict.
   */
  onQueued?: (taskId: string, label: string) => void;
}

/** The server's own constraint on the corpus path segment (`PublishRequest`). */
const DATASET_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$/;

function formatBytes(n: number | null): string {
  if (!n) return '—';
  const units = ['B', 'KB', 'MB', 'GB'];
  let value = n;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value >= 100 || unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function AcquireLensCard({
  modelId,
  modelRepoId,
  weightsPresent,
  hasArtifact,
  onQueued,
}: AcquireLensCardProps) {
  const [open, setOpen] = useState(false);
  const [mode, setMode] = useState<'acquire' | 'publish'>('acquire');
  const [repoId, setRepoId] = useState('neuronpedia/jacobian-lens');
  /**
   * SEPARATE TOKENS PER MODE.
   *
   * One shared field silently reused a READ token as the publish credential —
   * masked, so the only signal was a label. The endpoint's pre-flight only
   * tests that *a* token exists, so the request 202s and fails inside the worker
   * after taking a slot on the single-GPU queue.
   */
  const [readToken, setReadToken] = useState('');
  const [writeToken, setWriteToken] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [preview, setPreview] = useState<JLensAcquirePreview | null>(null);
  const [selected, setSelected] = useState<string>('');
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  /**
   * A job is already queued from this card.
   *
   * `busy` releases at the 202, not at the job's terminal state, and nothing
   * else about the form changes — so a second click queued a second download of
   * the same multi-gigabyte file, which the worker then refuses on the staging
   * guard AFTER paying the bandwidth twice. Cleared deliberately, so starting
   * another is a decision rather than a double-click.
   */
  const [queued, setQueued] = useState<string | null>(null);

  // A PREVIEW BELONGS TO THE MODEL IT WAS COMPUTED FOR. Every `fits_envelope`
  // verdict came from that model's dimensions, so a list left on screen after
  // the model changes shows badges computed for other weights — and the
  // selection would send a lens for one model against another, which the
  // endpoint cannot catch and the worker only discovers after downloading it.
  useEffect(() => {
    setPreview(null);
    setSelected('');
    setQueued(null);
    setNote(null);
    setError(null);
  }, [modelId]);

  // Publish-side state.
  const [targetRepo, setTargetRepo] = useState('');
  const [dataset, setDataset] = useState('mistudio');
  const [createRepo, setCreateRepo] = useState(false);

  const runPreview = async () => {
    setBusy(true);
    setError(null);
    setNote(null);
    setPreview(null);
    setSelected('');
    try {
      const out = await jlensApi.previewRepo({
        repo_id: repoId.trim(),
        model_id: modelId || undefined,
        access_token: readToken || undefined,
      });
      setPreview(out);
      if (!out.candidates.length) {
        setError(`No .pt or .safetensors files in ${out.repo_id}.`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not read that repo.');
    } finally {
      setBusy(false);
    }
  };

  const runAcquire = async () => {
    if (!selected || !preview) return;
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const accepted = await jlensApi.acquire({
        model_id: modelId,
        repo_id: preview.repo_id,
        path_in_repo: selected,
        // THE RESOLVED SHA, not the branch the preview started from. `main`
        // moves, and an acquisition pinned to it is not reproducible.
        revision: preview.revision,
        access_token: readToken || undefined,
      });
      setNote(`Acquiring — queued as ${accepted.task_id.slice(0, 8)}…`);
      setQueued(accepted.task_id);
      onQueued?.(accepted.task_id, 'Acquire');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The acquisition was refused.');
    } finally {
      setBusy(false);
    }
  };

  const runPublish = async () => {
    setBusy(true);
    setError(null);
    setNote(null);
    try {
      const accepted = await jlensApi.publish({
        model_id: modelId,
        target_repo: targetRepo.trim(),
        access_token: writeToken || undefined,
        dataset: dataset.trim(),
        create_repo: createRepo,
      });
      setNote(`Publishing — queued as ${accepted.task_id.slice(0, 8)}…`);
      setQueued(accepted.task_id);
      onQueued?.(accepted.task_id, 'Publish');
    } catch (err) {
      setError(err instanceof Error ? err.message : 'The upload was refused.');
    } finally {
      setBusy(false);
    }
  };

  const chosen: JLensAcquireCandidate | undefined = preview?.candidates.find(
    (c) => c.path === selected,
  );

  return (
    <section
      className="rounded-lg border border-slate-200 bg-white p-3 dark:border-slate-700 dark:bg-slate-800"
      data-testid="jlens-acquire"
    >
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-xs font-medium text-slate-600 dark:text-slate-400">
          Published lenses
        </span>
        <span className="text-[10px] text-slate-500 dark:text-slate-500">
          download one, or share yours
        </span>
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="ml-auto flex items-center gap-1 rounded border border-slate-300 px-2 py-1 text-xs text-slate-700 hover:bg-slate-100 dark:border-slate-600 dark:text-slate-300 dark:hover:bg-slate-700"
        >
          <Download className="h-3 w-3" />
          {open ? 'Close' : 'Browse…'}
        </button>
      </div>

      {open && (
        <div className="mt-3 space-y-3">
          <div className="flex gap-1 text-xs">
            {(['acquire', 'publish'] as const).map((m) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                // ANNOUNCED, not conveyed by background colour alone.
                aria-pressed={mode === m}
                className={`rounded px-2 py-1 ${
                  mode === m
                    ? 'bg-emerald-600 text-white'
                    : 'border border-slate-300 text-slate-600 dark:border-slate-600 dark:text-slate-300'
                }`}
              >
                {m === 'acquire' ? 'Download' : 'Publish'}
              </button>
            ))}
          </div>

          {/* THE PREREQUISITE, STATED BEFORE THE FETCH. Validating an acquired
              lens means reading out through it, which needs the weights. The
              server refuses at the door; discovering that after a 265 MB
              download is the expensive way to learn it. */}
          {/* SUPPRESSED WHEN THERE IS NO MODEL TO NAME. On a fresh session
              `modelId` is '' and this rendered "**  ** is not downloaded",
              naming nothing — round 1 fixed the button and left the misleading
              string beside it. The no-model note below is the right message for
              that state. */}
          {mode === 'acquire' && Boolean(modelId) && !weightsPresent && (
            <p
              className="rounded border border-amber-300 bg-amber-50 p-2 text-[11px] text-amber-800 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-300"
              data-testid="jlens-acquire-weights-missing"
            >
              <strong>{modelRepoId}</strong> is not downloaded. A lens cannot be
              validated — or read out — without its weights, so this will be
              refused. Download the model first.
            </p>
          )}

          <label className="flex flex-col gap-1">
            <span className="text-xs text-slate-600 dark:text-slate-400">
              {mode === 'acquire'
                ? 'Source repository'
                : 'Target repository'}
            </span>
            <input
              type="text"
              value={mode === 'acquire' ? repoId : targetRepo}
              onChange={(e) =>
                mode === 'acquire'
                  ? setRepoId(e.target.value)
                  : setTargetRepo(e.target.value)
              }
              placeholder={mode === 'acquire' ? 'owner/repo' : 'you/jacobian-lenses'}
              data-testid="jlens-acquire-repo"
              className="rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs text-slate-900 dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
            />
          </label>

          <label className="flex flex-col gap-1">
            <span className="text-xs text-slate-600 dark:text-slate-400">
              HuggingFace token{' '}
              {mode === 'publish' && (
                <span className="text-amber-700 dark:text-amber-400">
                  — needs WRITE access
                </span>
              )}
            </span>
            <div className="flex gap-1">
              <input
                type={showToken ? 'text' : 'password'}
                value={mode === 'acquire' ? readToken : writeToken}
                onChange={(e) =>
                  mode === 'acquire'
                    ? setReadToken(e.target.value)
                    : setWriteToken(e.target.value)
                }
                placeholder="falls back to the configured token"
                data-testid="jlens-acquire-token"
                className="flex-1 rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
              />
              <button
                type="button"
                onClick={() => setShowToken((v) => !v)}
                aria-label={showToken ? 'Hide token' : 'Show token'}
                className="rounded border border-slate-300 px-2 dark:border-slate-600"
              >
                {showToken ? (
                  <EyeOff className="h-3 w-3" />
                ) : (
                  <Eye className="h-3 w-3" />
                )}
              </button>
            </div>
          </label>

          {mode === 'acquire' ? (
            <>
              <button
                type="button"
                onClick={() => void runPreview()}
                disabled={busy || !repoId.trim()}
                className="rounded border border-slate-300 px-3 py-1.5 text-xs text-slate-700 disabled:opacity-50 dark:border-slate-600 dark:text-slate-300"
              >
                {busy ? (
                  <span className="flex items-center gap-1">
                    <Loader2 className="h-3 w-3 animate-spin" /> Reading…
                  </span>
                ) : (
                  'Look inside'
                )}
              </button>

              {preview && (
                <div className="space-y-1">
                  <p className="text-[10px] text-slate-500 dark:text-slate-500">
                    {preview.candidates.length} candidate
                    {preview.candidates.length === 1 ? '' : 's'} at{' '}
                    <span className="font-mono">
                      {preview.revision.slice(0, 12)}
                    </span>
                    {' — pinned, so what you inspect is what arrives'}
                  </p>
                  <ul className="max-h-56 space-y-[2px] overflow-y-auto">
                    {preview.candidates.map((c) => (
                      <li key={c.path}>
                        <label className="flex cursor-pointer items-center gap-2 rounded px-1 py-[2px] hover:bg-slate-50 dark:hover:bg-slate-700/40">
                          <input
                            type="radio"
                            name="jlens-candidate"
                            checked={selected === c.path}
                            onChange={() => setSelected(c.path)}
                            className="h-3 w-3"
                          />
                          <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-slate-800 dark:text-slate-100">
                            {c.path}
                          </span>
                          <span className="shrink-0 font-mono text-[10px] tabular-nums text-slate-500">
                            {formatBytes(c.size_bytes)}
                          </span>
                          {/* THE FIELD TO READ FIRST. A file beside a config
                              declares which weights it was fitted for, so its
                              identity can be CHECKED; one without leaves the
                              pairing resting on your assertion. */}
                          <span
                            className={`shrink-0 rounded px-1 text-[9px] ${
                              c.has_config
                                ? 'bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300'
                                : 'bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300'
                            }`}
                          >
                            {c.has_config ? 'declares a config' : 'no config'}
                          </span>
                          {c.fits_envelope === false && (
                            <span
                              className="shrink-0 rounded bg-red-100 px-1 text-[9px] text-red-800 dark:bg-red-900/40 dark:text-red-300"
                              title={c.envelope_detail ?? undefined}
                            >
                              too large
                            </span>
                          )}
                        </label>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* BOTH OUTCOMES EXPLAINED, because the badge cannot predict
                  either. `has_config` is directory-level presence of a file
                  with that name; the verdict comes from whether the config
                  NAMES a model, and one naming a DIFFERENT model is a hard
                  refusal after the bytes are already spent. Explaining only the
                  config-less case left the top-sorted, green-badged candidate
                  the least described. */}
              {chosen && (
                <p
                  className="text-[10px] text-amber-700 dark:text-amber-400"
                  data-testid="jlens-identity-note"
                >
                  {chosen.has_config ? (
                    <>
                      A config sits beside this file. If it names these weights
                      the artifact records <strong>verified</strong>; if it
                      names other weights the acquisition is{' '}
                      <strong>refused</strong> after the download; if it names
                      none, <strong>unverified</strong>.
                    </>
                  ) : (
                    <>
                      This file has no config beside it, so weight identity
                      cannot be checked. It will be adopted as{' '}
                      <strong>unverified</strong> and the artifact will record
                      that the pairing rests on your assertion.
                    </>
                  )}
                </p>
              )}

              {/* NULL IS NOT "FINE". The endpoint computes a verdict only
                  `if dims and c.size_bytes`, so null also means the Hub
                  reported no size, or the model row lacks the dimensions to
                  derive a bound. In both the check never ran — and permitting
                  silently is the case the preview exists for. */}
              {chosen && chosen.fits_envelope === null && (
                <p
                  className="text-[10px] text-amber-700 dark:text-amber-400"
                  data-testid="jlens-acquire-envelope-unknown"
                >
                  The size check did not run for this file — either the source
                  reported no size, or this model&rsquo;s recorded dimensions are
                  incomplete. It may still be refused after the download.
                </p>
              )}
              {chosen?.fits_envelope === false && (
                <p
                  className="text-[11px] text-amber-700 dark:text-amber-400"
                  data-testid="jlens-acquire-too-large"
                >
                  {chosen.envelope_detail ??
                    'This file is larger than a lens for these weights could be.'}{' '}
                  Downloading it would spend the bandwidth and a GPU-queue slot
                  to reach the same verdict — which is what this preview exists
                  to avoid.
                </p>
              )}
              {!modelId && (
                <p
                  className="text-[11px] text-amber-700 dark:text-amber-400"
                  data-testid="jlens-acquire-no-model"
                >
                  Choose a model above first — a lens is adopted FOR one, and
                  without it the request is refused with a 404 naming an empty id.
                </p>
              )}
              <button
                type="button"
                onClick={() => void runAcquire()}
                disabled={
                  busy ||
                  !selected ||
                  !modelId ||
                  Boolean(queued) ||
                  chosen?.fits_envelope === false
                }
                data-testid="jlens-acquire-run"
                className="rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
              >
                Download &amp; validate
              </button>
            </>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3">
                <label className="flex flex-col gap-1">
                  <span className="text-xs text-slate-600 dark:text-slate-400">
                    Corpus segment
                  </span>
                  <input
                    type="text"
                    value={dataset}
                    onChange={(e) => setDataset(e.target.value)}
                    data-testid="jlens-publish-dataset"
                    aria-invalid={!DATASET_PATTERN.test(dataset.trim())}
                    // REFERENCED ONLY WHEN IT EXISTS. The helper renders only on
                    // an invalid value, so an unconditional reference dangles in
                    // the default state — `mistudio` is valid.
                    aria-describedby={
                      DATASET_PATTERN.test(dataset.trim())
                        ? undefined
                        : 'jlens-dataset-help'
                    }
                    className="rounded border border-slate-300 bg-white px-2 py-1.5 font-mono text-xs dark:border-slate-600 dark:bg-slate-900 dark:text-slate-100"
                  />
                  {/* MIRRORED FROM THE SERVER. It is a path segment, so the
                      endpoint constrains it — and the obvious value to type is
                      the corpus's own name, `wikitext/wikitext-103`, whose
                      slash 422s with a regex the form gives no hint about. */}
                  {!DATASET_PATTERN.test(dataset.trim()) && (
                    <span
                      id="jlens-dataset-help"
                      className="text-[10px] text-amber-700 dark:text-amber-400"
                      data-testid="jlens-publish-dataset-invalid"
                    >
                      Letters, digits, dot, dash and underscore only — it is a
                      path segment, not a dataset id.
                    </span>
                  )}
                </label>
                <label className="mt-5 flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-400">
                  <input
                    type="checkbox"
                    checked={createRepo}
                    onChange={(e) => setCreateRepo(e.target.checked)}
                    className="h-3 w-3"
                  />
                  Create the repo if absent
                </label>
              </div>

              {/* WHAT LEAVES, ACCURATELY. This said the local validation
                  verdict "does not travel" — false, and self-contradicting two
                  clauses earlier: the README carries every check's name, status
                  and detail, and the README is uploaded. Only `validation.json`
                  is withheld, which is a file-level fact rather than a claim
                  about what a reader learns. A test asserted the false
                  sentence, so correcting the copy turned the suite red. */}
              <p
                className="text-[10px] text-slate-500 dark:text-slate-500"
                data-testid="jlens-publish-note"
              >
                Uploads the checkpoint, its recipe, any recorded interventions
                and the convergence trace, plus a README that lists{' '}
                <strong>every check and its status</strong> — including the two
                recorded <em>deferred</em>, which need a live external consumer
                and have never been run. The machine-readable{' '}
                <code>validation.json</code> itself is withheld, so a reader
                cannot mistake this installation&rsquo;s verdict for the
                lens&rsquo;s own.
                {createRepo && (
                  <>
                    {' '}
                    A repo created here is <strong>public</strong>.
                  </>
                )}
              </p>

              {/* THE OTHER GATE, NAMED. `hasArtifact` is slug presence only;
                  the endpoint ALSO refuses an artifact whose stored verdict no
                  longer matches its current weights, and the listing
                  deliberately carries no validity field — "an artifact's
                  validity is the outcome of running the suite, not a property
                  of the file". So this cannot be checked here, and saying that
                  a present artifact is the only requirement would be a claim
                  the card cannot keep. */}
              {hasArtifact && (
                <p className="text-[10px] text-slate-500 dark:text-slate-500">
                  Publishing also requires a validation verdict matching the
                  lens's current weights. If the model was re-downloaded since
                  it was validated, this will be refused and the reason will say
                  so.
                </p>
              )}
              {!hasArtifact && Boolean(modelRepoId) && (
                <p
                  className="rounded border border-amber-300 bg-amber-50 p-2 text-[11px] text-amber-800 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-300"
                  data-testid="jlens-publish-no-artifact"
                >
                  There is no published lens for {modelRepoId} yet. Fit or
                  download one first — a staged artifact is not published and is
                  not shipped.
                </p>
              )}

              <button
                type="button"
                onClick={() => void runPublish()}
                disabled={
                  busy ||
                  !targetRepo.trim() ||
                  !hasArtifact ||
                  !DATASET_PATTERN.test(dataset.trim()) ||
                  Boolean(queued)
                }
                data-testid="jlens-publish-run"
                className="flex items-center gap-1 rounded bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-slate-700"
              >
                <Upload className="h-3 w-3" />
                Publish
              </button>
            </>
          )}

          {/* IN BOTH MODES. It lived inside the acquire branch, so after a
              publish the button greyed out with no stated reason and the only
              recovery was to switch modes and click the link there. */}
          {queued && (
            <p
              className="text-[10px] text-slate-500 dark:text-slate-500"
              data-testid="jlens-queued-note"
            >
              A job is already queued from this card. Watch Running Work, or{' '}
              <button
                type="button"
                onClick={() => setQueued(null)}
                className="underline"
              >
                start another
              </button>
              .
            </p>
          )}
          {note && (
            <p className="text-[11px] text-emerald-700 dark:text-emerald-400">
              {note}
            </p>
          )}
          {error && (
            <p className="text-[11px] text-red-600 dark:text-red-400" role="alert">
              {error}
            </p>
          )}
        </div>
      )}
    </section>
  );
}

/**
 * ManifestDrawer (Feature 017) — a slide-over that renders a self-contained,
 * reproducible validation manifest: the intervention config, seeds, baseline,
 * null summary, per-edge effect sizes, and a Reproduce action.
 *
 * Copy discipline: validation IS the rung-2 causal tier, so causal language is
 * allowed HERE. A PASS is "causally validated (rung 2)"; a fail is "tested,
 * did not validate". Backend `verdict.reason` strings render verbatim.
 */

import { useEffect, useRef, useState } from 'react';
import { X as XIcon, RefreshCw, Copy, ArrowRight } from 'lucide-react';
import { circuitsApi } from '../../api/circuits';
import type { ValidationManifest, ValidationEdge } from '../../types/circuits';

function EdgeLabel({ up, down }: Pick<ValidationEdge, 'up' | 'down'>) {
  return (
    <span className="font-mono text-slate-700 dark:text-slate-300 whitespace-nowrap">
      L{up.layer}:f{up.feature_idx} → L{down.layer}:f{down.feature_idx}
    </span>
  );
}

function VerdictChip({ edge }: { edge: ValidationEdge }) {
  if (edge.rung === 2 && edge.verdict.passed) {
    return (
      <span className="rounded bg-emerald-500/10 text-emerald-300 px-1.5 py-0.5 text-[10px] whitespace-nowrap">
        causally validated (rung 2)
      </span>
    );
  }
  return (
    <span className="rounded bg-slate-100 dark:bg-slate-700/60 text-slate-600 dark:text-slate-400 px-1.5 py-0.5 text-[10px] whitespace-nowrap">
      tested, did not validate
    </span>
  );
}

export function ManifestDrawer({
  manifestId,
  onClose,
  onNavigate,
}: {
  manifestId: string;
  onClose: () => void;
  // Re-point the drawer at another manifest (used to jump to the fresh
  // reproduction child once it lands). Falls back to a plain notice if absent.
  onNavigate?: (id: string) => void;
}) {
  const [manifest, setManifest] = useState<ValidationManifest | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reproStatus, setReproStatus] = useState<string | null>(null);
  const [reproBusy, setReproBusy] = useState(false);
  // The child reproduction manifest, once we find it by polling (R1 P4 —
  // Reproduce used to fire a task whose verdict landed on a manifest the user
  // never saw).
  const [reproChildId, setReproChildId] = useState<string | null>(null);
  const mountedRef = useRef(true);
  useEffect(() => () => { mountedRef.current = false; }, []);

  useEffect(() => {
    setManifest(null);
    setError(null);
    setReproStatus(null);
    setReproChildId(null);
    circuitsApi
      .getManifest(manifestId)
      .then(setManifest)
      .catch((e) => setError(String(e.message ?? e)));
  }, [manifestId]);

  // After a Reproduce, poll the run's manifests for the child reproduction
  // (parent_manifest_id === this manifest) so its verdict is reachable.
  const pollForChild = (runId: string) => {
    let tries = 0;
    const tick = () => {
      if (!mountedRef.current) return;
      circuitsApi.listManifests({ discovery_run_id: runId })
        .then((r) => {
          if (!mountedRef.current) return;
          const child = r.manifests.find(
            (m) => m.kind === 'reproduction' && m.parent_manifest_id === manifestId);
          if (child) {
            setReproChildId(child.id);
            setReproStatus('complete');
          } else if (tries++ < 20) {
            setTimeout(tick, 2500);
          }
        })
        .catch(() => { if (mountedRef.current && tries++ < 20) setTimeout(tick, 2500); });
    };
    tick();
  };

  const reproduce = () => {
    setReproBusy(true);
    setError(null);
    setReproChildId(null);
    circuitsApi
      .reproduceManifest(manifestId)
      .then((r) => {
        setReproStatus(r.status);
        if (manifest?.discovery_run_id) pollForChild(manifest.discovery_run_id);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setReproBusy(false));
  };

  const payload = manifest?.payload;
  // edge_batch carries payload.edges; a reproduction carries reproduced_edges.
  const edges = payload?.edges ?? payload?.reproduced_edges ?? [];
  const config = (payload?.config ?? {}) as Record<string, unknown>;
  const cfg = (k: string) => (config[k] != null ? String(config[k]) : '—');

  return (
    <div className="fixed inset-0 z-50 flex justify-end" role="dialog" aria-modal="true">
      {/* backdrop */}
      <button
        aria-label="Close manifest"
        className="absolute inset-0 bg-black/50"
        onClick={onClose}
      />
      <div className="relative h-full w-full max-w-xl overflow-y-auto bg-white dark:bg-slate-900 border-l border-slate-300 dark:border-slate-700 shadow-xl">
        <div className="sticky top-0 flex items-center justify-between gap-2 border-b border-slate-300 dark:border-slate-700 bg-white dark:bg-slate-900 px-5 py-3">
          <div className="min-w-0">
            <h2 className="text-sm font-semibold text-slate-900 dark:text-slate-100">Validation manifest</h2>
            <p className="font-mono text-[11px] text-slate-500 truncate">{manifestId}</p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400"
            title="Close"
          >
            <XIcon className="w-4 h-4" />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
              {error}
            </div>
          )}

          {!manifest && !error && (
            <p className="text-xs text-slate-500">Loading manifest…</p>
          )}

          {manifest && (
            <>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
                <div className="text-slate-500">kind</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">{manifest.kind}</div>
                <div className="text-slate-500">intervention</div>
                <div className="text-slate-700 dark:text-slate-300">
                  {payload?.intervention?.kind ?? '—'}
                </div>
                <div className="text-slate-500">baseline</div>
                <div className="text-slate-700 dark:text-slate-300">
                  {payload?.intervention?.baseline ?? cfg('baseline')}
                </div>
                <div className="text-slate-500">ordering</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">{payload?.ordering ?? '—'}</div>
                <div className="text-slate-500">k</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">{payload?.k ?? '—'}</div>
                <div className="text-slate-500">seeds</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">
                  {payload?.seeds?.join(', ') ?? '—'}
                </div>
                <div className="text-slate-500">prompts/edge</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">{cfg('prompts_per_edge')}</div>
                <div className="text-slate-500">sign_frac</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">{cfg('sign_frac')}</div>
                <div className="text-slate-500">survival</div>
                <div className="text-slate-700 dark:text-slate-300 font-mono">
                  {payload?.survival != null ? `${(payload.survival * 100).toFixed(0)}%` : '—'}
                </div>
              </div>

              {payload?.null_summary && (
                <div className="rounded border border-slate-300 dark:border-slate-700/60 bg-slate-100 dark:bg-slate-800/40 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-400">
                  <span className="font-medium text-slate-700 dark:text-slate-300">Null summary:</span>{' '}
                  {payload.null_summary.kind ?? 'shuffled null'}
                  {payload.null_summary.samples != null &&
                    ` · ${payload.null_summary.samples} samples`}
                  {payload.null_summary.percentile != null &&
                    ` · p${payload.null_summary.percentile}`}
                </div>
              )}

              {/* reproduction manifests carry the tolerance verdict, nested
                  under payload.verdict (R1 #3 — was read at top level and
                  always showed "did not reproduce"). Three-state: true/false/
                  null (nothing compared). */}
              {manifest.kind === 'reproduction' && payload?.verdict && (
                <div
                  className={`rounded border px-3 py-2 text-[11px] ${
                    payload.verdict.within_tolerance === true
                      ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-300'
                      : payload.verdict.within_tolerance === false
                        ? 'border-amber-500/30 bg-amber-500/10 text-amber-300'
                        : 'border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-700/40 text-slate-700 dark:text-slate-300'
                  }`}
                >
                  <span className="font-medium">Reproduction verdict:</span>{' '}
                  {payload.verdict.within_tolerance === true
                    ? 'within tolerance — the manifest reproduced'
                    : payload.verdict.within_tolerance === false
                      ? 'outside tolerance — the manifest did not reproduce'
                      : (payload.verdict.reason ?? 'no verdict')}
                  {payload.verdict.max_delta != null
                    && ` (max Δ ${payload.verdict.max_delta})`}
                </div>
              )}

              <div>
                <h3 className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1.5">
                  Per-edge effect sizes
                </h3>
                {edges.length === 0 ? (
                  <p className="text-xs text-slate-500">This manifest has no edge verdicts.</p>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-slate-500 text-left">
                          <th className="py-1 pr-3 font-medium">edge</th>
                          <th className="py-1 pr-3 font-medium">ES</th>
                          <th className="py-1 pr-3 font-medium" title="null percentile value">
                            null
                          </th>
                          <th className="py-1 pr-3 font-medium">sign</th>
                          <th className="py-1 pr-3 font-medium">n</th>
                          <th className="py-1 font-medium">verdict</th>
                        </tr>
                      </thead>
                      <tbody>
                        {edges.map((e, i) => (
                          <tr key={i} className="border-t border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300">
                            <td className="py-1 pr-3">
                              <EdgeLabel up={e.up} down={e.down} />
                            </td>
                            <td className="py-1 pr-3 font-mono">{e.effect_size.toFixed(3)}</td>
                            <td className="py-1 pr-3 font-mono text-slate-600 dark:text-slate-400">
                              {e.null_percentile_value.toFixed(3)}
                            </td>
                            <td className="py-1 pr-3 font-mono">
                              {e.sign_consistency.toFixed(2)}
                            </td>
                            <td className="py-1 pr-3 font-mono text-slate-600 dark:text-slate-400">{e.n_prompts}</td>
                            <td className="py-1">
                              <div className="flex flex-col gap-0.5">
                                <VerdictChip edge={e} />
                                <span className="text-[10px] text-slate-500">
                                  {e.verdict.reason}
                                </span>
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>

              <div className="border-t border-slate-300 dark:border-slate-700/60 pt-3 space-y-2">
                <button
                  onClick={reproduce}
                  disabled={reproBusy || manifest.kind !== 'edge_batch'}
                  className="flex items-center gap-1.5 rounded bg-violet-600 hover:bg-violet-500 px-3 py-1.5 text-xs text-white disabled:opacity-50"
                  title={
                    manifest.kind === 'edge_batch'
                      ? 'Re-execute this manifest from its payload and check reproduction tolerance'
                      : 'Only edge_batch manifests are reproducible'
                  }
                >
                  {reproBusy ? (
                    <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  ) : (
                    <Copy className="w-3.5 h-3.5" />
                  )}
                  Reproduce
                </button>
                {reproStatus && !reproChildId && (
                  <p className="text-[11px] text-emerald-300">
                    Reproduction {reproStatus} — a fresh manifest with a tolerance verdict is
                    being computed.
                  </p>
                )}
                {reproChildId && (
                  onNavigate ? (
                    <button
                      onClick={() => onNavigate(reproChildId)}
                      className="flex items-center gap-1.5 text-[11px] text-emerald-300 hover:text-emerald-200"
                    >
                      Reproduction complete — open the new manifest
                      <ArrowRight className="w-3 h-3" />
                    </button>
                  ) : (
                    <p className="text-[11px] text-emerald-300">
                      Reproduction complete — new manifest{' '}
                      <span className="font-mono">{reproChildId.slice(0, 8)}</span>.
                    </p>
                  )
                )}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default ManifestDrawer;

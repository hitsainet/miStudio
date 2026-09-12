/**
 * Circuits panel (Feature 018) — the review/promotion surface for
 * cross-layer circuits. Every circuit and edge shows its evidence rung via
 * SERVER-rendered language (IDL-35); promotion is a badge, not a gate (and
 * reversible). List rows are slim summaries; full evidence loads on expand.
 * Discovery/validation tabs (016/017) join this panel as they land.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  GitBranch, ArrowUpRight, ArrowDownRight, Trash2, Download, Layers,
  Pencil, Check, X as XIcon, FileDown, Upload, Plus, Camera, Search,
  AlertTriangle, RotateCcw, ShieldCheck, FileText, RefreshCw, Microscope, Gauge,
  Wand2,
} from 'lucide-react';
import { circuitsApi } from '../../api/circuits';
import type {
  Circuit, CircuitSummary, CircuitEdge,
  CircuitCapture, CircuitCaptureCreate,
  DiscoveryRun, DiscoveryReport, DiscoveryCandidate, DiscoveryCreate,
  DiscoveryGranularity, DiscoveryMode, DiscoverySeedRef,
  ValidateConfig,
} from '../../types/circuits';
import { RungChip } from '../circuits/RungChip';
import { ManifestDrawer } from '../circuits/ManifestDrawer';
import { COMPONENTS } from '../../config/brand';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useSAEsStore } from '../../stores/saesStore';
import { useSteeringStore } from '../../stores/steeringStore';
import { SAEStatus } from '../../types/sae';

// Polling terminal states shared by both run kinds.
const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'estimated']);
const isActive = (status: string) => !TERMINAL.has(status);

function StatusBadge({ status }: { status: string }) {
  const tone =
    status === 'completed' ? 'bg-emerald-500/10 text-emerald-300'
    : status === 'failed' ? 'bg-red-500/10 text-red-300'
    : status === 'cancelled' ? 'bg-slate-100 text-slate-600 dark:bg-slate-700/60 dark:text-slate-400'
    : status === 'estimated' ? 'bg-sky-500/10 text-sky-300'
    : 'bg-violet-500/10 text-violet-300';
  return (
    <span className={`rounded px-1.5 py-0.5 text-[10px] ${tone}`}>{status}</span>
  );
}

function ProgressBar({ value }: { value: number | null }) {
  // Backend emits progress as 0–100 percent (run.progress) — trust that; a
  // real 1% must not be rescaled to 100%.
  const pct = Math.max(0, Math.min(100, Math.round(value ?? 0)));
  return (
    <div className="h-1.5 w-full rounded bg-slate-100 dark:bg-slate-700/60 overflow-hidden">
      <div className="h-full bg-violet-500 transition-all" style={{ width: `${pct}%` }} />
    </div>
  );
}

function EdgeRow({ edge }: { edge: CircuitEdge }) {
  const label = (n: CircuitEdge['up']) =>
    n.kind === 'cluster' ? `cluster:${n.cluster_profile_id}` : `#${n.feature_idx}`;
  return (
    <tr className={`text-xs ${edge.type === 'persistence' ? 'opacity-60' : ''}`}>
      <td className="py-1 pr-3 font-mono text-slate-700 dark:text-slate-300 whitespace-nowrap">
        L{edge.up.layer} {label(edge.up)} → L{edge.down.layer} {label(edge.down)}
      </td>
      <td className="py-1 pr-3">
        <span className={`rounded px-1 py-0.5 text-[10px] ${
          edge.type === 'persistence' ? 'bg-amber-500/10 text-amber-300'
          : edge.type === 'attention_mediated' ? 'bg-violet-500/10 text-violet-300'
          : 'bg-slate-100 text-slate-700 dark:bg-slate-700/60 dark:text-slate-300'}`}>
          {edge.type}
        </span>
      </td>
      <td className="py-1 pr-3 font-mono text-slate-600 dark:text-slate-400">R{edge.rung}</td>
      <td className="py-1 pr-3 text-slate-600 dark:text-slate-400">
        {[
          edge.coactivation?.pmi != null ? `PMI ${edge.coactivation.pmi.toFixed(2)}` : null,
          edge.coactivation?.support != null ? `n=${edge.coactivation.support}` : null,
          edge.attribution?.score != null ? `attr ${edge.attribution.score.toFixed(2)}` : null,
          edge.weight_prior != null ? `prior ${edge.weight_prior.toFixed(2)}` : null,
          edge.effect_size != null ? `ES ${edge.effect_size.toFixed(2)}` : null,
        ].filter(Boolean).join(' · ')}
      </td>
      <td className="py-1 text-slate-500">
        {edge.validation_manifest_ref && (
          <span className="font-mono text-[10px]" title="Validation manifest">
            {edge.validation_manifest_ref}
          </span>
        )}
      </td>
    </tr>
  );
}

function CircuitDetail({ id, onChanged }: { id: string; onChanged: () => void }) {
  const [circuit, setCircuit] = useState<Circuit | null>(null);
  const [error, setError] = useState<string | null>(null);       // load failures only
  const [saveError, setSaveError] = useState<string | null>(null); // keeps the draft alive (R2 B4)
  const [showPersistence, setShowPersistence] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState('');
  const [draftNarrative, setDraftNarrative] = useState('');
  // Faithfulness (rung 3) trigger + its own poll lifecycle.
  const [faithMode, setFaithMode] = useState<'necessity' | 'both'>('necessity');
  const [faithBusy, setFaithBusy] = useState(false);
  const [faithError, setFaithError] = useState<string | null>(null);
  // Calibration (Feature 20) trigger + poll lifecycle. The judge endpoint/model
  // are entered by the operator (an OpenAI-compatible endpoint — the served
  // miLLM works, a stronger model grades better).
  const [judgeEndpoint, setJudgeEndpoint] = useState('');
  const [judgeModel, setJudgeModel] = useState('');
  const [calibBusy, setCalibBusy] = useState(false);
  const [calibError, setCalibError] = useState<string | null>(null);

  const load = useCallback(() => {
    circuitsApi.get(id).then(setCircuit).catch((e) => setError(String(e.message ?? e)));
  }, [id]);

  useEffect(() => { load(); }, [load]);

  // Poll while a faithfulness run is pending/running (the run has its own
  // lifecycle — the circuit otherwise stays put). Stops on terminal.
  useEffect(() => {
    const s = circuit?.faithfulness_status;
    if (s !== 'pending' && s !== 'running') return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [circuit?.faithfulness_status, load]);

  const runFaithfulness = () => {
    if (!circuit) return;
    setFaithError(null);
    setFaithBusy(true);
    circuitsApi.startFaithfulness(circuit.id, { mode: faithMode })
      .then(() => load())
      .catch((e) => setFaithError(String(e.detail ?? e.message ?? e)))
      .finally(() => setFaithBusy(false));
  };

  // Poll while a calibration run is in flight (its own lifecycle, like faithfulness).
  useEffect(() => {
    const s = circuit?.calibration_status;
    if (s !== 'pending' && s !== 'running') return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [circuit?.calibration_status, load]);

  const runCalibration = () => {
    if (!circuit) return;
    setCalibError(null);
    setCalibBusy(true);
    circuitsApi.startCalibration(circuit.id, {
      judge_endpoint: judgeEndpoint.trim(),
      judge_model: judgeModel.trim(),
    })
      .then(() => load())
      .catch((e) => setCalibError(String(e.detail ?? e.message ?? e)))
      .finally(() => setCalibBusy(false));
  };

  if (error && !circuit) return <p className="text-xs text-red-300 mt-2">{error}</p>;
  if (!circuit) return <p className="text-xs text-slate-500 mt-2">Loading evidence…</p>;

  const edges = circuit.edges.filter((e) => showPersistence || e.type !== 'persistence');
  const hiddenCount = circuit.edges.length - edges.length;

  const saveEdit = () => {
    setSaveError(null);
    // Send the version we loaded (017 Task 3.0): if a validation pass bumped
    // the circuit meanwhile, this 409s instead of silently clobbering it.
    circuitsApi.update(circuit.id, {
      name: draftName, narrative: draftNarrative,
      expected_version: circuit.version,
    })
      .then((updated) => { setCircuit(updated); setEditing(false); onChanged(); })
      .catch((e) => setSaveError(
        // 409 → the draft is preserved and the user is told to reload (R2 B4).
        String(e.message ?? e).includes('409')
          ? 'This circuit changed since you opened it (a validation pass may have run). Reload and re-apply your edit.'
          : String(e.message ?? e)));
  };

  return (
    <div className="mt-3 border-t border-slate-200 dark:border-slate-700/60 pt-3 space-y-3">
      {editing ? (
        <div className="space-y-2">
          <input
            className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1 text-sm text-slate-900 dark:text-slate-100"
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
            aria-label="Circuit name"
          />
          <textarea
            className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1 text-sm text-slate-900 dark:text-slate-100"
            rows={4}
            value={draftNarrative}
            onChange={(e) => setDraftNarrative(e.target.value)}
            placeholder="Narrative (markdown)"
            aria-label="Circuit narrative"
          />
          {saveError && (
            <p className="text-xs text-red-300">{saveError}</p>
          )}
          <div className="flex gap-2">
            <button onClick={saveEdit}
              className="flex items-center gap-1 rounded bg-emerald-600 px-2 py-1 text-xs text-white">
              <Check className="w-3 h-3" /> Save
            </button>
            <button onClick={() => setEditing(false)}
              className="flex items-center gap-1 rounded bg-slate-100 dark:bg-slate-700 px-2 py-1 text-xs text-slate-800 dark:text-slate-200">
              <XIcon className="w-3 h-3" /> Cancel
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-start gap-2">
          <div className="flex-1 min-w-0">
            {circuit.narrative ? (
              <div className="text-sm text-slate-600 dark:text-slate-400">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    // House dark-theme markdown mapping (no typography plugin — R2 B10)
                    p: ({ children }) => <p className="mb-2 leading-relaxed">{children}</p>,
                    strong: ({ children }) => <strong className="font-semibold text-slate-800 dark:text-slate-200">{children}</strong>,
                    ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
                    ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
                    li: ({ children }) => <li className="text-slate-700 dark:text-slate-300">{children}</li>,
                    code: ({ children }) => <code className="px-1 py-0.5 rounded bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-mono text-xs">{children}</code>,
                    table: ({ children }) => (
                      <div className="my-2 overflow-x-auto">
                        <table className="text-xs border border-slate-300 dark:border-slate-700 border-collapse">{children}</table>
                      </div>
                    ),
                    th: ({ children }) => <th className="px-2 py-1 border border-slate-300 dark:border-slate-700 text-slate-800 dark:text-slate-200">{children}</th>,
                    td: ({ children }) => <td className="px-2 py-1 border border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300">{children}</td>,
                  }}
                >
                  {circuit.narrative}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-xs text-slate-500 italic">No narrative yet.</p>
            )}
          </div>
          <button
            onClick={() => { setDraftName(circuit.name); setDraftNarrative(circuit.narrative ?? ''); setEditing(true); }}
            className="p-1 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400 shrink-0"
            title="Edit name & narrative"
          >
            <Pencil className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      <div>
        <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400 mb-1">Members by layer</h4>
        {[...new Set(circuit.members.map((m) => m.layer))].sort((a, b) => a - b).map((layer) => {
          const layerMembers = circuit.members.filter((m) => m.layer === layer);
          const count = layerMembers.reduce(
            (n, m) => n + (m.member_kind === 'cluster_ref' ? (m.expanded_members?.length ?? 0) : 1), 0);
          return (
            <div key={layer} className="text-xs text-slate-700 dark:text-slate-300 mb-1">
              <span className="font-mono text-slate-500 mr-2">L{layer}</span>
              <span className="text-slate-500 mr-2">({count}/20)</span>
              {layerMembers.map((m, i) => (
                <span key={i} className="mr-2">
                  {m.member_kind === 'cluster_ref'
                    ? `[cluster ${m.cluster_name ?? m.cluster_profile_id}]`
                    : `#${m.feature?.feature_idx}${m.feature?.label ? ` ${m.feature.label}` : ''}`}
                </span>
              ))}
            </div>
          );
        })}
      </div>

      {circuit.edges.length > 0 && (
        <div className="overflow-x-auto">
          <div className="flex items-center gap-3 mb-1">
            <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400">Edges</h4>
            {hiddenCount > 0 && !showPersistence && (
              <button onClick={() => setShowPersistence(true)}
                className="text-[10px] text-amber-300/80 hover:text-amber-300">
                show {hiddenCount} persistence edge{hiddenCount !== 1 ? 's' : ''}
              </button>
            )}
            {showPersistence && (
              <button onClick={() => setShowPersistence(false)}
                className="text-[10px] text-slate-500 hover:text-slate-700 dark:hover:text-slate-300">
                hide persistence
              </button>
            )}
          </div>
          <table className="w-full">
            <tbody>{edges.map((e, i) => <EdgeRow key={i} edge={e} />)}</tbody>
          </table>
        </div>
      )}

      {(() => {
        const fs = circuit.faithfulness_status;
        const running = fs === 'pending' || fs === 'running';
        // Faithfulness needs at least one member and a producing discovery run
        // for its ablation prompts (the backend 409s otherwise).
        const canRun = circuit.members.length > 0 && circuit.discovery_run_id != null;
        return (
          <div className="border-t border-slate-200 dark:border-slate-700/60 pt-3 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400 flex items-center gap-1">
                <Microscope className="w-3.5 h-3.5 text-violet-300" />
                Faithfulness (rung 3)
              </h4>
              {/* mode toggle — necessity only, or necessity + sufficiency */}
              <div className="inline-flex rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 p-0.5">
                {(['necessity', 'both'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => setFaithMode(m)}
                    disabled={running}
                    className={`px-2 py-0.5 text-[11px] rounded disabled:opacity-50 ${
                      faithMode === m ? 'bg-violet-600 text-white' : 'text-slate-700 dark:text-slate-300 hover:text-white'}`}
                  >
                    {m === 'necessity' ? 'necessity' : 'both'}
                  </button>
                ))}
              </div>
              <button
                onClick={runFaithfulness}
                disabled={faithBusy || running || !canRun}
                className="flex items-center gap-1.5 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-2.5 py-1 text-[11px] text-slate-900 dark:text-slate-100 disabled:opacity-50"
                title={canRun
                  ? 'Ablate the circuit and test necessity/sufficiency of its members'
                  : 'Needs circuit members and a producing discovery run'}
              >
                {running
                  ? <RefreshCw className="w-3 h-3 animate-spin" />
                  : <Microscope className="w-3 h-3" />}
                {running ? 'running…'
                  : circuit.faithfulness ? 'Re-run faithfulness (rung 3)'
                  : 'Run faithfulness (rung 3)'}
              </button>
            </div>

            {!canRun && !circuit.faithfulness && (
              <p className="text-[11px] text-slate-500">
                Faithfulness needs circuit members and a producing discovery run for its
                ablation prompts.
              </p>
            )}

            {faithError && <p className="text-[11px] text-red-300">{faithError}</p>}

            {fs === 'failed' && !running && (
              <p className="text-[11px] text-red-300">Faithfulness run failed.</p>
            )}

            {circuit.faithfulness && (
              <p className="text-xs text-slate-600 dark:text-slate-400">
                Necessity {circuit.faithfulness.necessity?.toFixed(2) ?? '—'}
                {circuit.faithfulness.sufficiency != null &&
                  ` · sufficiency ${circuit.faithfulness.sufficiency.toFixed(2)}`}
                {circuit.faithfulness.sufficiency_k != null &&
                  ` (k=${circuit.faithfulness.sufficiency_k})`}
                {' '}— the circuit is causally faithful to the degree these effects hold.
              </p>
            )}
          </div>
        );
      })()}

      {/* ── Calibration (Feature 20): the usable steering band + dial clamp ── */}
      {(() => {
        const cs = circuit.calibration_status;
        const running = cs === 'pending' || cs === 'running';
        const canRun = circuit.members.length > 0;
        const cal = circuit.calibration;
        const dialFmt = (n: number) => n.toFixed(2);
        return (
          <div className="border-t border-slate-200 dark:border-slate-700/60 pt-3 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <h4 className="text-xs font-medium text-slate-600 dark:text-slate-400 flex items-center gap-1">
                <Gauge className="w-3.5 h-3.5 text-amber-300" />
                Strength calibration
              </h4>
              <button
                onClick={runCalibration}
                disabled={calibBusy || running || !canRun
                  || !judgeEndpoint.trim() || !judgeModel.trim()}
                className="flex items-center gap-1.5 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-2.5 py-1 text-[11px] text-slate-900 dark:text-slate-100 disabled:opacity-50"
                title="Find the usable dial band (onset + correctness cliff) and clamp the served dial to it"
              >
                {running
                  ? <RefreshCw className="w-3 h-3 animate-spin" />
                  : <Gauge className="w-3 h-3" />}
                {running ? 'calibrating…'
                  : cal ? 'Re-calibrate strength' : 'Calibrate strength'}
              </button>
            </div>

            {/* Judge endpoint/model — the correctness judge (OpenAI-compatible). */}
            <div className="flex items-center gap-2 flex-wrap">
              <input
                value={judgeEndpoint}
                onChange={(e) => setJudgeEndpoint(e.target.value)}
                disabled={running}
                placeholder="judge endpoint (…/v1)"
                aria-label="Judge endpoint"
                className="flex-1 min-w-[160px] rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-2 py-1 text-[11px] text-slate-800 dark:text-slate-200 placeholder:text-slate-500 disabled:opacity-50"
              />
              <input
                value={judgeModel}
                onChange={(e) => setJudgeModel(e.target.value)}
                disabled={running}
                placeholder="judge model"
                aria-label="Judge model"
                className="w-40 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 px-2 py-1 text-[11px] text-slate-800 dark:text-slate-200 placeholder:text-slate-500 disabled:opacity-50"
              />
            </div>

            {calibError && <p className="text-[11px] text-red-300">{calibError}</p>}
            {cs === 'failed' && !running && (
              <p className="text-[11px] text-red-300">Calibration run failed.</p>
            )}
            {cs === 'judge_unreliable' && !running && (
              <p className="text-[11px] text-amber-300">
                The judge graded the un-steered output as not correct — it can't
                reliably grade this circuit's model. Use a stronger judge model.
              </p>
            )}
            {cs === 'no_band' && !running && !cal && (
              <p className="text-[11px] text-amber-300">
                No usable band found — no dial above onset stayed correct. The
                circuit's served dial was left unchanged.
              </p>
            )}

            {cal && (
              <div className="space-y-1.5">
                {/* band-on-a-dial readout: onset — sweet — cliff over the range */}
                {(() => {
                  const rng = ((circuit.budget?.intensity_range as number[]) ?? [0, 2]);
                  const lo = rng[0] ?? 0; const hi = rng[1] ?? 2;
                  const span = Math.max(hi - lo, 1e-6);
                  const pct = (v: number) => `${Math.min(100, Math.max(0, ((v - lo) / span) * 100))}%`;
                  return (
                    <div className="relative h-6" title="usable band (shaded) with the sweet-spot dial">
                      <div className="absolute inset-x-0 top-1/2 h-1 -translate-y-1/2 rounded bg-slate-100 dark:bg-slate-700" />
                      <div
                        className="absolute top-1/2 h-1 -translate-y-1/2 rounded bg-amber-500/40"
                        style={{ left: pct(cal.onset), right: `calc(100% - ${pct(cal.cliff)})` }}
                      />
                      <div
                        className="absolute top-1/2 w-1.5 h-3 -translate-y-1/2 -translate-x-1/2 rounded bg-amber-300"
                        style={{ left: pct(cal.sweet_spot) }}
                        title={`sweet-spot ${dialFmt(cal.sweet_spot)}`}
                      />
                    </div>
                  );
                })()}
                <p className="text-xs text-slate-600 dark:text-slate-400 flex items-center gap-2 flex-wrap">
                  <span>onset <b className="text-slate-800 dark:text-slate-200">{dialFmt(cal.onset)}</b></span>
                  <span>· sweet-spot <b className="text-amber-200">{dialFmt(cal.sweet_spot)}</b></span>
                  <span>· cliff <b className="text-slate-800 dark:text-slate-200">{dialFmt(cal.cliff)}</b></span>
                  {cal.provisional && (
                    <span className="rounded bg-amber-900/40 border border-amber-700/50 px-1.5 py-0.5 text-[10px] text-amber-200">
                      provisional
                    </span>
                  )}
                  {cal.non_monotone && (
                    <span className="rounded bg-slate-100 dark:bg-slate-700 px-1.5 py-0.5 text-[10px] text-slate-700 dark:text-slate-300">
                      non-monotone
                    </span>
                  )}
                </p>
                <p className="text-[11px] text-slate-500">
                  The served dial is clamped to [{dialFmt(cal.onset)}, {dialFmt(cal.cliff)}]
                  and defaults to the sweet-spot — it can't reach the nonsense zone above the cliff.
                </p>
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}

function CircuitsListTab({ onNavigateToSteering }: { onNavigateToSteering?: () => void }) {
  const [rows, setRows] = useState<CircuitSummary[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  // Which circuit is being staged into steering (spinner + disable).
  const [steeringId, setSteeringId] = useState<string | null>(null);

  const getSAE = useSAEsStore((s) => s.getSAE);
  const loadCircuitIntoSteering = useSteeringStore((s) => s.loadCircuitIntoSteering);

  // "Steer this circuit": fetch the full circuit (members + saes), resolve its
  // primary SAE object, hydrate the steering selection, then navigate. The
  // list row only has the CircuitSummary, so the members/saes needed to build
  // the selection are fetched here (mirrors CircuitDetail's own get()).
  const steerCircuit = async (id: string) => {
    setSteeringId(id);
    setError(null);
    try {
      const circuit = await circuitsApi.get(id);
      const primaryRef = circuit.saes?.find((s) => s.mistudio_sae_id)?.mistudio_sae_id;
      if (!primaryRef) {
        setError('This circuit has no associated SAE — cannot load it into steering.');
        return;
      }
      const primarySAE = await getSAE(primaryRef);
      // The SAE must be READY, else allocation/generate 422s downstream with a
      // confusing error (R2 F3). Fail fast with a clear message.
      if (primarySAE?.status !== SAEStatus.READY) {
        setError(`This circuit's SAE (${primaryRef}) is not ready `
          + `(${primarySAE?.status ?? 'unknown'}) — download/prepare it first.`);
        return;
      }
      const ok = loadCircuitIntoSteering(circuit, primarySAE);
      if (!ok) {
        setError('This circuit has no loadable feature members to steer.');
        return;
      }
      onNavigateToSteering?.();
    } catch (e) {
      setError(`Failed to load circuit into steering: ${String((e as Error).message ?? e)}`);
    } finally {
      setSteeringId(null);
    }
  };

  const refresh = useCallback(() => {
    setLoading(true);
    circuitsApi.list()
      .then((r) => { setRows(r.circuits); setError(null); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const setPromoted = (id: string, promoted: boolean) =>
    circuitsApi.setPromoted(id, promoted)
      .then((updated) =>
        setRows((rs) => rs.map((r) => (r.id === id ? { ...r, promoted: updated.promoted } : r))))
      .catch((e) => setError(String(e.message ?? e)));

  const remove = (id: string) => {
    if (window.confirm('Delete this circuit? Its validation manifests survive.')) {
      circuitsApi.remove(id).then(refresh).catch((e) => setError(String(e.message ?? e)));
    }
  };

  const downloadSlices = (id: string, name: string) =>
    circuitsApi.exportSlices(id).then((r) => {
      // Save the full response — the parent_rung wrapper travels in the file
      // so the human-readable evidence context isn't dropped (R3-B9).
      const blob = new Blob([JSON.stringify(r, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const safe = name.toLowerCase().replace(/[^a-z0-9-_]+/g, '-').replace(/^-+|-+$/g, '') || 'circuit';
      a.download = `${safe}.slices.json`;
      // MIS-E2E-131(4): the anchor has to be IN the document. Firefox ignores
      // a click on a detached element, so this button was a silent no-op
      // there. The other three export sites in this repo all append first;
      // this was the only one that did not.
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      // Revoking on the very next line can also race the download in some
      // browsers — give the click a turn of the event loop first.
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }).catch((e) => setError(String(e.message ?? e)));

  const fileInput = useRef<HTMLInputElement>(null);
  const onImportFile = (file: File) => {
    file.text()
      .then((text) => circuitsApi.importDefinition(JSON.parse(text)))
      .then(refresh)
      .catch((e) => setError(`Import failed: ${String(e.message ?? e)}`));
  };

  return (
    <div className="space-y-4">
      <div className="flex items-start justify-between gap-4">
        <p className="text-slate-600 dark:text-slate-400 text-sm">
          Cross-layer feature circuits with graded evidence. Every artifact shows its rung —
          discovery results are Tier-1 (same-token) associations until validated.
        </p>
        <input
          ref={fileInput}
          type="file"
          accept=".json,application/json"
          className="hidden"
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) onImportFile(f);
            e.target.value = '';
          }}
        />
        <button
          onClick={() => fileInput.current?.click()}
          className="flex items-center gap-1.5 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-3 py-1.5 text-xs text-slate-800 dark:text-slate-200 shrink-0"
          title="Import a mistudio.circuit-definition/v1 file"
        >
          <Upload className="w-3.5 h-3.5" /> Import
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-slate-500 text-sm">Loading…</p>
      ) : rows.length === 0 ? (
        <div className={`${COMPONENTS.card.base} p-8 text-center`}>
          <Layers className="w-10 h-10 text-slate-600 mx-auto mb-3" />
          <h3 className="text-slate-700 dark:text-slate-300 font-medium mb-1">No circuits yet</h3>
          <p className="text-slate-500 text-sm">
            Circuits arrive from discovery runs (coming with the mining feature), via the
            MCP <span className="font-mono">create_circuit</span> /{' '}
            <span className="font-mono">import_circuit_definition</span> tools, or by
            importing a <span className="font-mono">.circuit.json</span> file.
          </p>
        </div>
      ) : (
        rows.map((c) => (
          <div key={c.id} className={`${COMPONENTS.card.base} p-4`}>
            <div className="flex items-center gap-3">
              <button
                className="text-left flex-1 min-w-0"
                onClick={() => setExpanded(expanded === c.id ? null : c.id)}
              >
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-slate-900 dark:text-slate-100 font-medium truncate">{c.name}</span>
                  <RungChip rung={c.rung} language={c.rung_language} nextStep={c.rung_next_step} />
                  {c.promoted && (
                    <span className="rounded bg-emerald-500/10 text-emerald-300 px-1.5 py-0.5 text-[10px]">
                      promoted
                    </span>
                  )}
                </div>
                <div className="text-xs text-slate-500 mt-0.5">
                  {c.member_count} members · {c.edge_count} edges ·{' '}
                  {c.layers.map((l) => `L${l}`).join('+')}
                </div>
              </button>
              {c.promoted && (
                <button
                  onClick={() => steerCircuit(c.id)}
                  disabled={steeringId === c.id}
                  className="flex items-center gap-1 rounded bg-emerald-600 hover:bg-emerald-500 disabled:opacity-60 disabled:cursor-wait px-2.5 py-1.5 text-xs text-white"
                  title="Load this circuit's members into the Steering panel (per-layer SAEs + validated hazards)"
                >
                  <Wand2 className="w-3.5 h-3.5" />
                  {steeringId === c.id ? 'Loading…' : 'Steer this circuit'}
                </button>
              )}
              {c.promoted ? (
                <button
                  onClick={() => setPromoted(c.id, false)}
                  className="flex items-center gap-1 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-2.5 py-1.5 text-xs text-slate-800 dark:text-slate-200"
                  title="Unpromote (removes the loadable-profile badge)"
                >
                  <ArrowDownRight className="w-3.5 h-3.5" /> Unpromote
                </button>
              ) : (
                <button
                  onClick={() => setPromoted(c.id, true)}
                  className="flex items-center gap-1 rounded bg-violet-600 hover:bg-violet-500 px-2.5 py-1.5 text-xs text-white"
                  title="Promote to a loadable multi-layer steering profile (badge, not gate)"
                >
                  <ArrowUpRight className="w-3.5 h-3.5" /> Promote
                </button>
              )}
              <a
                href={circuitsApi.exportUrl(c.id)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400"
                title="Export circuit-definition/v1"
              >
                <Download className="w-4 h-4" />
              </a>
              <button
                onClick={() => downloadSlices(c.id, c.name)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400"
                title="Export per-layer v1 slices (partial renderings for single-SAE consumers)"
              >
                <FileDown className="w-4 h-4" />
              </button>
              <button
                onClick={() => remove(c.id)}
                className="p-1.5 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400"
                title="Delete circuit"
              >
                <Trash2 className="w-4 h-4" />
              </button>
            </div>

            {expanded === c.id && <CircuitDetail id={c.id} onChanged={refresh} />}
          </div>
        ))
      )}
    </div>
  );
}

// ── Capture tab (Feature 016) ─────────────────────────────────────────────

interface LayerRow { layer: string; sae_id: string; }

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return '—';
  const mb = n / (1024 * 1024);
  return mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;
}

function CaptureTab() {
  const datasets = useDatasetsStore((s) => s.datasets);
  const fetchDatasets = useDatasetsStore((s) => s.fetchDatasets);
  const saes = useSAEsStore((s) => s.saes);
  const fetchSAEs = useSAEsStore((s) => s.fetchSAEs);

  const [datasetId, setDatasetId] = useState('');
  const [layerRows, setLayerRows] = useState<LayerRow[]>([{ layer: '', sae_id: '' }]);
  const [epsilon, setEpsilon] = useState('0.1');
  const [sampleCap, setSampleCap] = useState('2000');
  const [attnEnabled, setAttnEnabled] = useState(false);
  const [attnLayers, setAttnLayers] = useState('');
  const [attnTopK, setAttnTopK] = useState('4');

  const [estimateFor, setEstimateFor] = useState<CircuitCapture | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [captures, setCaptures] = useState<CircuitCapture[]>([]);
  // Guards the recursive estimate poll against firing after unmount (R1 CR#9).
  //
  // MIS-E2E-131(1): the cleanup set this false and NOTHING set it back. React
  // StrictMode — on in `main.tsx` — mounts, unmounts and remounts every
  // component in development, so by the time the real mount happened the flag
  // was already false and the estimate poll was dead on arrival. The "Cost
  // estimate" panel simply never appeared while developing, which is the worst
  // possible place for a feature to be invisible.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (datasets.length === 0) fetchDatasets();
    if (saes.length === 0) fetchSAEs();
  }, [datasets.length, saes.length, fetchDatasets, fetchSAEs]);

  const refreshCaptures = useCallback(() => {
    circuitsApi.listCaptures().then((r) => setCaptures(r.captures)).catch(() => {});
  }, []);
  useEffect(() => { refreshCaptures(); }, [refreshCaptures]);

  // Light poll while any capture is active.
  useEffect(() => {
    if (!captures.some((c) => isActive(c.status))) return;
    const t = setInterval(refreshCaptures, 2500);
    return () => clearInterval(t);
  }, [captures, refreshCaptures]);

  const readySaes = saes.filter((s) => s.status === SAEStatus.READY);

  const buildBody = (confirm: boolean): CircuitCaptureCreate | null => {
    const layers = layerRows
      .filter((r) => r.layer !== '' && r.sae_id !== '')
      .map((r) => ({ layer: parseInt(r.layer, 10), sae_id: r.sae_id }));
    if (!datasetId || layers.length === 0) {
      setError('Pick a dataset and at least one layer + SAE.');
      return null;
    }
    const body: CircuitCaptureCreate = {
      dataset_id: datasetId,
      layers,
      epsilon: parseFloat(epsilon) || 0.1,
      sample_cap: parseInt(sampleCap, 10) || 2000,
      confirm,
    };
    if (attnEnabled) {
      const al = attnLayers.split(',').map((s) => parseInt(s.trim(), 10)).filter((n) => !isNaN(n));
      body.attention_capture = { layers: al, top_k: parseInt(attnTopK, 10) || 4 };
    }
    return body;
  };

  const runEstimate = () => {
    setError(null);
    const body = buildBody(false);
    if (!body) return;
    setSubmitting(true);
    circuitsApi.createCapture(body)
      .then((r) => pollCapture(r.id))
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setSubmitting(false));
  };

  // After an estimate POST, poll the run until it reaches 'estimated' to show the card.
  const pollCapture = (id: string) => {
    const tick = () => {
      if (!mountedRef.current) return;
      circuitsApi.getCapture(id).then((run) => {
        if (!mountedRef.current) return;
        setEstimateFor(run);
        refreshCaptures();
        if (isActive(run.status)) setTimeout(tick, 2000);
      }).catch((e) => { if (mountedRef.current) setError(String(e.message ?? e)); });
    };
    tick();
  };

  const confirmRun = (id: string) => {
    setError(null);
    circuitsApi.confirmCapture(id)
      .then(() => { setEstimateFor(null); refreshCaptures(); })
      .catch((e) => setError(String(e.message ?? e)));
  };

  const cancelRun = (id: string) =>
    circuitsApi.cancelCapture(id).then(refreshCaptures).catch((e) => setError(String(e.message ?? e)));

  const deleteRun = (id: string) => {
    if (window.confirm('Delete this capture run and its activation store?')) {
      circuitsApi.deleteCapture(id).then(refreshCaptures).catch((e) => setError(String(e.message ?? e)));
    }
  };

  return (
    <div className="space-y-4">
      <p className="text-slate-600 dark:text-slate-400 text-sm">
        Capture co-activation events across layers on a held-out split. Run an estimate first,
        then confirm to launch the full pass.
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className={`${COMPONENTS.card.base} p-4 space-y-3`}>
        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Dataset</label>
          <select
            className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
            value={datasetId}
            onChange={(e) => setDatasetId(e.target.value)}
          >
            <option value="">Select a dataset…</option>
            {datasets.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
          </select>
        </div>

        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Layers &amp; SAEs</label>
          <div className="space-y-2">
            {layerRows.map((row, i) => {
              // The SAE knows its own layer (trained one-layer-per-SAE), so
              // derive + lock the layer box from the selected SAE — no
              // re-typing the same fact (and no chance to contradict it). Stays
              // editable only for an imported SAE that has no recorded layer.
              const selectedSae = readySaes.find((s) => s.id === row.sae_id);
              const saeLayer = selectedSae?.layer;
              const layerLocked = saeLayer != null;
              const shownLayer = layerLocked ? String(saeLayer) : row.layer;
              return (
              <div key={i} className="flex items-center gap-2">
                <input
                  type="number"
                  placeholder="layer"
                  className={`w-20 rounded border px-2 py-1.5 text-sm ${
                    layerLocked
                      ? 'bg-slate-100 border-slate-300 text-slate-600 dark:bg-slate-800/50 dark:border-slate-700 dark:text-slate-400 cursor-not-allowed'
                      : 'bg-white border-slate-300 text-slate-900 dark:bg-slate-800 dark:border-slate-600 dark:text-slate-100'}`}
                  value={shownLayer}
                  readOnly={layerLocked}
                  title={layerLocked
                    ? `Layer ${saeLayer} — set by the selected SAE (trained on this layer)`
                    : 'This SAE has no recorded layer — enter it'}
                  onChange={(e) => setLayerRows((rs) => rs.map((r, j) => j === i ? { ...r, layer: e.target.value } : r))}
                />
                <select
                  className="flex-1 rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
                  value={row.sae_id}
                  onChange={(e) => {
                    const sae = readySaes.find((s) => s.id === e.target.value);
                    setLayerRows((rs) => rs.map((r, j) => j === i
                      // adopt the SAE's own layer on select (locked box)
                      ? { ...r, sae_id: e.target.value,
                          layer: sae?.layer != null ? String(sae.layer) : r.layer }
                      : r));
                  }}
                >
                  <option value="">Select an SAE…</option>
                  {readySaes.map((s) => (
                    <option key={s.id} value={s.id}>
                      {s.name}{s.layer != null ? ` (L${s.layer})` : ''}
                    </option>
                  ))}
                </select>
                <button
                  onClick={() => setLayerRows((rs) => rs.length > 1 ? rs.filter((_, j) => j !== i) : rs)}
                  className="p-1.5 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400 disabled:opacity-30"
                  disabled={layerRows.length <= 1}
                  title="Remove layer"
                >
                  <XIcon className="w-3.5 h-3.5" />
                </button>
              </div>
              );
            })}
          </div>
          {layerRows.length < 8 && (
            <button
              onClick={() => setLayerRows((rs) => [...rs, { layer: '', sae_id: '' }])}
              className="mt-2 flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300"
            >
              <Plus className="w-3.5 h-3.5" /> Add layer
            </button>
          )}
        </div>

        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Epsilon</label>
            <input
              type="number" step="0.01"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={epsilon}
              onChange={(e) => setEpsilon(e.target.value)}
            />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Sample cap</label>
            <input
              type="number"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={sampleCap}
              onChange={(e) => setSampleCap(e.target.value)}
            />
          </div>
        </div>

        <div>
          <label className="flex items-center gap-2 text-xs text-slate-700 dark:text-slate-300">
            <input type="checkbox" checked={attnEnabled} onChange={(e) => setAttnEnabled(e.target.checked)} />
            Capture attention (top-k per head)
          </label>
          {attnEnabled && (
            <div className="flex gap-3 mt-2">
              <div className="flex-1">
                <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Attention layers (comma-sep)</label>
                <input
                  className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
                  placeholder="e.g. 4, 6, 8"
                  value={attnLayers}
                  onChange={(e) => setAttnLayers(e.target.value)}
                />
              </div>
              <div className="w-24">
                <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">top_k</label>
                <input
                  type="number"
                  className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
                  value={attnTopK}
                  onChange={(e) => setAttnTopK(e.target.value)}
                />
              </div>
            </div>
          )}
        </div>

        <button
          onClick={runEstimate}
          disabled={submitting}
          className="flex items-center gap-1.5 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-3 py-1.5 text-sm text-slate-900 dark:text-slate-100 disabled:opacity-50"
        >
          <Search className="w-4 h-4" /> Estimate
        </button>

        {estimateFor && estimateFor.status === 'estimated' && estimateFor.estimate && (
          <div className="rounded-lg border border-sky-500/30 bg-sky-500/10 p-3 space-y-2">
            <h4 className="text-xs font-medium text-sky-200">Cost estimate</h4>
            <div className="text-xs text-slate-700 dark:text-slate-300 flex flex-wrap gap-x-4 gap-y-1">
              <span>{estimateFor.estimate.events ?? '—'} events</span>
              <span>{fmtBytes(estimateFor.estimate.bytes)}</span>
              <span>{estimateFor.estimate.minutes != null ? `~${estimateFor.estimate.minutes} min` : '—'}</span>
            </div>
            {estimateFor.stale && (
              <p className="text-[11px] text-amber-300">
                A store for this configuration already exists but is stale.
              </p>
            )}
            <button
              onClick={() => confirmRun(estimateFor.id)}
              className="flex items-center gap-1.5 rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-1.5 text-sm text-white"
            >
              <Camera className="w-4 h-4" /> Run capture
            </button>
          </div>
        )}
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Capture runs</h3>
        {captures.length === 0 ? (
          <p className="text-xs text-slate-500">No capture runs yet.</p>
        ) : captures.map((c) => (
          <div key={c.id} className={`${COMPONENTS.card.base} p-3 space-y-2`}>
            <div className="flex items-center gap-2">
              <StatusBadge status={c.status} />
              <span className="text-xs font-mono text-slate-500">{c.id.slice(0, 8)}</span>
              {c.stale && (
                <span className="flex items-center gap-1 rounded bg-amber-500/10 text-amber-300 px-1.5 py-0.5 text-[10px]">
                  <AlertTriangle className="w-3 h-3" /> stale
                </span>
              )}
              <div className="flex-1" />
              {isActive(c.status) && (
                <button onClick={() => cancelRun(c.id)}
                  className="text-[11px] text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200">cancel</button>
              )}
              <button onClick={() => deleteRun(c.id)}
                className="p-1 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400" title="Delete">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
            {isActive(c.status) && <ProgressBar value={c.progress} />}
            <div className="text-xs text-slate-500 flex flex-wrap gap-x-3 gap-y-0.5">
              {c.layers && <span>{c.layers.map((l) => `L${l.layer}`).join('+')}</span>}
              <span>{fmtBytes(c.bytes)}</span>
              {c.events_total != null && <span>{c.events_total} events</span>}
              {c.split?.heldout_count != null && <span>held-out: {c.split.heldout_count}</span>}
              {c.error_message && <span className="text-red-300">{c.error_message}</span>}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Discovery tab (Feature 016) ───────────────────────────────────────────

function nodeLabel(n: DiscoveryCandidate['up']): string {
  if (n.cluster_name) return n.cluster_name;
  if (n.cluster_profile_id) return `L${n.layer}:${n.cluster_profile_id.slice(0, 8)}`;
  return `L${n.layer}:f${n.feature_idx}`;
}

export function RunReportCard({ report }: { report: DiscoveryReport }) {
  const caps = report.caps;
  const capWarn = caps?.candidates_truncated
    || (caps?.unit_cap_hit_layers?.length ?? 0) > 0
    || caps?.null_cap_hit;
  return (
    <div className={`${COMPONENTS.card.base} p-4 space-y-3`}>
      <div className="flex items-baseline gap-3">
        <div>
          <div className="text-2xl font-semibold text-emerald-300">
            {report.replication.rate == null
              ? 'n/a'
              : `${(report.replication.rate * 100).toFixed(0)}%`}
          </div>
          <div className="text-[11px] text-slate-500">
            {report.replication.rate == null
              ? 'replication rate — no held-out candidates tested'
              : `replication rate (${report.replication.replicated}/${report.replication.tested} held-out)`}
          </div>
        </div>
        <div className="flex-1" />
        <span className="rounded bg-slate-100 text-slate-700 dark:bg-slate-700/60 dark:text-slate-300 px-2 py-0.5 text-[10px]">
          {report.granularity} · {report.mode}
        </span>
      </div>

      <div className="text-xs text-slate-600 dark:text-slate-400 grid grid-cols-2 gap-x-4 gap-y-1">
        <span>null: {report.null_summary.method} ({report.null_summary.shuffles}× · p{report.null_summary.percentile})</span>
        <span>
          FDR: {report.fdr.discipline} · q={report.fdr.q} · {report.fdr.passed}/{report.fdr.tested} passed
          {report.fdr.p_resolution != null && (
            <span className="text-slate-500" title="Finest achievable p-value (pooled-standardized null); FDR can only pass edges above this floor.">
              {' '}· p-res {report.fdr.p_resolution.toExponential(1)}
            </span>
          )}
        </span>
      </div>

      <div>
        <h4 className="text-[11px] font-medium text-slate-600 dark:text-slate-400 mb-1">Stage counts</h4>
        <div className="text-xs text-slate-700 dark:text-slate-300 flex flex-wrap gap-x-3 gap-y-0.5">
          <span>considered {report.counts_by_stage.pairs_considered}</span>
          <span>→ support {report.counts_by_stage.post_support}</span>
          <span>→ null-tested {report.counts_by_stage.null_tested}</span>
          <span>→ post-FDR {report.counts_by_stage.post_fdr}</span>
          <span>→ persisted {report.counts_by_stage.candidates_persisted}</span>
        </div>
      </div>

      {capWarn && (
        <div className="rounded border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-[11px] text-amber-300 space-y-0.5">
          <div className="flex items-center gap-1 font-medium">
            <AlertTriangle className="w-3.5 h-3.5" /> Caps hit — results may be truncated
          </div>
          {caps.candidates_truncated && <div>Candidate list was truncated.</div>}
          {(caps.unit_cap_hit_layers?.length ?? 0) > 0 && (
            <div>Per-unit cap hit on layers: {caps.unit_cap_hit_layers.join(', ')}.</div>
          )}
          {caps.null_cap_hit && <div>Null-shuffle cap hit.</div>}
        </div>
      )}

      {report.uncovered_seeds?.length > 0 && (
        <div className="text-[11px] text-slate-600 dark:text-slate-400">
          <span className="font-medium text-slate-700 dark:text-slate-300">Uncovered seeds:</span>{' '}
          {report.uncovered_seeds.map((u, i) => (
            <span key={i} className="mr-2">L{u.layer} ({u.reason})</span>
          ))}
        </div>
      )}

      <div className="rounded border border-slate-300 dark:border-slate-700/60 bg-slate-100 dark:bg-slate-900/40 px-3 py-2 text-[11px] text-slate-600 dark:text-slate-400 italic">
        {report.lag0_disclosure}
      </div>
    </div>
  );
}

function DiscoveryRunDetail({ runId, onChanged }: { runId: string; onChanged: () => void }) {
  const [run, setRun] = useState<DiscoveryRun | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [attrBusy, setAttrBusy] = useState(false);

  const load = useCallback(() => {
    circuitsApi.getDiscovery(runId, true)
      .then(setRun)
      .catch((e) => setError(String(e.message ?? e)));
  }, [runId]);

  useEffect(() => { load(); }, [load]);

  // Light poll while the run OR its attribution pass is active (R2 B3 —
  // discovery status stays 'completed' during attribution, so we must also
  // watch attribution_status or the pass's progress never refreshes).
  useEffect(() => {
    if (!run) return;
    const active = isActive(run.status)
      || (run.attribution_status != null && isActive(run.attribution_status));
    if (!active) return;
    const t = setInterval(load, 2500);
    return () => clearInterval(t);
  }, [run, load]);

  const startAttribution = () => {
    if (!run) return;
    setAttrBusy(true);
    circuitsApi.startAttribution(run.id)
      .then(() => { load(); onChanged(); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setAttrBusy(false));
  };

  const cancelAttribution = () => {
    if (!run) return;
    circuitsApi.cancelAttribution(run.id)
      .then(() => load())
      .catch((e) => setError(String(e.message ?? e)));
  };

  const attrActive = run?.attribution_status != null
    && ['pending', 'running'].includes(run.attribution_status);

  if (error && !run) return <p className="text-xs text-red-300 mt-2">{error}</p>;
  if (!run) return <p className="text-xs text-slate-500 mt-2">Loading report…</p>;

  // After attribution, re-order by attr_rank so the re-ranking is VISIBLE
  // (US-5); before, keep the persisted co-activation order.
  const attributed = run.attribution_status === 'completed';
  const candidates = (run.candidates ?? []).slice().sort((a, b) => {
    if (!attributed) return (a.orderings?.coact_rank ?? 0) - (b.orderings?.coact_rank ?? 0);
    const ar = a.orderings?.attr_rank, br = b.orderings?.attr_rank;
    if (ar == null && br == null) return 0;
    if (ar == null) return 1;  // unattributed (e.g. deleted profile) sort last
    if (br == null) return -1;
    return ar - br;
  });

  return (
    <div className="mt-3 border-t border-slate-200 dark:border-slate-700/60 pt-3 space-y-3">
      {error && <p className="text-xs text-red-300">{error}</p>}
      {isActive(run.status) && <ProgressBar value={run.progress} />}
      {run.report && <RunReportCard report={run.report} />}

      {run.status === 'completed' && (
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={startAttribution}
            disabled={attrBusy || attrActive}
            className="flex items-center gap-1.5 rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-3 py-1.5 text-xs text-slate-900 dark:text-slate-100 disabled:opacity-50"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            {run.attribution_status === 'completed' ? 'Re-run attribution pass'
              : 'Run attribution pass'}
          </button>
          {/* Attribution's own lifecycle, distinct from the discovery status (R2 B3/B4) */}
          {attrActive && (
            <span className="flex items-center gap-2 text-[11px] text-violet-300">
              attribution {run.attribution_status}
              {run.attribution_progress != null
                && ` · ${Math.round(run.attribution_progress)}%`}
              <button onClick={cancelAttribution}
                className="rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-2 py-0.5 text-slate-800 dark:text-slate-200">
                cancel
              </button>
            </span>
          )}
          {run.attribution_status === 'completed' && (
            <span className="text-[11px] text-emerald-300">attribution complete</span>
          )}
          {run.attribution_status === 'failed' && (
            <span className="text-[11px] text-red-300"
              title={run.attribution_error ?? undefined}>
              attribution failed{run.attribution_error ? ` — ${run.attribution_error}` : ''}
            </span>
          )}
          {run.attribution_status === 'cancelled' && (
            <span className="text-[11px] text-slate-600 dark:text-slate-400">attribution cancelled</span>
          )}
        </div>
      )}

      {candidates.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-slate-500 text-left">
                <th className="py-1 pr-3 font-medium">candidate</th>
                <th className="py-1 pr-3 font-medium">PMI</th>
                <th className="py-1 pr-3 font-medium">support</th>
                <th className="py-1 pr-3 font-medium">null %</th>
                <th className="py-1 pr-3 font-medium">repl.</th>
                <th className="py-1 pr-3 font-medium">attribution</th>
                {attributed && <th className="py-1 pr-3 font-medium" title="co-activation rank → attribution rank">rank Δ</th>}
              </tr>
            </thead>
            <tbody>
              {candidates.map((c, i) => (
                <tr key={i} className="text-slate-700 dark:text-slate-300 border-t border-slate-200 dark:border-slate-800">
                  <td className="py-1 pr-3 font-mono whitespace-nowrap">
                    {nodeLabel(c.up)} → {nodeLabel(c.down)}
                  </td>
                  <td className="py-1 pr-3">{c.stats.pmi != null ? c.stats.pmi.toFixed(2) : '—'}</td>
                  <td className="py-1 pr-3">{c.stats.support ?? '—'}</td>
                  <td className="py-1 pr-3">{c.stats.null_pct != null ? c.stats.null_pct.toFixed(1) : '—'}</td>
                  <td className="py-1 pr-3">{c.replicated_heldout ? '✓' : '—'}</td>
                  <td className="py-1 pr-3">
                    {c.attribution?.score != null ? (
                      <span className="flex items-center gap-1">
                        {c.attribution.score.toFixed(2)}
                        {c.attribution.rung1_gate && (
                          <span className="rounded bg-emerald-500/10 text-emerald-300 px-1 py-0.5 text-[9px]">
                            rung 1
                          </span>
                        )}
                      </span>
                    ) : '—'}
                  </td>
                  {attributed && (
                    <td className="py-1 pr-3 font-mono text-slate-600 dark:text-slate-400">
                      {c.orderings?.coact_rank != null && c.orderings?.attr_rank != null
                        ? `${c.orderings.coact_rank + 1}→${c.orderings.attr_rank + 1}`
                        : '—'}
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : run.status === 'completed' ? (
        <p className="text-xs text-slate-500">No candidates passed the gates.</p>
      ) : null}
    </div>
  );
}

function DiscoveryTab() {
  const [captures, setCaptures] = useState<CircuitCapture[]>([]);
  const [captureId, setCaptureId] = useState('');
  const [force, setForce] = useState(false);
  const [mode, setMode] = useState<DiscoveryMode>('open');
  const [granularity, setGranularity] = useState<DiscoveryGranularity>('feature');
  const [seedRefsText, setSeedRefsText] = useState('');
  const [sMin, setSMin] = useState('20');
  const [nullShuffles, setNullShuffles] = useState('100');
  const [fdrQ, setFdrQ] = useState('0.05');

  const [runs, setRuns] = useState<DiscoveryRun[]>([]);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    circuitsApi.listCaptures().then((r) => setCaptures(r.captures)).catch(() => {});
  }, []);

  const refreshRuns = useCallback(() => {
    circuitsApi.listDiscoveries().then((r) => setRuns(r.discoveries)).catch(() => {});
  }, []);
  useEffect(() => { refreshRuns(); }, [refreshRuns]);

  useEffect(() => {
    if (!runs.some((r) => isActive(r.status))) return;
    const t = setInterval(refreshRuns, 2500);
    return () => clearInterval(t);
  }, [runs, refreshRuns]);

  // Default granularity Cluster when seeded, Feature when open — but only when
  // the user hasn't diverged mid-session; we flip on mode change.
  const onModeChange = (m: DiscoveryMode) => {
    setMode(m);
    setGranularity(m === 'seeded' ? 'cluster' : 'feature');
  };

  // Eligible captures: completed and non-stale (or force overrides staleness).
  const eligible = captures.filter(
    (c) => c.status === 'completed' && (!c.stale || force));

  const parseSeedRefs = (): DiscoverySeedRef[] => {
    // One ref per line: "layer:feature_idx" or "layer:cluster:<profile_id>".
    return seedRefsText.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => {
      const parts = line.split(':');
      const layer = parseInt(parts[0], 10);
      if (parts[1] === 'cluster' && parts[2]) {
        return { layer, cluster_profile_id: parts[2] };
      }
      return { layer, feature_idx: parseInt(parts[1], 10) };
    }).filter((r) => {
      // MIS-E2E-131(3): only `layer` was validated. A line like "6" — no
      // colon — produced `feature_idx: NaN`, which `JSON.stringify` writes as
      // `null`, so the POST body carried a silently malformed seed instead of
      // the user being told their input was incomplete.
      if (isNaN(r.layer)) return false;
      if ('cluster_profile_id' in r) return true;
      return !isNaN((r as { feature_idx: number }).feature_idx);
    });
  };

  const runDiscovery = () => {
    setError(null);
    if (!captureId) { setError('Select a capture run.'); return; }
    const body: DiscoveryCreate = {
      capture_run_id: captureId,
      granularity,
      mode,
      s_min: parseInt(sMin, 10) || 20,
      null_shuffles: parseInt(nullShuffles, 10) || 100,
      fdr_q: parseFloat(fdrQ) || 0.05,
      force,
    };
    if (mode === 'seeded') {
      const refs = parseSeedRefs();
      if (refs.length === 0) { setError('Seeded mode needs at least one seed ref.'); return; }
      body.seed_refs = refs;
    }
    setSubmitting(true);
    circuitsApi.createDiscovery(body)
      .then((r) => { refreshRuns(); setExpanded(r.id); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setSubmitting(false));
  };

  const deleteRun = (id: string) => {
    if (window.confirm('Delete this discovery run?')) {
      circuitsApi.deleteDiscovery(id).then(refreshRuns).catch((e) => setError(String(e.message ?? e)));
    }
  };

  const cancelRun = (id: string) =>
    circuitsApi.cancelDiscovery(id).then(refreshRuns).catch((e) => setError(String(e.message ?? e)));

  const Toggle = <T extends string>(
    { value, options, onChange }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void },
  ) => (
    <div className="inline-flex rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 p-0.5">
      {options.map((o) => (
        <button
          key={o.v}
          onClick={() => onChange(o.v)}
          className={`px-2.5 py-1 text-xs rounded ${
            value === o.v ? 'bg-violet-600 text-white' : 'text-slate-700 dark:text-slate-300 hover:text-white'}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );

  return (
    <div className="space-y-4">
      <p className="text-slate-600 dark:text-slate-400 text-sm">
        Mine candidate associations from a capture. Discovery produces graded evidence
        (rungs 0–1) — candidates and attribution support, never proof.
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className={`${COMPONENTS.card.base} p-4 space-y-3`}>
        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Capture run</label>
          <select
            className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
            value={captureId}
            onChange={(e) => setCaptureId(e.target.value)}
          >
            <option value="">Select a completed capture…</option>
            {eligible.map((c) => (
              <option key={c.id} value={c.id}>
                {c.id.slice(0, 8)} · {c.layers?.map((l) => `L${l.layer}`).join('+') ?? ''}
                {c.stale ? ' (stale)' : ''}
              </option>
            ))}
          </select>
          <label className="flex items-center gap-2 text-[11px] text-slate-600 dark:text-slate-400 mt-1.5">
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => {
                const next = e.target.checked;
                setForce(next);
                // MIS-E2E-131(2): un-ticking Force narrows `eligible` to
                // non-stale captures, but the selected id stayed put. The
                // select then showed NO selection while "Run discovery"
                // submitted the hidden stale id — and the server's refusal
                // named a capture the user could not see anywhere on screen.
                if (!next && captureId
                    && !captures.some((c) => c.id === captureId
                                             && c.status === 'completed' && !c.stale)) {
                  setCaptureId('');
                }
              }}
            />
            Force — allow mining a stale capture store
          </label>
        </div>

        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Mode</label>
            <Toggle
              value={mode}
              onChange={onModeChange}
              options={[{ v: 'open', label: 'Open' }, { v: 'seeded', label: 'Seeded' }]}
            />
          </div>
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Granularity</label>
            <Toggle
              value={granularity}
              onChange={setGranularity}
              options={[{ v: 'feature', label: 'Feature' }, { v: 'cluster', label: 'Cluster' }]}
            />
          </div>
        </div>

        {mode === 'seeded' && (
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">
              Seed refs (one per line: <span className="font-mono">layer:feature_idx</span> or{' '}
              <span className="font-mono">layer:cluster:profile_id</span>)
            </label>
            <textarea
              rows={3}
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100 font-mono"
              placeholder={'6:1234\n8:cluster:abc123'}
              value={seedRefsText}
              onChange={(e) => setSeedRefsText(e.target.value)}
            />
          </div>
        )}

        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">s_min</label>
            <input type="number"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={sMin} onChange={(e) => setSMin(e.target.value)} />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">null shuffles</label>
            <input type="number"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={nullShuffles} onChange={(e) => setNullShuffles(e.target.value)} />
          </div>
          <div className="flex-1">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">FDR q</label>
            <input type="number" step="0.01"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={fdrQ} onChange={(e) => setFdrQ(e.target.value)} />
          </div>
        </div>

        <button
          onClick={runDiscovery}
          disabled={submitting}
          className="flex items-center gap-1.5 rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-1.5 text-sm text-white disabled:opacity-50"
        >
          <Search className="w-4 h-4" /> Run discovery
        </button>
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Discovery runs</h3>
        {runs.length === 0 ? (
          <p className="text-xs text-slate-500">No discovery runs yet.</p>
        ) : runs.map((r) => (
          <div key={r.id} className={`${COMPONENTS.card.base} p-3`}>
            <div className="flex items-center gap-2">
              <button
                className="text-left flex-1 min-w-0 flex items-center gap-2"
                onClick={() => setExpanded(expanded === r.id ? null : r.id)}
              >
                <StatusBadge status={r.status} />
                <span className="text-xs font-mono text-slate-500">{r.id.slice(0, 8)}</span>
                <span className="text-xs text-slate-600 dark:text-slate-400">{r.candidate_count} candidates</span>
              </button>
              {isActive(r.status) && (
                <button onClick={() => cancelRun(r.id)}
                  className="text-[11px] text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200">cancel</button>
              )}
              <button onClick={() => deleteRun(r.id)}
                className="p-1 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400" title="Delete">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </div>
            {expanded === r.id && <DiscoveryRunDetail runId={r.id} onChanged={refreshRuns} />}
          </div>
        ))}
      </div>
    </div>
  );
}

// ── Validation tab (Feature 017) ──────────────────────────────────────────

const VALIDATION_ACTIVE = new Set(['pending', 'running']);
const isValidationActive = (s: string | null | undefined) =>
  s != null && VALIDATION_ACTIVE.has(s);

/** Per-edge results table for a validation pass. Rows come from the run's own
 *  candidates that carry a `validation` write-back. Copy discipline: a PASS is
 *  "causally validated (rung 2)"; a fail is "tested, did not validate". */
export function ValidationResults({
  run,
  onOpenManifest,
}: {
  run: DiscoveryRun;
  onOpenManifest: (id: string) => void;
}) {
  const validated = (run.candidates ?? []).filter((c) => c.validation != null);
  if (validated.length === 0) {
    return (
      <p className="text-xs text-slate-500">
        No per-edge results yet. Run a validation pass to test the top-K edges.
      </p>
    );
  }
  const batch = run.report?.validation ?? null;
  const byOrd = batch?.by_ordering ?? null;
  const coact = byOrd?.coact ?? null;
  const attr = byOrd?.attr ?? null;
  const bothOrderings = coact != null && attr != null;
  const pct = (v: number | null | undefined) =>
    v != null ? `${(v * 100).toFixed(0)}%` : '—';
  return (
    <div className="space-y-2">
      {batch && (
        <div className="rounded border border-emerald-500/30 bg-emerald-500/10 px-3 py-2 text-[11px] text-emerald-300">
          <span className="font-medium">Batch ({batch.ordering}):</span>{' '}
          {batch.passed}/{batch.k} edges causally validated (rung 2)
          {batch.survival != null && ` · ${(batch.survival * 100).toFixed(0)}% survival`}
          {batch.wall_clock_seconds != null && ` · ${batch.wall_clock_seconds}s`}
        </div>
      )}

      {/* Both-orderings uplift (US-1): attribution re-ranking's causal payoff,
          made legible — the survival rate for each ordering plus the delta. */}
      {bothOrderings && (() => {
        const uplift = batch?.uplift
          ?? ((attr!.survival != null && coact!.survival != null)
            ? attr!.survival - coact!.survival
            : null);
        const upliftPct = uplift != null ? Math.round(uplift * 100) : null;
        const tone = upliftPct == null ? 'text-slate-700 dark:text-slate-300'
          : upliftPct > 0 ? 'text-emerald-300'
          : upliftPct < 0 ? 'text-amber-300'
          : 'text-slate-700 dark:text-slate-300';
        const sign = upliftPct != null && upliftPct > 0 ? '+' : '';
        return (
          <div className="rounded border border-violet-500/30 bg-violet-500/10 px-3 py-2 text-[11px] text-slate-700 dark:text-slate-300 space-y-1">
            <div className="flex items-center gap-3 flex-wrap">
              <span>co-activation ordering: <span className="font-mono text-slate-800 dark:text-slate-200">{pct(coact!.survival)}</span> survival</span>
              <span>attribution ordering: <span className="font-mono text-slate-800 dark:text-slate-200">{pct(attr!.survival)}</span> survival</span>
              <span className={`font-medium ${tone}`}>
                attribution re-ranking uplift: {upliftPct != null ? `${sign}${upliftPct}%` : '—'}
              </span>
            </div>
            <p className="text-[10px] text-slate-500">
              {upliftPct == null || upliftPct === 0
                ? 'attribution re-ranking left the causal survival rate unchanged.'
                : upliftPct > 0
                  ? 'attribution re-ranking raised the causal survival rate — its top edges validated more often.'
                  : 'attribution re-ranking lowered the causal survival rate — co-activation ordering validated more often here.'}
            </p>
          </div>
        );
      })()}

      {/* Only one ordering validated so far — point at the missing half. */}
      {batch && !bothOrderings && (coact != null || attr != null) && (
        <p className="text-[11px] text-slate-500">
          Validate the {coact != null ? 'attribution' : 'co-activation'} ordering too to
          compare survival rates and see the attribution re-ranking uplift.
        </p>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="text-slate-500 text-left">
              <th className="py-1 pr-3 font-medium">edge</th>
              <th className="py-1 pr-3 font-medium">ordering</th>
              <th className="py-1 pr-3 font-medium" title="effect size">ES</th>
              <th className="py-1 pr-3 font-medium">status</th>
              <th className="py-1 font-medium">manifest</th>
            </tr>
          </thead>
          <tbody>
            {validated.map((c, i) => {
              const v = c.validation!;
              const failHist = c.tested_and_failed_history?.find(
                (h) => h.ordering === v.ordering);
              return (
                <tr key={i} className="border-t border-slate-200 dark:border-slate-800 text-slate-700 dark:text-slate-300">
                  <td className="py-1 pr-3 font-mono whitespace-nowrap">
                    {nodeLabel(c.up)} → {nodeLabel(c.down)}
                  </td>
                  <td className="py-1 pr-3 font-mono text-slate-600 dark:text-slate-400">{v.ordering}</td>
                  <td className="py-1 pr-3 font-mono">{v.effect_size.toFixed(3)}</td>
                  <td className="py-1 pr-3">
                    {v.passed && c.validated_rung === 2 ? (
                      <span className="rounded bg-emerald-500/10 text-emerald-300 px-1.5 py-0.5 text-[10px] whitespace-nowrap">
                        causally validated (rung 2)
                      </span>
                    ) : (
                      <span
                        className="rounded bg-slate-100 text-slate-600 dark:bg-slate-700/60 dark:text-slate-400 px-1.5 py-0.5 text-[10px] whitespace-nowrap"
                        title={failHist?.reason}
                      >
                        tested, did not validate
                      </span>
                    )}
                  </td>
                  <td className="py-1">
                    <button
                      onClick={() => onOpenManifest(v.manifest_id)}
                      className="flex items-center gap-1 text-[11px] text-violet-300 hover:text-violet-200"
                      title="Open validation manifest"
                    >
                      <FileText className="w-3 h-3" /> manifest
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ValidationTab() {
  const [runs, setRuns] = useState<DiscoveryRun[]>([]);
  const [runId, setRunId] = useState('');
  const [run, setRun] = useState<DiscoveryRun | null>(null);
  const [ordering, setOrdering] = useState<'coact' | 'attr'>('coact');
  const [k, setK] = useState('20');
  const [promptsPerEdge, setPromptsPerEdge] = useState('8');
  const [nullSamples, setNullSamples] = useState('20');
  const [signFrac, setSignFrac] = useState('0.8');
  const [baseline, setBaseline] = useState<'zero' | 'corpus_mean'>('zero');
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [manifestId, setManifestId] = useState<string | null>(null);

  // Only completed discovery runs (with candidates) can be validated.
  const completed = runs.filter(
    (r) => r.status === 'completed' && r.candidate_count > 0);

  useEffect(() => {
    circuitsApi.listDiscoveries().then((r) => setRuns(r.discoveries)).catch(() => {});
  }, []);

  const loadRun = useCallback(() => {
    if (!runId) { setRun(null); return; }
    circuitsApi.getDiscovery(runId, true)
      .then(setRun)
      .catch((e) => setError(String(e.message ?? e)));
  }, [runId]);

  useEffect(() => { loadRun(); }, [loadRun]);

  // Light 2.5s poll while a validation pass is in flight (validation has its
  // own lifecycle; the discovery status stays 'completed'). Stops on terminal.
  useEffect(() => {
    if (!run || !isValidationActive(run.validation_status)) return;
    const t = setInterval(loadRun, 2500);
    return () => clearInterval(t);
  }, [run, loadRun]);

  const startValidation = () => {
    if (!run) return;
    setError(null);
    const body: ValidateConfig = {
      ordering,
      k: parseInt(k, 10) || 20,
      prompts_per_edge: parseInt(promptsPerEdge, 10) || 8,
      null_samples: parseInt(nullSamples, 10) || 20,
      percentile: 95,
      sign_frac: parseFloat(signFrac) || 0.8,
      baseline,
      seed: 0,
    };
    setSubmitting(true);
    circuitsApi.startValidation(run.id, body)
      .then(() => loadRun())
      // 409s surface the backend detail verbatim (e.g. "run attribution first"
      // for attr ordering without a prior attribution pass).
      .catch((e) => setError(String(e.detail ?? e.message ?? e)))
      .finally(() => setSubmitting(false));
  };

  const cancelValidation = () => {
    if (!run) return;
    circuitsApi.cancelValidation(run.id)
      .then(() => loadRun())
      .catch((e) => setError(String(e.detail ?? e.message ?? e)));
  };

  const vActive = isValidationActive(run?.validation_status);

  const Toggle = <T extends string>(
    { value, options, onChange }: { value: T; options: { v: T; label: string }[]; onChange: (v: T) => void },
  ) => (
    <div className="inline-flex rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 p-0.5">
      {options.map((o) => (
        <button
          key={o.v}
          onClick={() => onChange(o.v)}
          className={`px-2.5 py-1 text-xs rounded ${
            value === o.v ? 'bg-violet-600 text-white' : 'text-slate-700 dark:text-slate-300 hover:text-white'}`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );

  return (
    <div className="space-y-4">
      <p className="text-slate-600 dark:text-slate-400 text-sm">
        Validation is the rung-2 causal tier: it intervenes on each top-K edge and
        tests the downstream effect against a support-matched null. A passing edge is
        causally validated (rung 2); a failing edge is tested but did not validate.
      </p>

      {error && (
        <div className="rounded-lg border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-300">
          {error}
        </div>
      )}

      <div className={`${COMPONENTS.card.base} p-4 space-y-3`}>
        <div>
          <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Discovery run</label>
          <select
            className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
            value={runId}
            onChange={(e) => setRunId(e.target.value)}
          >
            <option value="">Select a completed discovery run…</option>
            {completed.map((r) => (
              <option key={r.id} value={r.id}>
                {r.id.slice(0, 8)} · {r.candidate_count} candidates
              </option>
            ))}
          </select>
        </div>

        <div className="flex flex-wrap gap-4">
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Ordering</label>
            <Toggle
              value={ordering}
              onChange={setOrdering}
              options={[{ v: 'coact', label: 'Co-activation' }, { v: 'attr', label: 'Attribution' }]}
            />
            {ordering === 'attr' && (
              <p className="text-[10px] text-amber-300/80 mt-1">
                Attribution ordering requires a prior attribution pass on the run.
              </p>
            )}
          </div>
          <div>
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Baseline</label>
            <Toggle
              value={baseline}
              onChange={setBaseline}
              options={[{ v: 'zero', label: 'Zero' }, { v: 'corpus_mean', label: 'Corpus mean' }]}
            />
          </div>
        </div>

        <div className="flex flex-wrap gap-3">
          <div className="flex-1 min-w-[120px]">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Top-K</label>
            <input type="number"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={k} onChange={(e) => setK(e.target.value)} />
          </div>
          <div className="flex-1 min-w-[120px]">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Prompts / edge</label>
            <input type="number"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={promptsPerEdge} onChange={(e) => setPromptsPerEdge(e.target.value)} />
          </div>
          <div className="flex-1 min-w-[120px]">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Null samples</label>
            <input type="number"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={nullSamples} onChange={(e) => setNullSamples(e.target.value)} />
          </div>
          <div className="flex-1 min-w-[120px]">
            <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Sign fraction</label>
            <input type="number" step="0.05"
              className="w-full rounded bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 px-2 py-1.5 text-sm text-slate-900 dark:text-slate-100"
              value={signFrac} onChange={(e) => setSignFrac(e.target.value)} />
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <button
            onClick={startValidation}
            disabled={submitting || !run || vActive}
            className="flex items-center gap-1.5 rounded bg-emerald-600 hover:bg-emerald-500 px-3 py-1.5 text-sm text-white disabled:opacity-50"
          >
            <ShieldCheck className="w-4 h-4" /> Validate top-K
          </button>
          {vActive && (
            <span className="flex items-center gap-2 text-[11px] text-violet-300">
              validation {run?.validation_status}
              {run?.validation_progress != null && ` · ${Math.round(run.validation_progress)}%`}
              <button onClick={cancelValidation}
                className="rounded bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 px-2 py-0.5 text-slate-800 dark:text-slate-200">
                cancel
              </button>
            </span>
          )}
          {run?.validation_status === 'failed' && (
            <span className="text-[11px] text-red-300" title={run.validation_error ?? undefined}>
              validation failed{run.validation_error ? ` — ${run.validation_error}` : ''}
            </span>
          )}
          {run?.validation_status === 'cancelled' && (
            <span className="text-[11px] text-slate-600 dark:text-slate-400">validation cancelled</span>
          )}
        </div>

        {vActive && <ProgressBar value={run?.validation_progress ?? 0} />}
      </div>

      {run && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">Per-edge results</h3>
          <ValidationResults run={run} onOpenManifest={setManifestId} />
        </div>
      )}

      {manifestId && (
        <ManifestDrawer
          manifestId={manifestId}
          onClose={() => setManifestId(null)}
          onNavigate={setManifestId}
        />
      )}
    </div>
  );
}

// ── Top-level panel with tab switcher ─────────────────────────────────────

type CircuitTab = 'circuits' | 'capture' | 'discovery' | 'validation';

export function CircuitsPanel({ onNavigateToSteering }: { onNavigateToSteering?: () => void } = {}) {
  const [tab, setTab] = useState<CircuitTab>('circuits');

  const tabs: { id: CircuitTab; label: string }[] = [
    { id: 'circuits', label: 'Circuits' },
    { id: 'capture', label: 'Capture' },
    { id: 'discovery', label: 'Discovery' },
    { id: 'validation', label: 'Validation' },
  ];

  return (
    <div className="max-w-5xl mx-auto px-6 py-6 space-y-4">
      <div>
        <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <GitBranch className="w-5 h-5 text-violet-400" />
          Circuits
        </h1>
      </div>

      <div className="flex gap-1 border-b border-slate-200 dark:border-slate-700/60">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 ${
              tab === t.id
                ? 'border-violet-400 text-slate-900 dark:text-slate-100'
                : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200'}`}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'circuits' && <CircuitsListTab onNavigateToSteering={onNavigateToSteering} />}
      {tab === 'capture' && <CaptureTab />}
      {tab === 'discovery' && <DiscoveryTab />}
      {tab === 'validation' && <ValidationTab />}
    </div>
  );
}

export default CircuitsPanel;

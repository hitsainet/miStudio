/**
 * ProfilesMenu (Feature 014) — per-SAE list of saved cluster profiles with
 * load / export / delete / import. Unbound and imported profiles are badged;
 * the narrative shows in an expandable panel (react-markdown).
 */

import { useEffect, useRef, useState } from 'react';
import {
  BookMarked,
  ChevronDown,
  ChevronRight,
  Download,
  FileUp,
  Play,
  Trash2,
} from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { useClusterProfilesStore } from '../../stores/clusterProfilesStore';
import { useSteeringStore } from '../../stores/steeringStore';
import { ClusterProfile } from '../../types/clusterProfile';

export function ProfilesMenu() {
  const {
    profiles,
    isLoading,
    error,
    lastImport,
    fetchProfiles,
    deleteProfile,
    exportProfile,
    importPayload,
  } = useClusterProfilesStore();
  const { selectedSAE, loadProfileIntoSteering } = useSteeringStore();

  const [open, setOpen] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (open && selectedSAE) {
      // Fetch ALL profiles: server-side sae_id filtering would hide unbound
      // (imported) profiles, which must stay visible so they can be re-bound.
      fetchProfiles();
    }
  }, [open, selectedSAE, fetchProfiles]);

  if (!selectedSAE) return null;

  // Profiles bound to THIS SAE plus unbound ones (badged) — never other SAEs'.
  const visibleProfiles = profiles.filter((p) => !p.sae_id || p.sae_id === selectedSAE.id);

  const handleLoad = (profile: ClusterProfile) => {
    const ok = loadProfileIntoSteering(profile);
    let notice: string;
    if (ok) {
      notice = profile.sae_id
        ? `Loaded “${profile.name}” (${profile.members.length} members)`
        : `Loaded “${profile.name}” against ${selectedSAE.name ?? selectedSAE.id} (unbound profile — bound at load)`;
    } else if (!profile.members || profile.members.length === 0) {
      notice = 'Profile has no members to load';
    } else if (profile.members.length > 20) {
      notice = `Profile has ${profile.members.length} members — the steering panel holds at most 20`;
    } else if (profile.sae_id) {
      notice = 'Profile is bound to a different SAE — switch the SAE selector to load it';
    } else if (!selectedSAE.n_features) {
      notice = 'Cannot verify fit: the selected SAE has no recorded feature count';
    } else {
      notice = "Profile does not fit this SAE's feature space — select the matching SAE";
    }
    setNotice(notice);
  };

  const handleImportFile = async (file: File) => {
    try {
      const text = await file.text();
      const payload = JSON.parse(text) as Record<string, unknown>;
      // No forced binding: the server's compatibility matrix auto-binds to the
      // best local SAE (same id → same n_features+layer), warns, or imports
      // unbound. Forcing the currently-selected SAE would wrongly block
      // bundles authored against other local SAEs.
      const resp = await importPayload(payload, null);
      if (resp) {
        setNotice(
          `Imported ${resp.imported}, blocked ${resp.blocked}, errors ${resp.errors}`,
        );
      }
    } catch {
      setNotice('Not a valid JSON file');
    }
  };

  return (
    <div className="px-4 pb-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 text-xs text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        <BookMarked className="w-3 h-3 text-emerald-400" />
        Cluster profiles
      </button>

      {open && (
        <div className="mt-2 rounded-lg border border-slate-300 dark:border-slate-700/60 bg-slate-100 dark:bg-slate-900/60 p-2 space-y-1.5">
          <div className="flex items-center justify-between">
            <span className="text-[11px] text-slate-500">
              {isLoading ? 'Loading…' : `${visibleProfiles.length} available`}
            </span>
            <button
              onClick={() => fileRef.current?.click()}
              className="flex items-center gap-1 text-[11px] text-slate-600 dark:text-slate-400 hover:text-emerald-400"
              title="Import a .cluster.json definition or bundle"
            >
              <FileUp className="w-3 h-3" />
              Import
            </button>
            <input
              ref={fileRef}
              type="file"
              accept=".json,application/json"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (f) handleImportFile(f);
                e.target.value = '';
              }}
            />
          </div>

          {notice && <p className="text-[11px] text-cyan-400/90">{notice}</p>}
          {error && <p className="text-[11px] text-red-400">{error}</p>}
          {lastImport && lastImport.results.some((r) => r.warnings.length > 0) && (
            <div className="text-[10px] text-amber-400/90 space-y-0.5">
              {lastImport.results
                .flatMap((r) => r.warnings.map((w) => `${r.name}: ${w}`))
                .slice(0, 4)
                .map((w) => (
                  <p key={w}>{w}</p>
                ))}
            </div>
          )}

          {visibleProfiles.map((p) => (
            <div key={p.id} className="rounded border border-slate-200 dark:border-slate-800 bg-slate-100 dark:bg-slate-950/60 px-2 py-1.5">
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setExpandedId(expandedId === p.id ? null : p.id)}
                  className="flex-1 text-left text-xs text-slate-800 dark:text-slate-200 truncate hover:text-slate-900 dark:hover:text-white"
                  title={p.name}
                >
                  {p.name}
                  <span className="text-slate-500 ml-1.5">
                    {p.members.length} member{p.members.length === 1 ? '' : 's'}
                  </span>
                </button>
                {!p.sae_id && (
                  <span className="text-[9px] px-1 rounded bg-slate-100 dark:bg-slate-700/60 text-slate-600 dark:text-slate-400">unbound</span>
                )}
                {p.imported_from && (
                  <span className="text-[9px] px-1 rounded bg-indigo-500/15 text-indigo-300">imported</span>
                )}
                <button
                  onClick={() => handleLoad(p)}
                  className="text-slate-600 dark:text-slate-400 hover:text-emerald-400"
                  title="Load into steering (explicit tuned strengths)"
                >
                  <Play className="w-3 h-3" />
                </button>
                <button
                  onClick={() => exportProfile(p)}
                  className="text-slate-600 dark:text-slate-400 hover:text-cyan-400"
                  title="Export portable definition (.cluster.json)"
                >
                  <Download className="w-3 h-3" />
                </button>
                <button
                  onClick={() => {
                     
                    if (window.confirm(`Delete cluster profile “${p.name}”? This cannot be undone.`)) {
                      void deleteProfile(p.id);
                    }
                  }}
                  className="text-slate-600 dark:text-slate-400 hover:text-red-400"
                  title="Delete profile"
                >
                  <Trash2 className="w-3 h-3" />
                </button>
              </div>
              {expandedId === p.id && p.narrative && (
                <div className="mt-1.5 border-t border-slate-200 dark:border-slate-800 pt-1.5 text-[11px] text-slate-600 dark:text-slate-400 prose prose-invert prose-xs max-w-none max-h-40 overflow-y-auto">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>{p.narrative}</ReactMarkdown>
                </div>
              )}
            </div>
          ))}

          {!isLoading && visibleProfiles.length === 0 && (
            <p className="text-[11px] text-slate-600 py-1">
              No profiles yet — tune a cluster and “Save profile”.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

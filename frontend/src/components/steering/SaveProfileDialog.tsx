/**
 * SaveProfileDialog (Feature 014) — snapshot the current steering selection as
 * a durable cluster profile (name + narrative + tuned strengths + budget/λ).
 *
 * Save-then-export model: the dialog only SAVES; export always serializes the
 * stored profile from the ProfilesMenu. Opened from the SteeringPanel (and via
 * clusterProfilesStore.saveDialogOpen from the Clusters panel hand-off).
 */

import { useEffect, useState } from 'react';
import { X, Save } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';
import { useClusterProfilesStore } from '../../stores/clusterProfilesStore';
import {
  PROFILE_NAME_MAX,
  PROFILE_NARRATIVE_MAX,
  ProfileMember,
} from '../../types/clusterProfile';

export function SaveProfileDialog() {
  const { saveDialogOpen, setSaveDialogOpen, saveProfile, error, clearError } =
    useClusterProfilesStore();
  const { selectedSAE, selectedFeatures, clusterContext, clusterBudget, intensity } =
    useSteeringStore();

  const [name, setName] = useState('');
  const [narrative, setNarrative] = useState('');
  const [saving, setSaving] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);

  // Prefill from the cluster context each time the dialog opens.
  useEffect(() => {
    if (saveDialogOpen) {
      setName(clusterContext?.display_token ?? '');
      setNarrative('');
      setLocalError(null);
      clearError();
    }
  }, [saveDialogOpen, clusterContext, clearError]);

  if (!saveDialogOpen) return null;

  const canSave =
    !!selectedSAE && selectedFeatures.length > 0 && name.trim().length > 0 && !saving;

  const handleSave = async () => {
    if (!selectedSAE) return;
    if (name.trim().length === 0 || name.length > PROFILE_NAME_MAX) {
      setLocalError(`Name is required (max ${PROFILE_NAME_MAX} chars)`);
      return;
    }
    if (narrative.length > PROFILE_NARRATIVE_MAX) {
      setLocalError(`Narrative too long (max ${PROFILE_NARRATIVE_MAX} chars)`);
      return;
    }
    setSaving(true);
    try {
      const members: ProfileMember[] = selectedFeatures.map((f) => ({
        feature_idx: f.feature_idx,
        label: f.label ?? null,
        similarity: f.similarity ?? null,
        activation_frequency: f.activation_frequency ?? null,
        max_activation: f.max_activation ?? null,
        strength: f.strength,
        sign: f.strength < 0 ? -1 : 1,
        pinned: f.pinned ?? false,
        // Author meta round-trips (contract-rev review #5); backend
        // enrichment fills gaps but never overwrites these
        meta: f.meta ?? null,
      }));
      const profile = await saveProfile({
        sae_id: selectedSAE.id,
        source_group_id: clusterContext?.group_id ?? null,
        name: name.trim(),
        narrative: narrative.trim() || null,
        display_token: clusterContext?.display_token ?? null,
        members,
        budget: clusterBudget
          ? {
              B: clusterBudget.B,
              B_dir: clusterBudget.B_dir,
              G: clusterBudget.G,
              f_eff: clusterBudget.f_eff ?? null,
              formula_id: clusterBudget.formula_id ?? null,
              constants: clusterBudget.constants ?? null,
              intensity,
            }
          : { intensity },
      });
      if (profile) {
        // Saved profile becomes the active one; label tier 1 — the AUTHORED
        // name now titles blended results (works for hand-built selections
        // too: the profile itself becomes the cluster identity).
        useSteeringStore.setState({
          activeProfile: { id: profile.id, name: profile.name },
          clusterContext: {
            group_id: clusterContext?.group_id ?? profile.id,
            display_token: profile.name,
          },
        });
        setSaveDialogOpen(false);
      }
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60" onClick={() => setSaveDialogOpen(false)} />
      <div className="relative bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg p-6 w-full max-w-md shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium text-slate-900 dark:text-slate-100">Save Cluster Profile</h3>
          <button
            onClick={() => setSaveDialogOpen(false)}
            className="text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200"
            aria-label="Close"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
              Name *
            </label>
            <input
              type="text"
              value={name}
              maxLength={PROFILE_NAME_MAX}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Fear response cluster"
              className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">
              Narrative <span className="text-slate-500">(markdown, optional)</span>
            </label>
            <textarea
              value={narrative}
              maxLength={PROFILE_NARRATIVE_MAX}
              onChange={(e) => setNarrative(e.target.value)}
              rows={5}
              placeholder="What this cluster steers toward, evidence, tuning notes…"
              className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-emerald-500 resize-y"
            />
          </div>

          <p className="text-xs text-slate-500">
            Snapshots {selectedFeatures.length} member
            {selectedFeatures.length === 1 ? '' : 's'} with their tuned strengths
            {clusterBudget ? `, budget B=${clusterBudget.B.toFixed(1)}` : ''} and λ=
            {intensity.toFixed(2)}.
          </p>

          {(localError || error) && (
            <p className="text-xs text-red-400">{localError || error}</p>
          )}

          <div className="flex justify-end gap-2">
            <button
              onClick={() => setSaveDialogOpen(false)}
              className="px-3 py-1.5 text-sm text-slate-700 dark:text-slate-300 hover:text-slate-900 dark:hover:text-slate-100"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!canSave}
              className="flex items-center gap-1.5 px-3 py-1.5 text-sm rounded bg-emerald-600 hover:bg-emerald-500 disabled:bg-slate-100 dark:disabled:bg-slate-700 disabled:text-slate-500 text-white"
            >
              <Save className="w-3.5 h-3.5" />
              {saving ? 'Saving…' : 'Save profile'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

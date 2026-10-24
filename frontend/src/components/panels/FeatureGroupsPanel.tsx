/**
 * Feature Groups panel (Feature 010).
 *
 * Browse cross-feature groups — features sharing a top activating token with
 * similar activation context — for a completed extraction. Same REST
 * endpoints the MCP server's `groups` tools use: one source of truth.
 */

import { useEffect, useMemo, useState } from 'react';
import { Boxes, Sliders, BookMarked } from 'lucide-react';
import { getSAE, getSAEs } from '../../api/saes';
import type { SAE } from '../../types/sae';
import { composeSAEOptionLabel } from '../../utils/saeOptionLabel';
import { useFeatureGroupsStore, deriveSourceCluster } from '../../stores/featureGroupsStore';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useSteeringStore } from '../../stores/steeringStore';
import { useClusterProfilesStore } from '../../stores/clusterProfilesStore';
import { useFeatureGroupsWebSocket } from '../../hooks/useFeatureGroupsWebSocket';
import { ComputeIndexBanner } from '../featureGroups/ComputeIndexBanner';
import { GroupList } from '../featureGroups/GroupList';
import { RelatedFeaturesDrawer } from '../featureGroups/RelatedFeaturesDrawer';
import { FeatureDetailModal } from '../features/FeatureDetailModal';

interface FeatureGroupsPanelProps {
  onNavigateToSteering?: () => void;
}

export function FeatureGroupsPanel({ onNavigateToSteering }: FeatureGroupsPanelProps = {}) {
  const { allExtractions, fetchAllExtractions } = useFeaturesStore();
  const {
    extractionId,
    setExtraction,
    status,
    selection,
    clearSelection,
    error,
  } = useFeatureGroupsStore();
  const { selectSAE, addFeature, setClusterContext, requestClusterAllocation } = useSteeringStore();
  const [detailFeatureId, setDetailFeatureId] = useState<string | null>(null);
  const [handoffError, setHandoffError] = useState<string | null>(null);
  // Linked-SAE lookup so dropdown labels can show layer/hook/width — the
  // extraction records only carry name + model. Best-effort enrichment:
  // labels degrade gracefully (chips omitted) until/unless this resolves.
  const [saesById, setSaesById] = useState<Record<string, SAE>>({});

  useFeatureGroupsWebSocket(extractionId);

  useEffect(() => {
    void fetchAllExtractions(['completed']);
  }, [fetchAllExtractions]);

  useEffect(() => {
    void getSAEs({ limit: 100 })
      .then((response) => {
        const map: Record<string, SAE> = {};
        for (const sae of response.data) map[sae.id] = sae;
        setSaesById(map);
      })
      .catch(() => {
        // Label enrichment only — the dropdown still renders without it.
      });
  }, []);

  const completedExtractions = useMemo(
    () => allExtractions.filter((e) => e.status === 'completed'),
    [allExtractions]
  );

  // Auto-select the most recent completed extraction on first load
  useEffect(() => {
    if (!extractionId && completedExtractions.length > 0) {
      setExtraction(completedExtractions[0].id);
    }
  }, [extractionId, completedExtractions, setExtraction]);

  const currentExtraction = completedExtractions.find((e) => e.id === extractionId);

  const handleSteerSelected = async (openSaveDialog = false) => {
    if (!currentExtraction || selection.size === 0) return;
    setHandoffError(null);
    const saeId = currentExtraction.external_sae_id;
    if (!saeId) {
      setHandoffError('This extraction has no linked SAE — steering hand-off unavailable.');
      return;
    }
    try {
      // Cluster provenance from the selection's own stamps (Feature 012): each
      // member records which cluster it was selected FROM, so provenance is
      // independent of whichever cluster happens to be expanded right now.
      const members = [...selection.values()];
      const sourceCluster = deriveSourceCluster(members);

      const sae = await getSAE(saeId);
      selectSAE(sae); // clears prior selection + cluster context
      const layer = currentExtraction.layer_index ?? sae.layer ?? 0;
      let added = 0;
      for (const [featureId, member] of selection.entries()) {
        // Omit strength so the store auto-computes the baseline from the
        // member's activation frequency (Feature 011).
        const ok = addFeature({
          feature_idx: member.neuron_index,
          layer,
          label: null,
          feature_id: featureId,
          max_activation: member.max_activation,
          activation_frequency: member.activation_frequency,
          similarity: member.similarity,
        });
        if (!ok) break; // max features reached
        added += 1;
      }
      if (added > 0) {
        // Set context LAST — addFeature clears it on every mutation, and it
        // must only stand when the full selection made it across (Feature 012).
        if (sourceCluster && added === selection.size) {
          setClusterContext(sourceCluster);
          // Feature 013: fetch the principled cluster allocation (progressive —
          // solo baselines stay until it lands; failures are non-fatal).
          void requestClusterAllocation(members[0]?.cohesion ?? null);
        }
        clearSelection();
        onNavigateToSteering?.();
        // Feature 014: "Steer & save profile…" opens the SaveProfileDialog on
        // arrival in Steering (prefilled from the cluster context just set).
        if (openSaveDialog) {
          useClusterProfilesStore.getState().setSaveDialogOpen(true);
        }
      }
    } catch (e: any) {
      setHandoffError(e.detail || e.message || 'Failed to hand off to steering');
    }
  };

  return (
    // Fill the viewport below the sticky app header (h-14) so the group list
    // can scroll independently of the pinned heading matter.
    <div className="h-[calc(100vh-3.5rem)] flex flex-col max-w-6xl mx-auto px-6 pt-6">
      {/* ── Pinned header matter ── */}
      <div className="shrink-0">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Boxes className="w-5 h-5 text-emerald-400" />
            <h1 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Clusters</h1>
          </div>
          {selection.size > 0 && (
            <div className="flex items-center gap-2">
              <button
                onClick={() => void handleSteerSelected()}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-emerald-600 hover:bg-emerald-500 text-white rounded-md"
              >
                <Sliders className="w-4 h-4" />
                Steer selected ({selection.size})
              </button>
              <button
                onClick={() => void handleSteerSelected(true)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 text-emerald-300 border border-emerald-600/40 rounded-md"
                title="Hand off to Steering and open the Save Cluster Profile dialog"
              >
                <BookMarked className="w-4 h-4" />
                Steer &amp; save profile…
              </button>
            </div>
          )}
        </div>

        <p className="text-sm text-slate-500 mb-4">
          Clusters of features that fire on the same top activating token with similar surrounding
          context — candidate units of meaning. Select members and hand them to Steering to
          validate a cluster's hypothesized meaning.
        </p>

        <div className="mb-4">
          <label className="text-xs text-slate-500 block mb-1">Extraction</label>
          <select
            value={extractionId ?? ''}
            onChange={(e) => setExtraction(e.target.value || null)}
            className="bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-md px-3 py-2 text-sm text-slate-800 dark:text-slate-200 w-full max-w-xl"
          >
            <option value="">Select a completed extraction…</option>
            {completedExtractions.map((extraction) => {
              const sae = extraction.external_sae_id
                ? saesById[extraction.external_sae_id]
                : undefined;
              return (
                <option key={extraction.id} value={extraction.id}>
                  {composeSAEOptionLabel({
                    name: extraction.sae_name || extraction.id,
                    modelName: extraction.model_name ?? sae?.model_name,
                    layer: extraction.layer_index ?? sae?.layer,
                    hookType: extraction.hook_type ?? sae?.hook_type,
                    width: sae?.n_features ?? extraction.total_features,
                  })}
                </option>
              );
            })}
          </select>
        </div>

        {(handoffError || error) && (
          <p className="text-xs text-red-400 mb-3">{handoffError || error}</p>
        )}

        {extractionId && <ComputeIndexBanner />}
      </div>

      {/* ── Independently scrolling group list ── */}
      {extractionId && status?.status === 'completed' && (
        <GroupList onOpenFeature={setDetailFeatureId} />
      )}
      {!extractionId && completedExtractions.length === 0 && (
        <div className="text-sm text-slate-500 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-lg p-6 text-center">
          No completed extractions yet. Run a feature extraction from the SAEs or Extractions
          panel first.
        </div>
      )}

      <RelatedFeaturesDrawer onOpenFeature={setDetailFeatureId} />

      {detailFeatureId && (
        <FeatureDetailModal
          featureId={detailFeatureId}
          trainingId={currentExtraction?.training_id ?? null}
          onClose={() => setDetailFeatureId(null)}
        />
      )}
    </div>
  );
}

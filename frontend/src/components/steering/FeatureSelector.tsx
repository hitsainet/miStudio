/**
 * FeatureSelector - Sidebar component for managing selected features.
 *
 * Features:
 * - SAE selector dropdown
 * - List of selected features with strength sliders
 * - Feature browser toggle
 * - Clear all button
 * - Max features indicator (Feature 011: up to 20)
 * - Right-click context menu for viewing feature details
 */

import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Brain, Plus, Trash2, ChevronUp, Search, Eye, Copy, Boxes, BookMarked, Layers } from 'lucide-react';
import { useSteeringStore, MAX_SELECTED_FEATURES } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { SAEStatus } from '../../types/sae';
import { SelectedFeature } from '../../types/steering';
import { SelectedFeatureCard } from './SelectedFeatureCard';
import { ClusterBudgetBar } from './ClusterBudgetBar';
import { ProfilesMenu } from './ProfilesMenu';
import { SaveProfileDialog } from './SaveProfileDialog';
import { useClusterProfilesStore } from '../../stores/clusterProfilesStore';
import { FeatureBrowser } from './FeatureBrowser';
import { FeatureDetailModal } from '../features/FeatureDetailModal';
import { composeSAEOptionLabel } from '../../utils/saeOptionLabel';
import { COMPONENTS } from '../../config/brand';

// Context menu state interface
interface ContextMenuState {
  visible: boolean;
  x: number;
  y: number;
  instanceId: string | null;
  featureIdx: number | null;
  layer: number | null;
  featureId: string | null;
}

export function FeatureSelector() {
  const [showBrowser, setShowBrowser] = useState(false);

  // Context menu and modal state
  const [contextMenu, setContextMenu] = useState<ContextMenuState>({
    visible: false,
    x: 0,
    y: 0,
    instanceId: null,
    featureIdx: null,
    layer: null,
    featureId: null,
  });
  const [selectedFeatureForModal, setSelectedFeatureForModal] = useState<{
    featureId: string;
    trainingId: string;
  } | null>(null);
  const contextMenuRef = useRef<HTMLDivElement>(null);

  const {
    selectedSAE,
    selectedFeatures,
    selectSAE,
    removeFeature,
    duplicateFeature,
    updateFeatureStrength,
    setAdditionalStrengths,
    applyStrengthPreset,
    applyAutoBaseline,
    clearFeatures,
    clusterContext,
    clusterBudget,
    layerBudgets,
    rebalance,
    togglePin,
  } = useSteeringStore();
  // Feature 015: a budget governs the selection when EITHER the single 013
  // budget OR a per-layer (multi-layer) budget map is present.
  const budgetGoverns = !!clusterBudget || !!layerBudgets;
  const { setSaveDialogOpen } = useClusterProfilesStore();

  // Strength preset values
  const PRESETS = [
    { label: 'Subtle', value: 10 },
    { label: 'Moderate', value: 50 },
    { label: 'Strong', value: 100 },
  ] as const;

  const { saes } = useSAEsStore();
  const readySAEs = saes.filter((sae) => sae.status === SAEStatus.READY);

  // Close context menu when clicking outside or pressing Escape
  useEffect(() => {
    if (!contextMenu.visible) {
      return;
    }

    const handleClickOutside = (event: MouseEvent) => {
      if (contextMenuRef.current && !contextMenuRef.current.contains(event.target as Node)) {
        setContextMenu((prev) => ({ ...prev, visible: false }));
      }
    };

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setContextMenu((prev) => ({ ...prev, visible: false }));
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [contextMenu.visible]);

  // Handle right-click on selected feature card
  const handleSelectedFeatureContextMenu = (event: React.MouseEvent, feature: SelectedFeature) => {
    event.preventDefault();
    event.stopPropagation();
    setContextMenu({
      visible: true,
      x: event.clientX,
      y: event.clientY,
      instanceId: feature.instance_id,
      featureIdx: feature.feature_idx,
      layer: feature.layer,
      featureId: feature.feature_id,
    });
  };

  // Handle view feature details from context menu
  const handleViewFeatureDetails = async () => {
    if (contextMenu.featureIdx === null || !selectedSAE) {
      setContextMenu((prev) => ({ ...prev, visible: false }));
      return;
    }

    const saeId = selectedSAE.id;
    const trainingId = selectedSAE.training_id;
    const featureIdx = contextMenu.featureIdx;
    const layer = contextMenu.layer;
    setContextMenu((prev) => ({ ...prev, visible: false }));

    const openModal = (featureId: string) => {
      setSelectedFeatureForModal({
        featureId,
        trainingId: trainingId || saeId,
      });
    };

    // If feature_id is directly available in the selected feature, use it
    if (contextMenu.featureId) {
      openModal(contextMenu.featureId);
      return;
    }

    // Fallback: Try to find feature_id from the Feature Browser data
    const { featureBrowser } = useSAEsStore.getState();
    if (featureBrowser.data?.features) {
      const matchingFeature = featureBrowser.data.features.find(
        (f) => f.feature_idx === featureIdx && f.layer === layer
      );
      if (matchingFeature?.feature_id) {
        openModal(matchingFeature.feature_id);
        return;
      }
    }

    // API fallback: try training_id lookup, then SAE ID lookup
    try {
      if (trainingId) {
        const response = await axios.get(`/api/v1/trainings/${trainingId}/features/by-index/${featureIdx}`);
        if (response.data.feature_id) {
          openModal(response.data.feature_id);
          return;
        }
      }

      // Try SAE ID lookup (for external/downloaded SAEs)
      const response = await axios.get(`/api/v1/saes/${saeId}/features/by-index/${featureIdx}`);
      if (response.data.feature_id) {
        openModal(response.data.feature_id);
        return;
      }
    } catch (error) {
      console.log('[FeatureSelector] Feature lookup failed:', error);
    }

    alert(`Feature #${featureIdx} has not been extracted yet.\n\nRun an extraction job on the Extractions tab to view details for this feature.`);
  };

  // Handle duplicate feature from context menu
  const handleDuplicateFeature = () => {
    if (!contextMenu.instanceId) {
      setContextMenu((prev) => ({ ...prev, visible: false }));
      return;
    }

    const success = duplicateFeature(contextMenu.instanceId);
    if (!success) {
      console.log('[FeatureSelector] Could not duplicate feature - max limit reached or feature not found');
    }
    setContextMenu((prev) => ({ ...prev, visible: false }));
  };

  // Check if we can duplicate (under max limit)
  const canDuplicate = selectedFeatures.length < MAX_SELECTED_FEATURES;

  // Close feature detail modal
  const handleCloseModal = () => {
    setSelectedFeatureForModal(null);
  };

  const handleSAEChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const saeId = e.target.value;
    const sae = saes.find((s) => s.id === saeId);
    selectSAE(sae || null);
    setShowBrowser(false);
  };

  const canAddMore = selectedFeatures.length < MAX_SELECTED_FEATURES;

  return (
    <div className="h-full flex flex-col bg-slate-50 dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800">
      {/* Header */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-1">Feature Steering</h2>
        <p className="text-sm text-slate-600 dark:text-slate-400">Select up to {MAX_SELECTED_FEATURES} features to steer</p>
      </div>

      {/* SAE Selector */}
      <div className="p-4 border-b border-slate-200 dark:border-slate-800">
        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          <Brain className="w-4 h-4 inline mr-1.5" />
          Select SAE
        </label>
        <select
          value={selectedSAE?.id || ''}
          onChange={handleSAEChange}
          className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 transition-colors"
        >
          <option value="">Choose an SAE...</option>
          {readySAEs.map((sae) => (
            <option key={sae.id} value={sae.id}>
              {composeSAEOptionLabel({
                name: sae.name,
                modelName: sae.model_name,
                layer: sae.layer,
                hookType: sae.hook_type,
                width: sae.n_features,
              })}
            </option>
          ))}
        </select>
        {readySAEs.length === 0 && (
          <p className="mt-2 text-xs text-slate-500">
            No SAEs ready. Download one from the SAEs tab.
          </p>
        )}
      </div>

      {/* Selected Features */}
      <div className="flex-1 overflow-y-auto">
        {selectedSAE ? (
          <>
            {/* Cluster provenance chip (Feature 012). Count derives from the live
                selection — while context stands, the selection IS the hand-off set. */}
            {clusterContext && (
              <div className="px-4 pt-3 flex items-center gap-2 flex-wrap">
                <span
                  className={`${COMPONENTS.badge.default} gap-1.5 border-cyan-500/40 bg-cyan-500/10 text-cyan-300`}
                  title={`These features were handed off together from the "${clusterContext.display_token}" cluster`}
                >
                  <Boxes className="w-3 h-3" />
                  {clusterContext.display_token}
                  <span className="text-cyan-500/70">
                    · {selectedFeatures.length} member{selectedFeatures.length !== 1 ? 's' : ''} selected
                  </span>
                </span>
                {/* Feature 015: layer-span badge (FPRD §4). Shown when a loaded
                    circuit spans >1 distinct layer, e.g. "L13+L14". */}
                {(() => {
                  const layers = [...new Set(selectedFeatures.map((f) => f.layer))].sort((a, b) => a - b);
                  if (layers.length <= 1) return null;
                  return (
                    <span
                      className={`${COMPONENTS.badge.default} gap-1.5 border-violet-500/40 bg-violet-500/10 text-violet-300`}
                      title={`This selection spans ${layers.length} layers`}
                    >
                      <Layers className="w-3 h-3" />
                      {layers.map((l) => `L${l}`).join('+')}
                    </span>
                  );
                })()}
              </div>
            )}

            {/* Cluster budget bar (Feature 013) */}
            <ClusterBudgetBar />

            {/* Saved cluster profiles (Feature 014): list/load/export/import */}
            <ProfilesMenu />

            {/* Feature count header */}
            <div className="p-4 pb-2 flex items-center justify-between">
              <h3 className="text-sm font-medium text-slate-700 dark:text-slate-300">
                Selected Features ({selectedFeatures.length}/{MAX_SELECTED_FEATURES})
              </h3>
              <div className="flex items-center gap-3">
                {selectedFeatures.length > 0 && (
                  <button
                    onClick={() => setSaveDialogOpen(true)}
                    className="text-xs text-emerald-400/90 hover:text-emerald-300 flex items-center gap-1"
                    title="Save the current selection + strengths as a cluster profile"
                  >
                    <BookMarked className="w-3 h-3" />
                    Save profile
                  </button>
                )}
                {selectedFeatures.length > 0 && (
                  <button
                    onClick={clearFeatures}
                    className="text-xs text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 flex items-center gap-1"
                  >
                    <Trash2 className="w-3 h-3" />
                    Clear all
                  </button>
                )}
              </div>
            </div>

            {/* Strength Presets */}
            {selectedFeatures.length > 0 && (
              <div className="px-4 pb-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-slate-500">Apply to all:</span>
                  {PRESETS.map((preset) => (
                    <button
                      key={preset.label}
                      onClick={() => applyStrengthPreset(preset.value)}
                      className="bg-white dark:bg-slate-800 hover:bg-slate-100 dark:hover:bg-slate-700 rounded px-3 py-1 text-xs text-slate-700 dark:text-slate-300 transition-colors"
                    >
                      {preset.label}
                    </button>
                  ))}
                  {/* Feature 011: recompute each tile's baseline from its stored
                      activation frequency (falls back to default where absent). */}
                  <button
                    onClick={applyAutoBaseline}
                    className="bg-emerald-500/15 hover:bg-emerald-500/25 text-emerald-400 rounded px-3 py-1 text-xs transition-colors"
                    title="Set each feature's strength from its activation frequency"
                  >
                    Auto
                  </button>
                </div>
              </div>
            )}

            {/* Selected features list */}
            <div className="px-4 space-y-3">
              {selectedFeatures.map((feature) => (
                <SelectedFeatureCard
                  key={feature.instance_id}
                  feature={feature}
                  onStrengthChange={(strength) =>
                    budgetGoverns
                      ? rebalance(feature.layer, feature.instance_id, strength)
                      : updateFeatureStrength(feature.instance_id, strength)
                  }
                  onAdditionalStrengthsChange={(strengths) =>
                    setAdditionalStrengths(feature.instance_id, strengths)
                  }
                  onRemove={() => removeFeature(feature.instance_id)}
                  onTogglePin={budgetGoverns ? () => togglePin(feature.instance_id) : undefined}
                  onContextMenu={handleSelectedFeatureContextMenu}
                />
              ))}

              {selectedFeatures.length === 0 && (
                <div className="text-center py-8 text-slate-500">
                  <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">No features selected</p>
                  <p className="text-xs mt-1">Use the browser below to find features</p>
                </div>
              )}
            </div>

            {/* Add feature button / Browser toggle */}
            <div className="p-4">
              {canAddMore ? (
                <button
                  onClick={() => setShowBrowser(!showBrowser)}
                  className={`w-full py-2 rounded-lg flex items-center justify-center gap-2 transition-colors ${
                    showBrowser
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                      : `${COMPONENTS.button.secondary}`
                  }`}
                >
                  {showBrowser ? (
                    <>
                      <ChevronUp className="w-4 h-4" />
                      Hide Browser
                    </>
                  ) : (
                    <>
                      <Plus className="w-4 h-4" />
                      Add Feature
                    </>
                  )}
                </button>
              ) : (
                <div className="text-center py-2 text-sm text-slate-500">
                  Maximum {MAX_SELECTED_FEATURES} features selected
                </div>
              )}
            </div>

            {/* Feature Browser (inline) */}
            {showBrowser && selectedSAE && (
              <div className="px-4 pb-4">
                <FeatureBrowser saeId={selectedSAE.id} />
              </div>
            )}
          </>
        ) : (
          <div className="p-4 text-center text-slate-500">
            <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
            <p>Select an SAE to begin</p>
          </div>
        )}
      </div>

      {/* Context Menu */}
      {contextMenu.visible && (
        <div
          ref={contextMenuRef}
          className="fixed z-50 bg-white dark:bg-slate-800 border border-slate-200 dark:border-slate-700 rounded-lg shadow-xl py-1 min-w-[180px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            onClick={handleViewFeatureDetails}
            disabled={!selectedSAE?.training_id}
            className="w-full px-4 py-2 text-left text-sm text-slate-800 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Eye className="w-4 h-4" />
            View Feature Details
          </button>
          <button
            onClick={handleDuplicateFeature}
            disabled={!canDuplicate}
            className="w-full px-4 py-2 text-left text-sm text-slate-800 dark:text-slate-200 hover:bg-slate-100 dark:hover:bg-slate-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            title={canDuplicate ? 'Create a copy with negated strength' : `Maximum ${MAX_SELECTED_FEATURES} features reached`}
          >
            <Copy className="w-4 h-4" />
            Duplicate (Negated)
          </button>
          {contextMenu.featureIdx !== null && (
            <div className="px-4 py-1 text-xs text-slate-500 border-t border-slate-300 dark:border-slate-700 mt-1">
              Feature #{contextMenu.featureIdx} • L{contextMenu.layer}
            </div>
          )}
          {!canDuplicate && (
            <div className="px-4 py-1 text-xs text-amber-500 border-t border-slate-300 dark:border-slate-700 mt-1">
              Max {MAX_SELECTED_FEATURES} features - remove one to duplicate
            </div>
          )}
          {!selectedSAE?.training_id && (
            <div className="px-4 py-1 text-xs text-amber-500 border-t border-slate-300 dark:border-slate-700 mt-1">
              Feature details not available for downloaded SAEs
            </div>
          )}
        </div>
      )}

      {/* Feature Detail Modal */}
      {selectedFeatureForModal && (
        <FeatureDetailModal
          featureId={selectedFeatureForModal.featureId}
          trainingId={selectedFeatureForModal.trainingId}
          onClose={handleCloseModal}
        />
      )}

      {/* Save Cluster Profile dialog (Feature 014) */}
      <SaveProfileDialog />
    </div>
  );
}

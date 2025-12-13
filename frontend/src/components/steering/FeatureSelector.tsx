/**
 * FeatureSelector - Sidebar component for managing selected features.
 *
 * Features:
 * - SAE selector dropdown
 * - List of selected features with strength sliders
 * - Feature browser toggle
 * - Clear all button
 * - Max 4 features indicator
 * - Right-click context menu for viewing feature details
 */

import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Brain, Plus, Trash2, ChevronUp, Search, Eye, Copy } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { SAEStatus } from '../../types/sae';
import { SelectedFeature } from '../../types/steering';
import { SelectedFeatureCard } from './SelectedFeatureCard';
import { FeatureBrowser } from './FeatureBrowser';
import { FeatureDetailModal } from '../features/FeatureDetailModal';
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
    clearFeatures,
  } = useSteeringStore();

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
    if (contextMenu.featureIdx === null || !selectedSAE?.training_id) {
      console.log('[FeatureSelector] Cannot view details - no featureIdx or training_id', {
        featureIdx: contextMenu.featureIdx,
        trainingId: selectedSAE?.training_id,
      });
      setContextMenu((prev) => ({ ...prev, visible: false }));
      return;
    }

    const trainingId = selectedSAE.training_id;
    const featureIdx = contextMenu.featureIdx;
    const layer = contextMenu.layer;
    setContextMenu((prev) => ({ ...prev, visible: false }));

    // If feature_id is directly available in the selected feature, use it
    if (contextMenu.featureId) {
      console.log('[FeatureSelector] Using feature_id directly:', contextMenu.featureId);
      setSelectedFeatureForModal({
        featureId: contextMenu.featureId,
        trainingId: trainingId,
      });
      return;
    }

    // Fallback: Try to find feature_id from the Feature Browser data
    // This handles cases where feature_id wasn't preserved when adding to selected features
    const { featureBrowser } = useSAEsStore.getState();
    if (featureBrowser.data?.features) {
      const matchingFeature = featureBrowser.data.features.find(
        (f) => f.feature_idx === featureIdx && f.layer === layer
      );
      if (matchingFeature?.feature_id) {
        console.log('[FeatureSelector] Found feature_id from Feature Browser:', matchingFeature.feature_id);
        setSelectedFeatureForModal({
          featureId: matchingFeature.feature_id,
          trainingId: trainingId,
        });
        return;
      }
    }

    // API fallback: Look up feature_id by training_id + feature_idx
    // This handles manual index entry where feature_id wasn't available at add time
    try {
      console.log('[FeatureSelector] Looking up feature_id via API for feature_idx:', featureIdx);
      const response = await axios.get(`/api/v1/trainings/${trainingId}/features/by-index/${featureIdx}`);
      const { feature_id } = response.data;

      if (feature_id) {
        console.log('[FeatureSelector] Found feature_id via API:', feature_id);
        setSelectedFeatureForModal({
          featureId: feature_id,
          trainingId: trainingId,
        });
        return;
      }
    } catch (error) {
      console.log('[FeatureSelector] API lookup failed:', error);
    }

    // Final fallback: feature wasn't extracted, show helpful message
    console.log('[FeatureSelector] Feature not extracted - feature_id is null for feature_idx:', featureIdx);
    alert(`Feature #${featureIdx} has not been extracted yet.\n\nOnly features that activated during an extraction job have detailed data available.\n\nTo view details for this feature, run a new extraction job on the Extractions tab that includes this feature.`);
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
  const canDuplicate = selectedFeatures.length < 4;

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

  const canAddMore = selectedFeatures.length < 4;

  return (
    <div className="h-full flex flex-col bg-slate-950 border-r border-slate-800">
      {/* Header */}
      <div className="p-4 border-b border-slate-800">
        <h2 className="text-lg font-semibold text-slate-100 mb-1">Feature Steering</h2>
        <p className="text-sm text-slate-400">Select up to 4 features to steer</p>
      </div>

      {/* SAE Selector */}
      <div className="p-4 border-b border-slate-800">
        <label className="block text-sm font-medium text-slate-300 mb-2">
          <Brain className="w-4 h-4 inline mr-1.5" />
          Select SAE
        </label>
        <select
          value={selectedSAE?.id || ''}
          onChange={handleSAEChange}
          className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 transition-colors"
        >
          <option value="">Choose an SAE...</option>
          {readySAEs.map((sae) => (
            <option key={sae.id} value={sae.id}>
              {sae.name} {sae.layer != null && `(L${sae.layer})`}
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
            {/* Feature count header */}
            <div className="p-4 pb-2 flex items-center justify-between">
              <h3 className="text-sm font-medium text-slate-300">
                Selected Features ({selectedFeatures.length}/4)
              </h3>
              {selectedFeatures.length > 0 && (
                <button
                  onClick={clearFeatures}
                  className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
                >
                  <Trash2 className="w-3 h-3" />
                  Clear all
                </button>
              )}
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
                      className="bg-slate-800 hover:bg-slate-700 rounded px-3 py-1 text-xs text-slate-300 transition-colors"
                    >
                      {preset.label}
                    </button>
                  ))}
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
                    updateFeatureStrength(feature.instance_id, strength)
                  }
                  onAdditionalStrengthsChange={(strengths) =>
                    setAdditionalStrengths(feature.instance_id, strengths)
                  }
                  onRemove={() => removeFeature(feature.instance_id)}
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
                  Maximum 4 features selected
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
          className="fixed z-50 bg-slate-800 border border-slate-700 rounded-lg shadow-xl py-1 min-w-[180px]"
          style={{ left: contextMenu.x, top: contextMenu.y }}
        >
          <button
            onClick={handleViewFeatureDetails}
            disabled={!selectedSAE?.training_id}
            className="w-full px-4 py-2 text-left text-sm text-slate-200 hover:bg-slate-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <Eye className="w-4 h-4" />
            View Feature Details
          </button>
          <button
            onClick={handleDuplicateFeature}
            disabled={!canDuplicate}
            className="w-full px-4 py-2 text-left text-sm text-slate-200 hover:bg-slate-700 flex items-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
            title={canDuplicate ? 'Create a copy with negated strength' : 'Maximum 4 features reached'}
          >
            <Copy className="w-4 h-4" />
            Duplicate (Negated)
          </button>
          {contextMenu.featureIdx !== null && (
            <div className="px-4 py-1 text-xs text-slate-500 border-t border-slate-700 mt-1">
              Feature #{contextMenu.featureIdx} • L{contextMenu.layer}
            </div>
          )}
          {!canDuplicate && (
            <div className="px-4 py-1 text-xs text-amber-500 border-t border-slate-700 mt-1">
              Max 4 features - remove one to duplicate
            </div>
          )}
          {!selectedSAE?.training_id && (
            <div className="px-4 py-1 text-xs text-amber-500 border-t border-slate-700 mt-1">
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
    </div>
  );
}

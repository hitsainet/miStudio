/**
 * FeatureSelector - Sidebar component for managing selected features.
 *
 * Features:
 * - SAE selector dropdown
 * - List of selected features with strength sliders
 * - Feature browser toggle
 * - Clear all button
 * - Max 4 features indicator
 */

import { useState } from 'react';
import { Brain, Plus, Trash2, ChevronUp, Search } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { SAEStatus } from '../../types/sae';
import { SelectedFeatureCard } from './SelectedFeatureCard';
import { FeatureBrowser } from './FeatureBrowser';
import { COMPONENTS } from '../../config/brand';

export function FeatureSelector() {
  const [showBrowser, setShowBrowser] = useState(false);

  const {
    selectedSAE,
    selectedFeatures,
    selectSAE,
    removeFeature,
    updateFeatureStrength,
    clearFeatures,
  } = useSteeringStore();

  const { saes } = useSAEsStore();
  const readySAEs = saes.filter((sae) => sae.status === SAEStatus.READY);

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

            {/* Selected features list */}
            <div className="px-4 space-y-3">
              {selectedFeatures.map((feature) => (
                <SelectedFeatureCard
                  key={`${feature.feature_idx}-${feature.layer}`}
                  feature={feature}
                  onStrengthChange={(strength) =>
                    updateFeatureStrength(feature.feature_idx, feature.layer, strength)
                  }
                  onRemove={() => removeFeature(feature.feature_idx, feature.layer)}
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
    </div>
  );
}

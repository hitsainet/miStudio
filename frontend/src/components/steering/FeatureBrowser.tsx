/**
 * FeatureBrowser - Component for browsing and selecting SAE features.
 *
 * Features:
 * - Search features by label/description
 * - Paginated feature list
 * - Click to select feature for steering
 * - Shows feature statistics (activation count, mean, max)
 * - Layer filter
 * - Manual feature index entry
 */

import { useState, useEffect } from 'react';
import { Search, Hash, Plus, Layers, ChevronLeft, ChevronRight, Zap } from 'lucide-react';
import { useSAEsStore } from '../../stores/saesStore';
import { useSteeringStore } from '../../stores/steeringStore';
import { SAEFeatureSummary } from '../../types/sae';
import { FEATURE_COLOR_ORDER, FEATURE_COLORS } from '../../types/steering';
import { COMPONENTS } from '../../config/brand';

interface FeatureBrowserProps {
  saeId: string;
}

export function FeatureBrowser({ saeId }: FeatureBrowserProps) {
  const [searchQuery, setSearchQuery] = useState('');
  const [manualFeatureIdx, setManualFeatureIdx] = useState('');
  const [manualLayer, setManualLayer] = useState('');
  const [currentPage, setCurrentPage] = useState(0);
  const pageSize = 20;

  const { featureBrowser, browseFeatures, clearFeatureBrowser } = useSAEsStore();
  const { selectedFeatures, addFeature, selectedSAE } = useSteeringStore();

  // Get available layers from the selected SAE
  const saeLayer = selectedSAE?.layer;
  const availableLayers = saeLayer != null ? [saeLayer] : [];

  // Auto-select layer when SAE changes
  useEffect(() => {
    if (saeLayer != null) {
      setManualLayer(String(saeLayer));
    }
  }, [saeLayer]);

  // Load features on mount and when search changes
  useEffect(() => {
    const timer = setTimeout(() => {
      browseFeatures(saeId, {
        skip: currentPage * pageSize,
        limit: pageSize,
        search: searchQuery || undefined,
      });
    }, 300);
    return () => clearTimeout(timer);
  }, [saeId, searchQuery, currentPage, pageSize, browseFeatures]);

  // Cleanup on unmount
  useEffect(() => {
    return () => clearFeatureBrowser();
  }, [clearFeatureBrowser]);

  const handleSelectFeature = (feature: SAEFeatureSummary) => {
    const success = addFeature({
      feature_idx: feature.feature_idx,
      layer: feature.layer,
      strength: 100, // Default strength
      label: feature.label,
    });

    if (!success) {
      console.log('[FeatureBrowser] Could not add feature - max reached or duplicate');
    }
  };

  const handleManualAdd = () => {
    const featureIdx = parseInt(manualFeatureIdx, 10);
    const layer = parseInt(manualLayer, 10);

    if (isNaN(featureIdx) || featureIdx < 0) {
      return;
    }

    // Use selected SAE's layer as fallback
    const effectiveLayer = isNaN(layer) ? (saeLayer ?? 0) : layer;

    const success = addFeature({
      feature_idx: featureIdx,
      layer: effectiveLayer,
      strength: 100,
      label: null,
    });

    if (success) {
      setManualFeatureIdx('');
    }
  };

  const isFeatureSelected = (featureIdx: number, layer: number) => {
    return selectedFeatures.some((f) => f.feature_idx === featureIdx && f.layer === layer);
  };

  const getNextColor = () => {
    const usedColors = selectedFeatures.map((f) => f.color);
    return FEATURE_COLOR_ORDER.find((c) => !usedColors.includes(c));
  };

  const canAddMore = selectedFeatures.length < 4;
  const nextColor = getNextColor();

  return (
    <div className={`${COMPONENTS.card.base} p-4 space-y-4`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-slate-100">Browse Features</h3>
        {selectedFeatures.length > 0 && (
          <span className="text-sm text-slate-400">
            {selectedFeatures.length}/4 selected
          </span>
        )}
      </div>

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
        <input
          type="text"
          placeholder="Search features by label..."
          value={searchQuery}
          onChange={(e) => {
            setSearchQuery(e.target.value);
            setCurrentPage(0);
          }}
          className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 transition-colors"
        />
      </div>

      {/* Manual feature entry */}
      <div className="p-3 bg-slate-900/50 rounded-lg space-y-2">
        {/* Row 1: Layer selector and Add button */}
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-2">
            <label className="text-xs text-slate-400">Layer</label>
            <select
              value={manualLayer}
              onChange={(e) => setManualLayer(e.target.value)}
              className="px-3 py-1.5 bg-slate-900 border border-slate-700 rounded focus:outline-none focus:border-emerald-500 text-slate-100 text-sm"
            >
              {availableLayers.map((layer) => (
                <option key={layer} value={layer}>
                  L{layer}
                </option>
              ))}
            </select>
          </div>
          <div className="flex-1" />
          <button
            onClick={handleManualAdd}
            disabled={!canAddMore || !manualFeatureIdx}
            className={`px-3 py-1.5 rounded flex items-center gap-1 text-sm ${COMPONENTS.button.primary} disabled:opacity-50`}
          >
            <Plus className="w-4 h-4" />
            Add
          </button>
        </div>
        {/* Row 2: Feature Index input */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Feature Index</label>
          <input
            type="number"
            min="0"
            placeholder="e.g., 1234"
            value={manualFeatureIdx}
            onChange={(e) => setManualFeatureIdx(e.target.value)}
            className="w-full px-3 py-1.5 bg-slate-900 border border-slate-700 rounded focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 text-sm"
          />
        </div>
      </div>

      {/* Loading state */}
      {featureBrowser.loading && (
        <div className="text-center py-8">
          <div className="inline-block animate-spin rounded-full h-6 w-6 border-2 border-slate-700 border-t-emerald-500"></div>
          <p className="text-slate-400 mt-2 text-sm">Loading features...</p>
        </div>
      )}

      {/* Error state */}
      {featureBrowser.error && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {featureBrowser.error}
        </div>
      )}

      {/* Features list */}
      {featureBrowser.data && !featureBrowser.loading && (
        <>
          {featureBrowser.data.features.length === 0 ? (
            <div className="text-center py-8 text-slate-400">
              {searchQuery ? 'No features match your search' : 'No features found'}
            </div>
          ) : (
            <div className="space-y-2 max-h-80 overflow-y-auto">
              {featureBrowser.data.features.map((feature) => {
                const isSelected = isFeatureSelected(feature.feature_idx, feature.layer);
                const colorClass = nextColor ? FEATURE_COLORS[nextColor] : FEATURE_COLORS.teal;

                return (
                  <div
                    key={`${feature.feature_idx}-${feature.layer}`}
                    className={`p-3 rounded-lg border transition-all ${
                      isSelected
                        ? 'bg-emerald-500/10 border-emerald-500/50 cursor-not-allowed'
                        : canAddMore
                        ? 'bg-slate-800 border-slate-700 hover:border-slate-600 cursor-pointer'
                        : 'bg-slate-800 border-slate-700 opacity-50 cursor-not-allowed'
                    }`}
                    onClick={() => {
                      if (!isSelected && canAddMore) {
                        handleSelectFeature(feature);
                      }
                    }}
                  >
                    <div className="flex items-start justify-between gap-3">
                      <div className="flex-1 min-w-0">
                        {/* Feature identifier */}
                        <div className="flex items-center gap-2 text-sm mb-1">
                          <span className={`flex items-center gap-1 ${isSelected ? 'text-emerald-400' : 'text-slate-200'} font-medium`}>
                            <Hash className="w-3.5 h-3.5" />
                            {feature.feature_idx}
                          </span>
                          <span className="text-slate-500">•</span>
                          <span className="flex items-center gap-1 text-slate-400">
                            <Layers className="w-3.5 h-3.5" />
                            L{feature.layer}
                          </span>
                        </div>

                        {/* Label */}
                        {feature.label ? (
                          <p className="text-sm text-slate-300 line-clamp-2">{feature.label}</p>
                        ) : (
                          <p className="text-sm text-slate-500 italic">No label</p>
                        )}

                        {/* Stats */}
                        {(feature.activation_count != null || feature.mean_activation != null) && (
                          <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                            {feature.activation_count != null && (
                              <span className="flex items-center gap-1">
                                <Zap className="w-3 h-3" />
                                {feature.activation_count.toLocaleString()} activations
                              </span>
                            )}
                            {feature.mean_activation != null && (
                              <span>avg: {feature.mean_activation.toFixed(3)}</span>
                            )}
                            {feature.max_activation != null && (
                              <span>max: {feature.max_activation.toFixed(3)}</span>
                            )}
                          </div>
                        )}
                      </div>

                      {/* Add indicator */}
                      {!isSelected && canAddMore && (
                        <div className={`p-1.5 rounded ${colorClass.light}`}>
                          <Plus className={`w-4 h-4 ${colorClass.text}`} />
                        </div>
                      )}
                      {isSelected && (
                        <span className="text-xs text-emerald-400 font-medium">Selected</span>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          {/* Pagination */}
          {featureBrowser.data.pagination.total > pageSize && (
            <div className="flex items-center justify-between pt-2 border-t border-slate-800">
              <button
                onClick={() => setCurrentPage((p) => Math.max(0, p - 1))}
                disabled={currentPage === 0}
                className="p-1.5 rounded hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeft className="w-4 h-4 text-slate-400" />
              </button>
              <span className="text-sm text-slate-400">
                Page {currentPage + 1} of {Math.ceil(featureBrowser.data.pagination.total / pageSize)}
              </span>
              <button
                onClick={() => setCurrentPage((p) => p + 1)}
                disabled={!featureBrowser.data.pagination.has_more}
                className="p-1.5 rounded hover:bg-slate-800 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronRight className="w-4 h-4 text-slate-400" />
              </button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

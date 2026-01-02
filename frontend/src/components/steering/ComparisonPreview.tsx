/**
 * ComparisonPreview - Visual preview cards showing what will be generated.
 *
 * Displays:
 * - Unsteered baseline card (gray dot)
 * - Selected feature cards with colors, names, strength (α), and layer
 * - Placeholder card for adding more features
 */

import { Plus } from 'lucide-react';
import { SelectedFeature, FEATURE_COLORS } from '../../types/steering';

interface ComparisonPreviewProps {
  selectedFeatures: SelectedFeature[];
  onAddFeature?: () => void;
  maxFeatures?: number;
}

export function ComparisonPreview({
  selectedFeatures,
  onAddFeature,
  maxFeatures = 4,
}: ComparisonPreviewProps) {
  const canAddMore = selectedFeatures.length < maxFeatures;

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium text-slate-500 uppercase tracking-wider">
        Comparison Preview
      </h4>
      <div className="flex flex-wrap gap-2">
        {/* Unsteered baseline card - always shown */}
        <div className="flex items-center gap-2 px-3 py-2 bg-slate-800/50 border border-slate-700 rounded-lg">
          <div className="w-2.5 h-2.5 rounded-full bg-slate-500" />
          <div className="text-sm">
            <span className="text-slate-400">Unsteered</span>
            <span className="text-slate-600 ml-1">(baseline)</span>
          </div>
        </div>

        {/* Selected feature cards */}
        {selectedFeatures.map((feature) => {
          const colors = FEATURE_COLORS[feature.color];
          const strengthSign = feature.strength >= 0 ? '+' : '';
          const featureLabel = feature.label || `Feature ${feature.feature_idx}`;

          return (
            <div
              key={feature.instance_id}
              className={`flex items-center gap-2 px-3 py-2 ${colors.light} border ${colors.border}/30 rounded-lg`}
            >
              <div className={`w-2.5 h-2.5 rounded-full ${colors.bg}`} />
              <div className="text-sm">
                <span className={colors.text}>{featureLabel}</span>
                <span className="text-slate-500 ml-2">
                  α={strengthSign}{feature.strength}
                </span>
                <span className="text-slate-600 ml-1">
                  L{feature.layer}
                </span>
              </div>
            </div>
          );
        })}

        {/* Add feature placeholder */}
        {canAddMore && onAddFeature && (
          <button
            onClick={onAddFeature}
            className="flex items-center gap-2 px-3 py-2 border border-dashed border-slate-700 rounded-lg text-slate-500 hover:text-slate-400 hover:border-slate-600 transition-colors"
          >
            <Plus className="w-4 h-4" />
            <span className="text-sm">Add Feature</span>
          </button>
        )}
      </div>
    </div>
  );
}

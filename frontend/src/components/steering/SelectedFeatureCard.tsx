/**
 * SelectedFeatureCard - Card displaying a selected feature for steering.
 *
 * Features:
 * - Color-coded border matching feature color
 * - Feature index and layer display
 * - Label/description if available
 * - Strength slider with warning zones
 * - Remove button
 * - Drag handle for reordering
 * - Right-click context menu for viewing feature details
 */

import { GripVertical, X, Hash, Layers } from 'lucide-react';
import { SelectedFeature, FEATURE_COLORS } from '../../types/steering';
import { StrengthSlider } from './StrengthSlider';

interface SelectedFeatureCardProps {
  feature: SelectedFeature;
  onStrengthChange: (strength: number) => void;
  onRemove: () => void;
  onContextMenu?: (event: React.MouseEvent, feature: SelectedFeature) => void;
  isDragging?: boolean;
  dragHandleProps?: Record<string, any>;
}

export function SelectedFeatureCard({
  feature,
  onStrengthChange,
  onRemove,
  onContextMenu,
  isDragging = false,
  dragHandleProps,
}: SelectedFeatureCardProps) {
  const colorClasses = FEATURE_COLORS[feature.color];

  const handleContextMenu = (event: React.MouseEvent) => {
    if (onContextMenu) {
      event.preventDefault();
      event.stopPropagation();
      onContextMenu(event, feature);
    }
  };

  return (
    <div
      className={`rounded-lg border-2 p-3 transition-all ${colorClasses.border} ${colorClasses.light} ${
        isDragging ? 'opacity-50 scale-95' : ''
      }`}
      onContextMenu={handleContextMenu}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          {/* Drag handle */}
          {dragHandleProps && (
            <div
              {...dragHandleProps}
              className="cursor-grab active:cursor-grabbing p-1 -m-1 rounded hover:bg-white/10"
            >
              <GripVertical className="w-4 h-4 text-slate-500" />
            </div>
          )}

          {/* Feature identifier */}
          <div className="flex items-center gap-3">
            {/* Color dot */}
            <div className={`w-3 h-3 rounded-full ${colorClasses.bg}`} />

            {/* Feature index and layer */}
            <div className="flex items-center gap-2 text-sm">
              <span className={`flex items-center gap-1 ${colorClasses.text} font-medium`}>
                <Hash className="w-3.5 h-3.5" />
                {feature.feature_idx}
              </span>
              <span className="text-slate-500">•</span>
              <span className="flex items-center gap-1 text-slate-400">
                <Layers className="w-3.5 h-3.5" />
                L{feature.layer}
              </span>
            </div>
          </div>
        </div>

        {/* Remove button */}
        <button
          onClick={onRemove}
          className="p-1 rounded hover:bg-white/10 text-slate-400 hover:text-slate-200 transition-colors"
          title="Remove feature"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Label if available */}
      {feature.label && (
        <p className="text-sm text-slate-300 mb-3 line-clamp-2">{feature.label}</p>
      )}

      {/* Strength slider */}
      <StrengthSlider
        value={feature.strength}
        onChange={onStrengthChange}
        color={feature.color}
        compact
      />
    </div>
  );
}

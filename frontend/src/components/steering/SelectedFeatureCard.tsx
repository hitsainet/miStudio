/**
 * SelectedFeatureCard - Card displaying a selected feature for steering.
 *
 * Features:
 * - Color-coded border matching feature color
 * - Feature index and layer display
 * - Label/description if available
 * - Strength slider with warning zones
 * - Additional strengths for multi-strength testing (up to 3 text inputs)
 * - Remove button
 * - Drag handle for reordering
 * - Right-click context menu for viewing feature details
 */

import { useState } from 'react';
import { GripVertical, X, Hash, Layers, Plus } from 'lucide-react';
import { SelectedFeature, FEATURE_COLORS, getStrengthWarningLevel } from '../../types/steering';
import { StrengthSlider } from './StrengthSlider';

interface SelectedFeatureCardProps {
  feature: SelectedFeature;
  onStrengthChange: (strength: number) => void;
  onAdditionalStrengthsChange: (strengths: number[]) => void;
  onRemove: () => void;
  onContextMenu?: (event: React.MouseEvent, feature: SelectedFeature) => void;
  isDragging?: boolean;
  dragHandleProps?: Record<string, any>;
  disabled?: boolean;
}

/**
 * Get CSS classes for a strength input based on its warning level.
 */
function getStrengthInputClasses(strength: number | null): string {
  if (strength === null) {
    return 'border-slate-700 text-slate-500 placeholder-slate-600';
  }
  const level = getStrengthWarningLevel(strength);
  switch (level) {
    case 'extreme':
      return 'border-red-500/50 text-red-400 bg-red-500/10';
    case 'caution':
      return 'border-amber-500/50 text-amber-400 bg-amber-500/10';
    default:
      return 'border-emerald-500/50 text-emerald-400 bg-emerald-500/10';
  }
}

export function SelectedFeatureCard({
  feature,
  onStrengthChange,
  onAdditionalStrengthsChange,
  onRemove,
  onContextMenu,
  isDragging = false,
  dragHandleProps,
  disabled = false,
}: SelectedFeatureCardProps) {
  const colorClasses = FEATURE_COLORS[feature.color];

  // Local state for editing additional strengths
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [editValue, setEditValue] = useState('');

  const additionalStrengths = feature.additional_strengths || [];
  const canAddMore = additionalStrengths.length < 3;

  const handleContextMenu = (event: React.MouseEvent) => {
    if (onContextMenu) {
      event.preventDefault();
      event.stopPropagation();
      onContextMenu(event, feature);
    }
  };

  // Calculate what the next additional strength would be (for tooltip)
  const getNextStrengthInfo = (): { value: number; formula: string } => {
    const currentCount = additionalStrengths.length;
    let value: number;
    let formula: string;

    if (currentCount === 0) {
      value = feature.strength * 3;
      formula = `${feature.strength} × 3`;
    } else if (currentCount === 1) {
      value = additionalStrengths[0] * 2;
      formula = `${additionalStrengths[0]} × 2`;
    } else {
      value = additionalStrengths[1] * 1.2;
      formula = `${additionalStrengths[1]} × 1.2`;
    }

    return {
      value: Math.round(Math.min(300, Math.max(-300, value))),
      formula,
    };
  };

  const handleAddStrength = () => {
    if (canAddMore) {
      const { value } = getNextStrengthInfo();
      onAdditionalStrengthsChange([...additionalStrengths, value]);
    }
  };

  const handleRemoveStrength = (index: number) => {
    const newStrengths = additionalStrengths.filter((_, i) => i !== index);
    onAdditionalStrengthsChange(newStrengths);
  };

  const handleStrengthInputBlur = (index: number) => {
    const parsed = parseFloat(editValue);
    if (!isNaN(parsed)) {
      const clamped = Math.max(-300, Math.min(300, parsed));
      const newStrengths = [...additionalStrengths];
      newStrengths[index] = clamped;
      onAdditionalStrengthsChange(newStrengths);
    }
    setEditingIndex(null);
    setEditValue('');
  };

  const handleStrengthInputKeyDown = (e: React.KeyboardEvent, index: number) => {
    if (e.key === 'Enter') {
      handleStrengthInputBlur(index);
    } else if (e.key === 'Escape') {
      setEditingIndex(null);
      setEditValue('');
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

      {/* Primary Strength slider */}
      <StrengthSlider
        value={feature.strength}
        onChange={onStrengthChange}
        color={feature.color}
        compact
        disabled={disabled}
      />

      {/* Additional Strengths Section */}
      <div className="mt-3 pt-3 border-t border-slate-700/50">
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs text-slate-500">Additional Strengths</span>
          {canAddMore && (
            <button
              onClick={handleAddStrength}
              disabled={disabled}
              className="flex items-center gap-1 text-xs text-emerald-400 hover:text-emerald-300 disabled:opacity-50 disabled:cursor-not-allowed"
              title={`Add strength: ${getNextStrengthInfo().formula} = ${getNextStrengthInfo().value}`}
            >
              <Plus className="w-3 h-3" />
              Add
            </button>
          )}
        </div>

        {additionalStrengths.length === 0 ? (
          <p className="text-xs text-slate-600 italic">
            Click "Add" to test multiple strengths
          </p>
        ) : (
          <div className="flex flex-wrap gap-2">
            {additionalStrengths.map((strength, index) => (
              <div key={index} className="flex items-center gap-1">
                {editingIndex === index ? (
                  <input
                    type="text"
                    value={editValue}
                    onChange={(e) => setEditValue(e.target.value)}
                    onBlur={() => handleStrengthInputBlur(index)}
                    onKeyDown={(e) => handleStrengthInputKeyDown(e, index)}
                    autoFocus
                    className={`w-16 px-2 py-1 text-sm font-mono text-center rounded border transition-colors bg-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-500 ${getStrengthInputClasses(parseFloat(editValue) || null)}`}
                    placeholder="—"
                  />
                ) : (
                  <button
                    onClick={() => {
                      setEditingIndex(index);
                      setEditValue(strength.toString());
                    }}
                    disabled={disabled}
                    className={`w-16 px-2 py-1 text-sm font-mono text-center rounded border transition-colors hover:brightness-110 disabled:cursor-not-allowed ${getStrengthInputClasses(strength)}`}
                    title={`Click to edit strength ${index + 1}`}
                  >
                    {strength > 0 ? '+' : ''}{strength}
                  </button>
                )}
                <button
                  onClick={() => handleRemoveStrength(index)}
                  disabled={disabled}
                  className="p-0.5 text-slate-500 hover:text-red-400 transition-colors disabled:cursor-not-allowed"
                  title="Remove this strength"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

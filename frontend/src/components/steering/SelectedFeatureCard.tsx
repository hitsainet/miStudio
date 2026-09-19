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
import { GripVertical, X, Hash, Layers, Plus, ChevronDown, ChevronRight, Sparkles, Pin } from 'lucide-react';
import { SelectedFeature, FEATURE_COLORS, getStrengthWarningLevel } from '../../types/steering';
import { StrengthSlider } from './StrengthSlider';

interface SelectedFeatureCardProps {
  feature: SelectedFeature;
  onStrengthChange: (strength: number) => void;
  onAdditionalStrengthsChange: (strengths: number[]) => void;
  onRemove: () => void;
  /** Unpin a pinned member so it rejoins budget rebalancing (Feature 013). */
  onTogglePin?: () => void;
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
    return 'border-slate-300 dark:border-slate-700 text-slate-500 placeholder-slate-600';
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
  onTogglePin,
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

  // Feature 011: additional strengths collapsed by default to keep the tile
  // compact; auto-expanded when the feature already carries some.
  const [showAdditional, setShowAdditional] = useState(additionalStrengths.length > 0);

  // Feature 011: provenance badge — where this tile's strength came from.
  const strengthSource = feature.strengthSource;
  const freq = feature.activation_frequency;
  const maxAct = feature.max_activation;

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
      className={`rounded-lg border-2 p-2 transition-all ${colorClasses.border} ${colorClasses.light} ${
        isDragging ? 'opacity-50 scale-95' : ''
      }`}
      onContextMenu={handleContextMenu}
    >
      {/* Header row: drag · id/layer · stats · remove — everything on one line */}
      <div className="flex items-center gap-2 min-w-0">
        {/* Drag handle */}
        {dragHandleProps && (
          <div
            {...dragHandleProps}
            className="cursor-grab active:cursor-grabbing p-0.5 -m-0.5 rounded hover:bg-white/10 shrink-0"
          >
            <GripVertical className="w-3.5 h-3.5 text-slate-500" />
          </div>
        )}

        {/* Color dot */}
        <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${colorClasses.bg}`} />

        {/* Feature index and layer */}
        <span className={`flex items-center gap-0.5 ${colorClasses.text} font-medium text-sm shrink-0`}>
          <Hash className="w-3 h-3" />
          {feature.feature_idx}
        </span>
        <span
          className="flex items-center gap-0.5 text-slate-500 text-xs shrink-0"
          title={
            feature.sae_id
              ? `Layer ${feature.layer} · steered through SAE ${feature.sae_id}`
              : `Layer ${feature.layer}`
          }
        >
          <Layers className="w-3 h-3" />
          L{feature.layer}
        </span>

        {/* Provenance + stats badges (Feature 011) */}
        {strengthSource === 'auto' && (
          <span
            className="flex items-center gap-0.5 text-[10px] font-medium text-emerald-400 bg-emerald-500/10 rounded px-1 py-0.5 shrink-0"
            title={
              freq != null
                ? `Baseline auto-set from activation frequency ${freq.toFixed(3)}`
                : 'Baseline auto-set'
            }
          >
            <Sparkles className="w-2.5 h-2.5" />
            auto
          </span>
        )}
        {strengthSource === 'cluster' && (
          <span
            className="flex items-center gap-0.5 text-[10px] font-medium text-cyan-400 bg-cyan-500/10 rounded px-1 py-0.5 shrink-0"
            title="Strength set by the cluster budget model (edit to pin; others rebalance)"
          >
            <Sparkles className="w-2.5 h-2.5" />
            cluster
          </span>
        )}
        {/* Pins only mean anything under a governing budget — the parent passes
            onTogglePin only in cluster mode, which doubles as the visibility gate
            (a stale badge after the budget clears would be a false trust signal). */}
        {feature.pinned && onTogglePin && (
          <button
            type="button"
            onClick={onTogglePin}
            disabled={disabled}
            className="flex items-center gap-0.5 text-[10px] font-medium text-amber-400 bg-amber-500/10 rounded px-1 py-0.5 shrink-0 hover:bg-amber-500/20 disabled:cursor-default"
            title="Pinned — excluded from budget rebalancing. Click to unpin."
          >
            <Pin className="w-2.5 h-2.5" />
            pinned
          </button>
        )}
        {strengthSource === 'default' && (
          <span
            className="text-[10px] font-medium text-slate-500 bg-slate-100 dark:bg-slate-700/40 rounded px-1 py-0.5 shrink-0"
            title="No activation frequency available — using the default strength"
          >
            default
          </span>
        )}

        {/* Compact stats, right-aligned before remove */}
        <span className="ml-auto flex items-center gap-2 text-[10px] text-slate-500 shrink-0 font-mono">
          {freq != null && <span title="Activation frequency">f {freq.toFixed(3)}</span>}
          {maxAct != null && <span title="Max activation">m {maxAct.toFixed(2)}</span>}
        </span>

        {/* Remove button */}
        <button
          onClick={onRemove}
          className="p-0.5 rounded hover:bg-white/10 text-slate-600 dark:text-slate-400 hover:text-slate-800 dark:hover:text-slate-200 transition-colors shrink-0"
          title="Remove feature"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>

      {/* Meta line: source cluster-profile + feature label (member meta rev).
          The description tooltip surfaces the enriched meta when present. */}
      {(feature.label || feature.sourceProfileName) && (
        <div className="flex items-center gap-1.5 mt-1 min-w-0">
          {feature.sourceProfileName && (
            <span
              className="shrink-0 max-w-[130px] truncate text-[10px] font-medium text-cyan-300 bg-cyan-500/10 border border-cyan-500/20 rounded px-1 py-px"
              title={`Member of cluster profile "${feature.sourceProfileName}"`}
            >
              {feature.sourceProfileName}
            </span>
          )}
          {feature.label && (
            <p
              className="text-xs text-slate-600 dark:text-slate-400 truncate min-w-0"
              title={
                typeof feature.meta?.['description'] === 'string'
                  ? (feature.meta['description'] as string)
                  : feature.label
              }
            >
              {feature.label}
            </p>
          )}
        </div>
      )}

      {/* Primary Strength slider */}
      <div className="mt-1.5">
        <StrengthSlider
          value={feature.strength}
          onChange={onStrengthChange}
          color={feature.color}
          compact
          disabled={disabled}
        />
      </div>

      {/* Additional Strengths — collapsed behind an expander to save space */}
      <div className="mt-1.5">
        <button
          onClick={() => setShowAdditional((v) => !v)}
          className="flex items-center gap-1 text-[11px] text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
          title="Test multiple strengths for this feature"
        >
          {showAdditional ? (
            <ChevronDown className="w-3 h-3" />
          ) : (
            <ChevronRight className="w-3 h-3" />
          )}
          Additional strengths
          {additionalStrengths.length > 0 && (
            <span className="text-slate-600 dark:text-slate-400">({additionalStrengths.length})</span>
          )}
        </button>

        {showAdditional && (
          <div className="mt-1.5 pl-4">
            <div className="flex flex-wrap items-center gap-1.5">
              {additionalStrengths.map((strength, index) => (
                <div key={index} className="flex items-center gap-0.5">
                  {editingIndex === index ? (
                    <input
                      type="text"
                      value={editValue}
                      onChange={(e) => setEditValue(e.target.value)}
                      onBlur={() => handleStrengthInputBlur(index)}
                      onKeyDown={(e) => handleStrengthInputKeyDown(e, index)}
                      autoFocus
                      className={`w-14 px-1.5 py-0.5 text-xs font-mono text-center rounded border transition-colors bg-white dark:bg-slate-900 focus:outline-none focus:ring-1 focus:ring-emerald-500 ${getStrengthInputClasses(parseFloat(editValue) || null)}`}
                      placeholder="—"
                    />
                  ) : (
                    <button
                      onClick={() => {
                        setEditingIndex(index);
                        setEditValue(strength.toString());
                      }}
                      disabled={disabled}
                      className={`w-14 px-1.5 py-0.5 text-xs font-mono text-center rounded border transition-colors hover:brightness-110 disabled:cursor-not-allowed ${getStrengthInputClasses(strength)}`}
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
                    <X className="w-2.5 h-2.5" />
                  </button>
                </div>
              ))}
              {canAddMore && (
                <button
                  onClick={handleAddStrength}
                  disabled={disabled}
                  className="flex items-center gap-0.5 text-xs text-emerald-400 hover:text-emerald-300 disabled:opacity-50 disabled:cursor-not-allowed"
                  title={`Add strength: ${getNextStrengthInfo().formula} = ${getNextStrengthInfo().value}`}
                >
                  <Plus className="w-3 h-3" />
                  Add
                </button>
              )}
              {additionalStrengths.length === 0 && !canAddMore && (
                <span className="text-xs text-slate-600 italic">Max reached</span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

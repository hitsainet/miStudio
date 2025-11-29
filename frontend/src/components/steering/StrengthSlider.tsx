/**
 * StrengthSlider - Custom slider for steering strength control.
 *
 * Neuronpedia-compatible calibration:
 * - Range: -200 to +200 (raw coefficients)
 * - Values like 0.07 = subtle, 80 = strong effect
 * - Color-coded zones (normal, caution, extreme)
 * - Visual warning indicators
 * - Precise value input (supports decimals)
 * - Shows coefficient value
 */

import { useState, useRef, useEffect } from 'react';
import { AlertTriangle, AlertCircle } from 'lucide-react';
import {
  STRENGTH_THRESHOLDS,
  getStrengthWarningLevel,
  FeatureColor,
  FEATURE_COLORS,
} from '../../types/steering';

interface StrengthSliderProps {
  value: number;
  onChange: (value: number) => void;
  color?: FeatureColor;
  disabled?: boolean;
  compact?: boolean;
}

export function StrengthSlider({
  value,
  onChange,
  color = 'teal',
  disabled = false,
  compact = false,
}: StrengthSliderProps) {
  const [inputValue, setInputValue] = useState(value.toString());
  const [isEditing, setIsEditing] = useState(false);
  const sliderRef = useRef<HTMLDivElement>(null);

  const warningLevel = getStrengthWarningLevel(value);

  // Sync input value with prop
  useEffect(() => {
    if (!isEditing) {
      setInputValue(value.toString());
    }
  }, [value, isEditing]);

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setInputValue(e.target.value);
  };

  const handleInputBlur = () => {
    setIsEditing(false);
    const parsed = parseFloat(inputValue);
    if (!isNaN(parsed)) {
      const clamped = Math.max(-200, Math.min(200, parsed));
      onChange(clamped);
      setInputValue(clamped.toString());
    } else {
      setInputValue(value.toString());
    }
  };

  const handleInputKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      handleInputBlur();
    }
  };

  const handleSliderClick = (e: React.MouseEvent) => {
    if (disabled || !sliderRef.current) return;
    const rect = sliderRef.current.getBoundingClientRect();
    const percent = (e.clientX - rect.left) / rect.width;
    const newValue = Math.round(-200 + percent * 400); // -200 to 200
    onChange(Math.max(-200, Math.min(200, newValue)));
  };

  const handleSliderDrag = (e: React.MouseEvent) => {
    if (e.buttons !== 1 || disabled || !sliderRef.current) return;
    handleSliderClick(e);
  };

  // Calculate thumb position (0-100%) - range is -200 to +200 (400 total)
  const thumbPosition = ((value + 200) / 400) * 100;

  // Calculate zone positions for the -200 to +200 range
  const zonePositions = {
    extremeLow: ((STRENGTH_THRESHOLDS.EXTREME_LOW + 200) / 400) * 100,
    cautionLow: ((STRENGTH_THRESHOLDS.CAUTION_LOW + 200) / 400) * 100,
    cautionHigh: ((STRENGTH_THRESHOLDS.CAUTION_HIGH + 200) / 400) * 100,
    extremeHigh: ((STRENGTH_THRESHOLDS.EXTREME_HIGH + 200) / 400) * 100,
  };

  const colorClasses = FEATURE_COLORS[color];

  return (
    <div className={`${compact ? 'space-y-1' : 'space-y-2'}`}>
      {/* Header with value and multiplier */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {/* Editable value input */}
          <input
            type="text"
            value={inputValue}
            onChange={handleInputChange}
            onFocus={() => setIsEditing(true)}
            onBlur={handleInputBlur}
            onKeyDown={handleInputKeyDown}
            disabled={disabled}
            className={`w-16 px-2 py-0.5 text-sm font-mono text-center rounded border transition-colors ${
              warningLevel === 'extreme'
                ? 'bg-red-500/10 border-red-500/50 text-red-400'
                : warningLevel === 'caution'
                ? 'bg-amber-500/10 border-amber-500/50 text-amber-400'
                : 'bg-slate-900 border-slate-700 text-slate-100'
            } focus:outline-none focus:ring-1 focus:ring-emerald-500 disabled:opacity-50`}
          />
          <span className="text-xs text-slate-500">strength</span>
        </div>

        <div className="flex items-center gap-2">
          {/* Coefficient display - shows the raw steering coefficient */}
          <span
            className={`text-sm font-mono ${
              warningLevel === 'extreme'
                ? 'text-red-400'
                : warningLevel === 'caution'
                ? 'text-amber-400'
                : 'text-slate-400'
            }`}
            title="Raw steering coefficient (Neuronpedia-compatible)"
          >
            coef: {value}
          </span>

          {/* Warning icons */}
          {warningLevel === 'extreme' && (
            <span title="Extreme strength - may cause incoherent output">
              <AlertCircle className="w-4 h-4 text-red-400" />
            </span>
          )}
          {warningLevel === 'caution' && (
            <span title="High strength - use with caution">
              <AlertTriangle className="w-4 h-4 text-amber-400" />
            </span>
          )}
        </div>
      </div>

      {/* Slider */}
      <div
        ref={sliderRef}
        className={`relative ${compact ? 'h-4' : 'h-6'} rounded-full cursor-pointer select-none ${
          disabled ? 'opacity-50 cursor-not-allowed' : ''
        }`}
        onClick={handleSliderClick}
        onMouseMove={handleSliderDrag}
      >
        {/* Background track with zones */}
        <div className="absolute inset-0 rounded-full overflow-hidden">
          {/* Extreme low zone (red) */}
          <div
            className="absolute h-full bg-red-500/30"
            style={{ left: 0, width: `${zonePositions.extremeLow}%` }}
          />
          {/* Caution low zone (amber) */}
          <div
            className="absolute h-full bg-amber-500/20"
            style={{ left: `${zonePositions.extremeLow}%`, width: `${zonePositions.cautionLow - zonePositions.extremeLow}%` }}
          />
          {/* Normal zone (slate) */}
          <div
            className="absolute h-full bg-slate-700"
            style={{ left: `${zonePositions.cautionLow}%`, width: `${zonePositions.cautionHigh - zonePositions.cautionLow}%` }}
          />
          {/* Caution high zone (amber) */}
          <div
            className="absolute h-full bg-amber-500/20"
            style={{ left: `${zonePositions.cautionHigh}%`, width: `${zonePositions.extremeHigh - zonePositions.cautionHigh}%` }}
          />
          {/* Extreme high zone (red) */}
          <div
            className="absolute h-full bg-red-500/30"
            style={{ left: `${zonePositions.extremeHigh}%`, right: 0 }}
          />
        </div>

        {/* Filled track (colored based on feature) */}
        <div
          className={`absolute h-full rounded-l-full ${colorClasses.bg} opacity-60`}
          style={{ left: 0, width: `${thumbPosition}%` }}
        />

        {/* Zero line marker - at 50% for -200 to +200 range */}
        <div
          className="absolute w-0.5 h-full bg-slate-500"
          style={{ left: '50%' }}
          title="Baseline (strength = 0)"
        />

        {/* Thumb */}
        <div
          className={`absolute top-1/2 -translate-y-1/2 ${compact ? 'w-4 h-4' : 'w-5 h-5'} rounded-full shadow-lg border-2 transition-transform ${
            warningLevel === 'extreme'
              ? 'bg-red-500 border-red-400'
              : warningLevel === 'caution'
              ? 'bg-amber-500 border-amber-400'
              : `${colorClasses.bg} ${colorClasses.border}`
          }`}
          style={{ left: `${thumbPosition}%`, transform: 'translate(-50%, -50%)' }}
        />
      </div>

      {/* Scale labels */}
      {!compact && (
        <div className="flex justify-between text-xs text-slate-500">
          <span>-200</span>
          <span>-100</span>
          <span>0</span>
          <span>100</span>
          <span>200</span>
        </div>
      )}

      {/* Warning message */}
      {warningLevel !== 'normal' && !compact && (
        <div
          className={`text-xs px-2 py-1 rounded ${
            warningLevel === 'extreme'
              ? 'bg-red-500/10 text-red-400'
              : 'bg-amber-500/10 text-amber-400'
          }`}
        >
          {warningLevel === 'extreme'
            ? 'Extreme strength may cause incoherent or repetitive output'
            : 'High strength - monitor output quality'}
        </div>
      )}
    </div>
  );
}

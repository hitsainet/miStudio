/**
 * GenerationConfig - Configuration panel for text generation parameters.
 *
 * Features:
 * - Basic params: max tokens, temperature, top_p, top_k, num_samples
 * - Advanced params (collapsible): repetition penalty, presence penalty, etc.
 * - Seed input for reproducibility
 * - Reset to defaults button
 */

import { useState } from 'react';
import { Settings, ChevronDown, ChevronUp, RotateCcw } from 'lucide-react';
import { useSteeringStore } from '../../stores/steeringStore';
import { COMPONENTS } from '../../config/brand';

interface GenerationConfigProps {
  compact?: boolean;
}

export function GenerationConfig({ compact = false }: GenerationConfigProps) {
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { generationParams, advancedParams, setGenerationParams, setAdvancedParams, resetParams } =
    useSteeringStore();

  const handleParamChange = (key: string, value: string | number) => {
    setGenerationParams({ [key]: value });
  };

  const handleAdvancedChange = (key: string, value: string | number | boolean) => {
    setAdvancedParams({ [key]: value });
  };

  return (
    <div className={`${COMPONENTS.card.base} ${compact ? 'p-3' : 'p-4'} space-y-4`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className={`font-semibold text-slate-100 flex items-center gap-2 ${compact ? 'text-sm' : 'text-base'}`}>
          <Settings className="w-4 h-4" />
          Generation Config
        </h3>
        <button
          onClick={resetParams}
          className="text-xs text-slate-500 hover:text-slate-300 flex items-center gap-1"
          title="Reset to defaults"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>
      </div>

      {/* Basic parameters */}
      <div className={`grid ${compact ? 'grid-cols-2' : 'grid-cols-3'} gap-3`}>
        {/* Max Tokens */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Max Tokens</label>
          <input
            type="number"
            min="1"
            max="2048"
            value={generationParams.max_new_tokens}
            onChange={(e) => handleParamChange('max_new_tokens', parseInt(e.target.value, 10))}
            className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Temperature */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Temperature</label>
          <input
            type="number"
            min="0"
            max="2"
            step="0.1"
            value={generationParams.temperature}
            onChange={(e) => handleParamChange('temperature', parseFloat(e.target.value))}
            className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Top P */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Top P</label>
          <input
            type="number"
            min="0"
            max="1"
            step="0.05"
            value={generationParams.top_p}
            onChange={(e) => handleParamChange('top_p', parseFloat(e.target.value))}
            className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Top K */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Top K</label>
          <input
            type="number"
            min="1"
            max="100"
            value={generationParams.top_k}
            onChange={(e) => handleParamChange('top_k', parseInt(e.target.value, 10))}
            className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Num Samples */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Samples</label>
          <input
            type="number"
            min="1"
            max="5"
            value={generationParams.num_samples}
            onChange={(e) => handleParamChange('num_samples', parseInt(e.target.value, 10))}
            className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
          />
        </div>

        {/* Seed */}
        <div>
          <label className="block text-xs text-slate-400 mb-1">Seed (optional)</label>
          <input
            type="number"
            placeholder="Random"
            value={generationParams.seed ?? ''}
            onChange={(e) =>
              handleParamChange('seed', e.target.value ? parseInt(e.target.value, 10) : undefined as any)
            }
            className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-emerald-500"
          />
        </div>
      </div>

      {/* Advanced toggle */}
      <button
        onClick={() => setShowAdvanced(!showAdvanced)}
        className="w-full py-1.5 text-xs text-slate-400 hover:text-slate-300 flex items-center justify-center gap-1 border-t border-slate-800 pt-3"
      >
        {showAdvanced ? (
          <>
            <ChevronUp className="w-3 h-3" />
            Hide Advanced Settings
          </>
        ) : (
          <>
            <ChevronDown className="w-3 h-3" />
            Show Advanced Settings
          </>
        )}
      </button>

      {/* Advanced parameters */}
      {showAdvanced && (
        <div className={`grid ${compact ? 'grid-cols-2' : 'grid-cols-3'} gap-3 pt-2`}>
          {/* Repetition Penalty */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Repetition Penalty</label>
            <input
              type="number"
              min="1"
              max="2"
              step="0.1"
              value={advancedParams?.repetition_penalty ?? 1.15}
              onChange={(e) => handleAdvancedChange('repetition_penalty', parseFloat(e.target.value))}
              className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Presence Penalty */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Presence Penalty</label>
            <input
              type="number"
              min="-2"
              max="2"
              step="0.1"
              value={advancedParams?.presence_penalty ?? 0}
              onChange={(e) => handleAdvancedChange('presence_penalty', parseFloat(e.target.value))}
              className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Frequency Penalty */}
          <div>
            <label className="block text-xs text-slate-400 mb-1">Frequency Penalty</label>
            <input
              type="number"
              min="-2"
              max="2"
              step="0.1"
              value={advancedParams?.frequency_penalty ?? 0}
              onChange={(e) => handleAdvancedChange('frequency_penalty', parseFloat(e.target.value))}
              className="w-full px-2 py-1.5 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Do Sample */}
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="do-sample"
              checked={advancedParams?.do_sample ?? true}
              onChange={(e) => handleAdvancedChange('do_sample', e.target.checked)}
              className="w-4 h-4 rounded border-slate-700 bg-slate-900 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-950"
            />
            <label htmlFor="do-sample" className="text-xs text-slate-400">
              Enable Sampling
            </label>
          </div>
        </div>
      )}

      {/* Quick info */}
      {!compact && (
        <div className="text-xs text-slate-500 pt-2 border-t border-slate-800">
          <p>
            <strong>Temperature</strong>: Controls randomness. Higher = more creative.
          </p>
          <p className="mt-1">
            <strong>Top P/K</strong>: Limits token selection for coherence.
          </p>
        </div>
      )}
    </div>
  );
}

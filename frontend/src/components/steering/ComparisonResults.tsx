/**
 * ComparisonResults - Display steering comparison results.
 *
 * Features:
 * - Side-by-side display of unsteered and steered outputs
 * - Color-coded feature cards
 * - Generation metrics (perplexity, coherence, etc.)
 * - Copy output button
 * - Save experiment button
 */

import { useState } from 'react';
import { Copy, Check, Save, Clock, Hash, Layers, Zap } from 'lucide-react';
import {
  SteeringComparisonResponse,
  SteeredOutput,
  UnsteeredOutput,
  GenerationMetrics,
  FEATURE_COLORS,
} from '../../types/steering';
import { useSteeringStore } from '../../stores/steeringStore';
import { COMPONENTS } from '../../config/brand';

interface ComparisonResultsProps {
  comparison: SteeringComparisonResponse;
  onSaveExperiment?: () => void;
}

export function ComparisonResults({ comparison, onSaveExperiment }: ComparisonResultsProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const { selectedFeatures } = useSteeringStore();

  const handleCopy = async (text: string, id: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
    }
  };

  const formatTime = (ms: number): string => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const renderMetrics = (metrics: GenerationMetrics | null) => {
    if (!metrics) return null;

    return (
      <div className="flex items-center gap-4 text-xs text-slate-500 mt-2">
        {metrics.perplexity !== null && (
          <span title="Perplexity">PPL: {metrics.perplexity.toFixed(2)}</span>
        )}
        {metrics.coherence !== null && (
          <span title="Coherence Score">Coh: {metrics.coherence.toFixed(2)}</span>
        )}
        {metrics.token_count !== undefined && (
          <span title="Token Count">{metrics.token_count} tokens</span>
        )}
        <span title="Generation Time" className="flex items-center gap-1">
          <Clock className="w-3 h-3" />
          {formatTime(metrics.generation_time_ms)}
        </span>
      </div>
    );
  };

  const renderOutput = (
    output: UnsteeredOutput | SteeredOutput,
    title: string,
    id: string
  ) => {
    const isSteered = 'feature_config' in output;
    const featureConfig = isSteered ? (output as SteeredOutput).feature_config : null;
    const featureColor = featureConfig?.color;
    const colorClasses = featureColor ? FEATURE_COLORS[featureColor] : null;

    return (
      <div
        className={`${COMPONENTS.card.base} p-4 ${
          colorClasses ? `border-l-4 ${colorClasses.border}` : ''
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            {colorClasses && (
              <div className={`w-3 h-3 rounded-full ${colorClasses.bg}`} />
            )}
            <h4 className={`font-medium ${colorClasses ? colorClasses.text : 'text-slate-300'}`}>
              {title}
            </h4>
          </div>
          <button
            onClick={() => handleCopy(output.text, id)}
            className="p-1.5 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
            title="Copy output"
          >
            {copiedId === id ? (
              <Check className="w-4 h-4 text-emerald-400" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </button>
        </div>

        {/* Feature config for steered outputs */}
        {featureConfig && (
          <div className="flex items-center gap-3 mb-3 text-sm">
            <span className={`flex items-center gap-1 ${colorClasses?.text || 'text-slate-400'}`}>
              <Hash className="w-3.5 h-3.5" />
              {featureConfig.feature_idx}
            </span>
            <span className="text-slate-500">•</span>
            <span className="flex items-center gap-1 text-slate-400">
              <Layers className="w-3.5 h-3.5" />
              L{featureConfig.layer}
            </span>
            <span className="text-slate-500">•</span>
            <span className={`flex items-center gap-1 ${
              featureConfig.strength > 150 ? 'text-amber-400' :
              featureConfig.strength > 250 ? 'text-red-400' : 'text-slate-400'
            }`}>
              <Zap className="w-3.5 h-3.5" />
              {featureConfig.strength > 0 ? '+' : ''}{featureConfig.strength}
            </span>
          </div>
        )}

        {/* Output text */}
        <div className="bg-slate-900/50 rounded-lg p-3">
          <p className="text-slate-200 whitespace-pre-wrap font-mono text-sm leading-relaxed">
            {output.text}
          </p>
        </div>

        {/* Metrics */}
        {renderMetrics(output.metrics)}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Header with summary */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">Comparison Results</h3>
          <p className="text-sm text-slate-400 mt-1">
            Generated in {formatTime(comparison.total_time_ms)} •{' '}
            {comparison.steered.length} steered output{comparison.steered.length !== 1 ? 's' : ''}
          </p>
        </div>
        {onSaveExperiment && (
          <button
            onClick={onSaveExperiment}
            className={`px-4 py-2 flex items-center gap-2 ${COMPONENTS.button.secondary}`}
          >
            <Save className="w-4 h-4" />
            Save Experiment
          </button>
        )}
      </div>

      {/* Prompt display */}
      <div className={`${COMPONENTS.card.base} p-3`}>
        <span className="text-xs text-slate-500 uppercase tracking-wide">Prompt</span>
        <p className="text-slate-200 mt-1">{comparison.prompt}</p>
      </div>

      {/* Results grid */}
      <div className="grid gap-4">
        {/* Unsteered baseline */}
        {comparison.unsteered && (
          renderOutput(comparison.unsteered, 'Baseline (Unsteered)', 'unsteered')
        )}

        {/* Steered outputs */}
        {comparison.steered.map((output, index) => {
          // Find matching selected feature for color
          const matchingFeature = selectedFeatures.find(
            (f) =>
              f.feature_idx === output.feature_config.feature_idx &&
              f.layer === output.feature_config.layer
          );
          const displayTitle = matchingFeature?.label || `Feature #${output.feature_config.feature_idx}`;

          return renderOutput(
            output,
            displayTitle,
            `steered-${index}`
          );
        })}
      </div>

      {/* Metrics summary */}
      {comparison.metrics_summary && Object.keys(comparison.metrics_summary).length > 0 && (
        <div className={`${COMPONENTS.card.base} p-4`}>
          <h4 className="text-sm font-medium text-slate-300 mb-2">Metrics Summary</h4>
          <pre className="text-xs text-slate-400 bg-slate-900/50 p-3 rounded overflow-x-auto">
            {JSON.stringify(comparison.metrics_summary, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

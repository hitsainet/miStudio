/**
 * ComparisonResults - Display steering comparison results.
 *
 * Features:
 * - Side-by-side display of unsteered and steered outputs
 * - Multi-strength results display (when additional_strengths provided)
 * - Batch results display (multiple prompts)
 * - Color-coded feature cards
 * - Generation metrics (perplexity, coherence, etc.)
 * - Copy output button on all tiles
 * - Copy all button to copy all outputs
 * - Save experiment button
 */

import { useState, useCallback } from 'react';
import { Copy, Check, Save, Clock, Hash, Layers, Zap, CopyPlus, Trash2, AlertCircle, CheckCircle } from 'lucide-react';
import {
  SteeringComparisonResponse,
  SteeredOutput,
  SteeredOutputMulti,
  UnsteeredOutput,
  GenerationMetrics,
  FEATURE_COLORS,
  getStrengthWarningLevel,
  BatchPromptResult,
} from '../../types/steering';
import { useSteeringStore } from '../../stores/steeringStore';
import { COMPONENTS } from '../../config/brand';

interface ComparisonResultsProps {
  comparison?: SteeringComparisonResponse;
  batchResults?: BatchPromptResult[];
  onSaveExperiment?: () => void;
  onClearBatchResults?: () => void;
  isRunning?: boolean; // True if batch is still processing
}

export function ComparisonResults({ comparison, batchResults, onSaveExperiment, onClearBatchResults, isRunning = false }: ComparisonResultsProps) {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const { selectedFeatures, generationParams, advancedParams, selectedSAE } = useSteeringStore();

  // Determine mode: batch vs single
  const isBatchMode = batchResults && batchResults.length > 0;

  // Check if this is multi-strength mode (only for single comparison mode)
  const isMultiStrengthMode = comparison?.steered_multi && comparison.steered_multi.length > 0;

  const handleCopy = useCallback(async (text: string, id: string) => {
    if (!text) {
      console.warn('[ComparisonResults] No text to copy for id:', id);
      return;
    }
    try {
      await navigator.clipboard.writeText(text);
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    } catch (error) {
      console.error('Failed to copy:', error);
      // Fallback for older browsers
      try {
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        document.body.appendChild(textArea);
        textArea.select();
        document.execCommand('copy');
        document.body.removeChild(textArea);
        setCopiedId(id);
        setTimeout(() => setCopiedId(null), 2000);
      } catch (fallbackError) {
        console.error('Fallback copy also failed:', fallbackError);
      }
    }
  }, []);

  // Build content for a single comparison result
  const buildSingleComparisonContent = useCallback((comp: SteeringComparisonResponse, promptIndex?: number) => {
    const lines: string[] = [];

    if (promptIndex !== undefined) {
      lines.push(`=== PROMPT ${promptIndex + 1} ===`);
    } else {
      lines.push('=== PROMPT ===');
    }
    lines.push(comp.prompt);
    lines.push('');

    // Add unsteered baseline
    if (comp.unsteered) {
      lines.push('--- BASELINE (Unsteered) ---');
      lines.push(comp.unsteered.text);
      if (comp.unsteered.metrics?.perplexity) {
        lines.push(`Perplexity: ${comp.unsteered.metrics.perplexity.toFixed(2)}`);
      }
      lines.push('');
    }

    // Check if this comparison has multi-strength results
    const hasMultiStrength = comp.steered_multi && comp.steered_multi.length > 0;

    if (hasMultiStrength && comp.steered_multi) {
      // Multi-strength mode: show all results for each feature
      comp.steered_multi.forEach((multiOutput) => {
        const featureConfig = multiOutput.feature_config;
        const matchingFeature = selectedFeatures.find(
          (f) =>
            f.feature_idx === featureConfig.feature_idx &&
            f.layer === featureConfig.layer
        );
        const label = matchingFeature?.label || `Feature #${featureConfig.feature_idx}`;

        lines.push(`--- ${label} [#${featureConfig.feature_idx}] (Layer ${featureConfig.layer}) - Multi-Strength ---`);
        lines.push('');

        // Combine primary and additional results, sorted by absolute strength
        const allResults = [
          multiOutput.primary_result,
          ...multiOutput.additional_results,
        ].sort((a, b) => Math.abs(a.strength) - Math.abs(b.strength));

        allResults.forEach((result) => {
          const isPrimary = result.strength === featureConfig.strength;
          lines.push(`  [Strength ${result.strength > 0 ? '+' : ''}${result.strength}${isPrimary ? ' (Primary)' : ''}]`);
          lines.push(`  ${result.text}`);
          if (result.metrics?.perplexity) {
            lines.push(`  Perplexity: ${result.metrics.perplexity.toFixed(2)}`);
          }
          lines.push('');
        });
      });
    } else {
      // Single-strength mode: original steered outputs
      comp.steered.forEach((output) => {
        const matchingFeature = selectedFeatures.find(
          (f) =>
            f.feature_idx === output.feature_config.feature_idx &&
            f.layer === output.feature_config.layer
        );
        const label = matchingFeature?.label || `Feature #${output.feature_config.feature_idx}`;

        lines.push(`--- ${label} [#${output.feature_config.feature_idx}] (L${output.feature_config.layer}, Strength ${output.feature_config.strength > 0 ? '+' : ''}${output.feature_config.strength}) ---`);
        lines.push(output.text);
        if (output.metrics?.perplexity) {
          lines.push(`Perplexity: ${output.metrics.perplexity.toFixed(2)}`);
        }
        lines.push('');
      });
    }

    return lines;
  }, [selectedFeatures]);

  // Build all content for "copy all" feature
  const buildAllContent = useCallback(() => {
    const lines: string[] = [];

    // Add configuration section
    lines.push('=== CONFIGURATION ===');
    if (selectedSAE) {
      lines.push(`SAE: ${selectedSAE.name || selectedSAE.id}`);
    }

    // Use first available model_id
    const modelId = isBatchMode && batchResults?.[0]?.comparison
      ? batchResults[0].comparison.model_id
      : comparison?.model_id;
    if (modelId) {
      lines.push(`Model: ${modelId}`);
    }
    lines.push(`Temperature: ${generationParams.temperature}`);
    lines.push(`Top P: ${generationParams.top_p}`);
    lines.push(`Top K: ${generationParams.top_k}`);
    lines.push(`Max Tokens: ${generationParams.max_new_tokens}`);
    if (generationParams.seed !== undefined) {
      lines.push(`Seed: ${generationParams.seed}`);
    }
    // Add advanced params if set
    if (advancedParams) {
      lines.push('');
      lines.push('[Advanced Parameters]');
      lines.push(`Repetition Penalty: ${advancedParams.repetition_penalty}`);
      lines.push(`Presence Penalty: ${advancedParams.presence_penalty}`);
      lines.push(`Frequency Penalty: ${advancedParams.frequency_penalty}`);
      lines.push(`Do Sample: ${advancedParams.do_sample}`);
      if (advancedParams.stop_sequences && advancedParams.stop_sequences.length > 0) {
        lines.push(`Stop Sequences: ${advancedParams.stop_sequences.join(', ')}`);
      }
    }
    lines.push('');

    // Handle batch mode vs single comparison mode
    if (isBatchMode && batchResults) {
      // Batch mode: iterate through all prompt results
      lines.push(`=== BATCH RESULTS (${batchResults.length} prompts) ===`);
      lines.push('');

      batchResults.forEach((result, index) => {
        if (result.status === 'completed' && result.comparison) {
          lines.push(...buildSingleComparisonContent(result.comparison, index));
        } else if (result.status === 'failed') {
          lines.push(`=== PROMPT ${index + 1} (FAILED) ===`);
          lines.push(result.prompt);
          lines.push(`Error: ${result.error || 'Unknown error'}`);
          lines.push('');
        }
      });
    } else if (comparison) {
      // Single comparison mode
      // Add prompt
      lines.push('=== PROMPT ===');
      lines.push(comparison.prompt);
      lines.push('');

      // Add unsteered baseline
      if (comparison.unsteered) {
        lines.push('=== BASELINE (Unsteered) ===');
        lines.push(comparison.unsteered.text);
        if (comparison.unsteered.metrics?.perplexity) {
          lines.push(`Perplexity: ${comparison.unsteered.metrics.perplexity.toFixed(2)}`);
        }
        lines.push('');
      }

      if (isMultiStrengthMode && comparison.steered_multi) {
        // Multi-strength mode: show all results for each feature
        comparison.steered_multi.forEach((multiOutput) => {
          const featureConfig = multiOutput.feature_config;
          const matchingFeature = selectedFeatures.find(
            (f) =>
              f.feature_idx === featureConfig.feature_idx &&
              f.layer === featureConfig.layer
          );
          const label = matchingFeature?.label || `Feature #${featureConfig.feature_idx}`;
          const featureIdx = featureConfig.feature_idx;

          lines.push(`=== ${label.toUpperCase()} [#${featureIdx}] (Layer ${featureConfig.layer}) - Multi-Strength ===`);
          lines.push('');

          // Combine primary and additional results, sorted by absolute strength
          const allResults = [
            multiOutput.primary_result,
            ...multiOutput.additional_results,
          ].sort((a, b) => Math.abs(a.strength) - Math.abs(b.strength));

          allResults.forEach((result) => {
            const isPrimary = result.strength === featureConfig.strength;
            lines.push(`--- Strength ${result.strength > 0 ? '+' : ''}${result.strength}${isPrimary ? ' (Primary)' : ''} ---`);
            lines.push(result.text);
            if (result.metrics?.perplexity) {
              lines.push(`Perplexity: ${result.metrics.perplexity.toFixed(2)}`);
            }
            lines.push('');
          });
        });
      } else {
        // Single-strength mode
        comparison.steered.forEach((output) => {
          const matchingFeature = selectedFeatures.find(
            (f) =>
              f.feature_idx === output.feature_config.feature_idx &&
              f.layer === output.feature_config.layer
          );
          const label = matchingFeature?.label || `Feature #${output.feature_config.feature_idx}`;
          const featureIdx = output.feature_config.feature_idx;

          lines.push(`=== ${label.toUpperCase()} [#${featureIdx}] (Layer ${output.feature_config.layer}, Strength ${output.feature_config.strength > 0 ? '+' : ''}${output.feature_config.strength}) ===`);
          lines.push(output.text);
          if (output.metrics?.perplexity) {
            lines.push(`Perplexity: ${output.metrics.perplexity.toFixed(2)}`);
          }
          lines.push('');
        });
      }
    }

    return lines.join('\n');
  }, [comparison, batchResults, isBatchMode, selectedFeatures, isMultiStrengthMode, generationParams, advancedParams, selectedSAE, buildSingleComparisonContent]);

  const handleCopyAll = useCallback(async () => {
    const allContent = buildAllContent();
    await handleCopy(allContent, 'all');
  }, [buildAllContent, handleCopy]);

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

  // Render multi-strength results for a single feature
  const renderMultiStrengthFeature = (multiOutput: SteeredOutputMulti, featureIndex: number) => {
    const featureConfig = multiOutput.feature_config;
    const colorClasses = FEATURE_COLORS[featureConfig.color];

    // Find matching selected feature for label
    const matchingFeature = selectedFeatures.find(
      (f) =>
        f.feature_idx === featureConfig.feature_idx &&
        f.layer === featureConfig.layer
    );
    const displayTitle = matchingFeature?.label || `Feature #${featureConfig.feature_idx}`;

    // Combine primary and additional results, sorted by absolute strength
    const allResults = [
      multiOutput.primary_result,
      ...multiOutput.additional_results,
    ].sort((a, b) => Math.abs(a.strength) - Math.abs(b.strength));

    return (
      <div key={`multi-${featureIndex}`} className={`${COMPONENTS.card.base} p-4 border-l-4 ${colorClasses.border}`}>
        {/* Feature Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${colorClasses.bg}`} />
            <h4 className={`font-medium ${colorClasses.text}`}>{displayTitle}</h4>
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <span className="flex items-center gap-1">
                <Hash className="w-3.5 h-3.5" />
                {featureConfig.feature_idx}
              </span>
              <span>•</span>
              <span className="flex items-center gap-1">
                <Layers className="w-3.5 h-3.5" />
                L{featureConfig.layer}
              </span>
            </div>
            <span className="text-xs bg-emerald-500/20 text-emerald-400 px-2 py-0.5 rounded">
              {allResults.length} strengths
            </span>
          </div>
        </div>

        {/* Results Stack - Vertical layout, full width */}
        <div className="flex flex-col gap-3">
          {allResults.map((result, idx) => {
            const isPrimary = result.strength === featureConfig.strength;
            const warningLevel = getStrengthWarningLevel(result.strength);

            return (
              <div
                key={idx}
                className={`rounded-lg p-3 w-full ${
                  isPrimary
                    ? `bg-${featureConfig.color}-500/10 border border-${featureConfig.color}-500/30`
                    : 'bg-slate-900/50'
                }`}
              >
                {/* Strength Header */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-3">
                    <span className={`text-xs ${isPrimary ? colorClasses.text : 'text-slate-400'}`}>
                      {isPrimary ? 'Primary' : `Strength`}
                    </span>
                    <span className={`font-mono text-sm font-medium ${
                      warningLevel === 'extreme' ? 'text-red-400' :
                      warningLevel === 'caution' ? 'text-amber-400' :
                      'text-slate-300'
                    }`}>
                      {result.strength > 0 ? '+' : ''}{result.strength}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    {/* Metrics inline in header */}
                    {result.metrics && (
                      <div className="flex items-center gap-4 text-xs text-slate-500">
                        <span>PPL: <span className="text-slate-300">{result.metrics.perplexity?.toFixed(1) ?? '—'}</span></span>
                        <span>Coh: <span className="text-slate-300">{result.metrics.coherence?.toFixed(2) ?? '—'}</span></span>
                        <span>Tok: <span className="text-slate-300">{result.metrics.token_count ?? '—'}</span></span>
                      </div>
                    )}
                    <button
                      onClick={() => handleCopy(result.text, `multi-${featureIndex}-${idx}`)}
                      className="p-1 rounded hover:bg-slate-700 text-slate-500 hover:text-slate-300 transition-colors"
                      title="Copy output"
                    >
                      {copiedId === `multi-${featureIndex}-${idx}` ? (
                        <Check className="w-3.5 h-3.5 text-emerald-400" />
                      ) : (
                        <Copy className="w-3.5 h-3.5" />
                      )}
                    </button>
                  </div>
                </div>

                {/* Output Text - fixed height with scroll */}
                <div className="text-sm text-slate-200 h-40 overflow-y-auto font-mono whitespace-pre-wrap bg-slate-950/50 rounded p-2">
                  {result.text}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  // Calculate output count for header
  const outputCount = isBatchMode
    ? batchResults?.filter(r => r.status === 'completed').length || 0
    : isMultiStrengthMode && comparison?.steered_multi
      ? comparison.steered_multi.reduce((acc, m) => acc + 1 + m.additional_results.length, 0)
      : comparison?.steered.length || 0;

  // Calculate batch stats
  const batchStats = isBatchMode && batchResults ? {
    completed: batchResults.filter(r => r.status === 'completed').length,
    failed: batchResults.filter(r => r.status === 'failed').length,
    total: batchResults.length,
  } : null;

  // Render a single batch result (one prompt's comparison)
  const renderBatchResult = (result: BatchPromptResult, index: number) => {
    if (result.status === 'failed') {
      return (
        <div key={`batch-${index}`} className={`${COMPONENTS.card.base} p-4 border-l-4 border-red-500`}>
          <div className="flex items-center gap-2 mb-2">
            <AlertCircle className="w-5 h-5 text-red-400" />
            <h4 className="font-medium text-red-400">Prompt {index + 1} - Failed</h4>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-3 mb-2">
            <p className="text-slate-300 text-sm">{result.prompt}</p>
          </div>
          <p className="text-red-400 text-sm">{result.error || 'Unknown error'}</p>
        </div>
      );
    }

    // Show "processing" indicator for currently running prompt
    if (result.status === 'running') {
      return (
        <div key={`batch-${index}`} className={`${COMPONENTS.card.base} p-4 border-l-4 border-emerald-500 animate-pulse`}>
          <div className="flex items-center gap-2 mb-2">
            <div className="w-4 h-4 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            <h4 className="font-medium text-emerald-400">Prompt {index + 1} - Processing...</h4>
          </div>
          <div className="bg-slate-900/50 rounded-lg p-3">
            <p className="text-slate-300 text-sm">{result.prompt}</p>
          </div>
        </div>
      );
    }

    // Skip pending prompts (not started yet)
    if (result.status !== 'completed' || !result.comparison) {
      return null;
    }

    const comp = result.comparison;
    // Check if this comparison has multi-strength results
    const hasMultiStrength = comp.steered_multi && comp.steered_multi.length > 0;

    return (
      <div key={`batch-${index}`} className="space-y-3">
        {/* Prompt header */}
        <div className={`${COMPONENTS.card.base} p-3 border-l-4 border-emerald-500`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-emerald-400" />
              <span className="text-sm font-medium text-emerald-400">Prompt {index + 1}</span>
              <span className="text-xs text-slate-500">({formatTime(comp.total_time_ms)})</span>
            </div>
            <button
              onClick={() => handleCopy(comp.prompt, `batch-prompt-${index}`)}
              className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
              title="Copy prompt"
            >
              {copiedId === `batch-prompt-${index}` ? (
                <Check className="w-3.5 h-3.5 text-emerald-400" />
              ) : (
                <Copy className="w-3.5 h-3.5" />
              )}
            </button>
          </div>
          <p className="text-slate-200 mt-2">{comp.prompt}</p>
        </div>

        {/* Unsteered baseline */}
        {comp.unsteered && renderOutput(comp.unsteered, 'Baseline (Unsteered)', `batch-${index}-unsteered`)}

        {/* Steered outputs - handle both multi-strength and single-strength modes */}
        {hasMultiStrength && comp.steered_multi ? (
          /* Multi-strength mode */
          comp.steered_multi.map((multiOutput, featureIndex) =>
            renderMultiStrengthFeature(multiOutput, featureIndex)
          )
        ) : (
          /* Single-strength mode */
          comp.steered.map((output, outputIndex) => {
            const matchingFeature = selectedFeatures.find(
              (f) =>
                f.feature_idx === output.feature_config.feature_idx &&
                f.layer === output.feature_config.layer
            );
            const displayTitle = matchingFeature?.label || `Feature #${output.feature_config.feature_idx}`;
            return renderOutput(output, displayTitle, `batch-${index}-steered-${outputIndex}`);
          })
        )}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Header with summary */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-slate-100">
            {isBatchMode ? (isRunning ? 'Batch Results (Processing...)' : 'Batch Results') : 'Comparison Results'}
          </h3>
          <p className="text-sm text-slate-400 mt-1">
            {isBatchMode && batchStats ? (
              <>
                {batchStats.completed} of {batchStats.total} prompts completed
                {batchStats.failed > 0 && (
                  <span className="text-red-400"> • {batchStats.failed} failed</span>
                )}
                {isRunning && batchStats.completed < batchStats.total && (
                  <span className="text-emerald-400"> • Processing next prompt...</span>
                )}
              </>
            ) : comparison ? (
              <>
                Generated in {formatTime(comparison.total_time_ms)} •{' '}
                {outputCount} steered output{outputCount !== 1 ? 's' : ''}
                {isMultiStrengthMode && ' (multi-strength mode)'}
              </>
            ) : null}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {isBatchMode && onClearBatchResults && (
            <button
              onClick={onClearBatchResults}
              className={`px-4 py-2 flex items-center gap-2 ${COMPONENTS.button.ghost} text-red-400 hover:text-red-300`}
              title="Clear batch results"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          )}
          <button
            onClick={handleCopyAll}
            className={`px-4 py-2 flex items-center gap-2 ${COMPONENTS.button.ghost}`}
            title="Copy all outputs to clipboard"
          >
            {copiedId === 'all' ? (
              <>
                <Check className="w-4 h-4 text-emerald-400" />
                Copied!
              </>
            ) : (
              <>
                <CopyPlus className="w-4 h-4" />
                Copy All
              </>
            )}
          </button>
          {!isBatchMode && onSaveExperiment && (
            <button
              onClick={onSaveExperiment}
              className={`px-4 py-2 flex items-center gap-2 ${COMPONENTS.button.secondary}`}
            >
              <Save className="w-4 h-4" />
              Save Experiment
            </button>
          )}
        </div>
      </div>

      {/* Batch mode results */}
      {isBatchMode && batchResults && (
        <div className="space-y-6">
          {batchResults.map((result, index) => renderBatchResult(result, index))}
        </div>
      )}

      {/* Single comparison mode - Prompt display */}
      {!isBatchMode && comparison && (
        <>
          <div className={`${COMPONENTS.card.base} p-3`}>
            <div className="flex items-center justify-between">
              <span className="text-xs text-slate-500 uppercase tracking-wide">Prompt</span>
              <button
                onClick={() => handleCopy(comparison.prompt, 'prompt')}
                className="p-1 rounded hover:bg-slate-800 text-slate-400 hover:text-slate-200 transition-colors"
                title="Copy prompt"
              >
                {copiedId === 'prompt' ? (
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
              </button>
            </div>
            <p className="text-slate-200 mt-1">{comparison.prompt}</p>
          </div>

          {/* Results grid */}
          <div className="grid gap-4">
            {/* Unsteered baseline */}
            {comparison.unsteered && (
              renderOutput(comparison.unsteered, 'Baseline (Unsteered)', 'unsteered')
            )}

            {/* Multi-strength mode: render feature cards with multiple results */}
            {isMultiStrengthMode && comparison.steered_multi ? (
              comparison.steered_multi.map((multiOutput, index) =>
                renderMultiStrengthFeature(multiOutput, index)
              )
            ) : (
              /* Single-strength mode: original steered outputs */
              comparison.steered.map((output, index) => {
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
              })
            )}
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
        </>
      )}
    </div>
  );
}

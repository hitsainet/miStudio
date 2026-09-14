/**
 * ExtractionTemplateForm component for creating and editing extraction templates.
 *
 * Configuration fields match StartExtractionModal exactly:
 * - Evaluation Samples + Top-K grid
 * - Dead Neuron Filtering (slider + number input)
 * - Context Window (collapsible, prefix/suffix inputs)
 * - Token Filtering (collapsible, 6 individual checkboxes)
 * - Auto-NLP Analysis checkbox
 *
 * Filter settings, dead neuron threshold, and auto-NLP are stored in extra_metadata.
 */

import React, { useState, useEffect } from 'react';
import { Save, X, ChevronDown } from 'lucide-react';
import {
  ExtractionTemplate,
  ExtractionTemplateCreate,
  ExtractionTemplateUpdate,
  HookType,
} from '../../types/extractionTemplate';

interface ExtractionTemplateFormProps {
  template?: ExtractionTemplate;
  onSubmit: (data: ExtractionTemplateCreate | ExtractionTemplateUpdate) => Promise<void>;
  onCancel?: () => void;
  className?: string;
}

export function ExtractionTemplateForm({
  template,
  onSubmit,
  onCancel,
  className = '',
}: ExtractionTemplateFormProps) {
  const isEditMode = !!template;

  // Template metadata
  const [name, setName] = useState(template?.name || '');
  const [description, setDescription] = useState(template?.description || '');
  const [layerIndicesInput, setLayerIndicesInput] = useState(
    template?.layer_indices.join(', ') || ''
  );
  const [hookTypes, setHookTypes] = useState<string[]>(template?.hook_types || ['residual']);
  const [isFavorite, setIsFavorite] = useState(template?.is_favorite || false);

  // Extraction config (matching StartExtractionModal defaults)
  const [maxSamples, setMaxSamples] = useState(template?.max_samples || 10000);
  const [batchSize, setBatchSize] = useState(template?.batch_size || 8);
  const [topKExamples, setTopKExamples] = useState(template?.top_k_examples || 100);

  // Context window
  const [contextPrefixTokens, setContextPrefixTokens] = useState(template?.context_prefix_tokens ?? 25);
  const [contextSuffixTokens, setContextSuffixTokens] = useState(template?.context_suffix_tokens ?? 25);
  const [showContextWindow, setShowContextWindow] = useState(false);

  // Token filtering (6 individual checkboxes, stored in extra_metadata)
  const [filterSpecial, setFilterSpecial] = useState(template?.extra_metadata?.filter_special ?? true);
  const [filterSingleChar, setFilterSingleChar] = useState(template?.extra_metadata?.filter_single_char ?? true);
  const [filterPunctuation, setFilterPunctuation] = useState(template?.extra_metadata?.filter_punctuation ?? true);
  const [filterNumbers, setFilterNumbers] = useState(template?.extra_metadata?.filter_numbers ?? true);
  const [filterFragments, setFilterFragments] = useState(template?.extra_metadata?.filter_fragments ?? true);
  const [filterStopWords, setFilterStopWords] = useState(template?.extra_metadata?.filter_stop_words ?? false);
  const [showFilters, setShowFilters] = useState(false);

  // Dead neuron filtering (stored in extra_metadata)
  const [minActivationFrequency, setMinActivationFrequency] = useState(
    template?.extra_metadata?.min_activation_frequency ?? 0.001
  );

  // NLP processing (stored in extra_metadata)
  const [autoNlp, setAutoNlp] = useState(template?.extra_metadata?.auto_nlp ?? true);

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset form when template changes
  useEffect(() => {
    if (template) {
      setName(template.name);
      setDescription(template.description || '');
      setLayerIndicesInput(template.layer_indices.join(', '));
      setHookTypes(template.hook_types as string[]);
      setMaxSamples(template.max_samples);
      setBatchSize(template.batch_size);
      setTopKExamples(template.top_k_examples);
      setIsFavorite(template.is_favorite);
      setContextPrefixTokens(template.context_prefix_tokens ?? 25);
      setContextSuffixTokens(template.context_suffix_tokens ?? 25);

      // Load filter settings from extra_metadata
      const meta = template.extra_metadata || {};
      setFilterSpecial(meta.filter_special ?? true);
      setFilterSingleChar(meta.filter_single_char ?? true);
      setFilterPunctuation(meta.filter_punctuation ?? true);
      setFilterNumbers(meta.filter_numbers ?? true);
      setFilterFragments(meta.filter_fragments ?? true);
      setFilterStopWords(meta.filter_stop_words ?? false);
      setMinActivationFrequency(meta.min_activation_frequency ?? 0.001);
      setAutoNlp(meta.auto_nlp ?? true);
    }
  }, [template]);

  const parseLayerIndices = (input: string): number[] | null => {
    try {
      const trimmed = input.trim();
      if (!trimmed) return null;
      const indices = trimmed
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map((s) => parseInt(s, 10));
      if (indices.some(isNaN)) return null;
      return Array.from(new Set(indices)).sort((a, b) => a - b);
    } catch {
      return null;
    }
  };

  const handleHookTypeToggle = (hookType: string) => {
    setHookTypes((prev) =>
      prev.includes(hookType) ? prev.filter((ht) => ht !== hookType) : [...prev, hookType]
    );
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!name.trim()) {
      setError('Template name is required');
      return;
    }

    const layerIndices = parseLayerIndices(layerIndicesInput);
    if (!layerIndices || layerIndices.length === 0) {
      setError('Please enter valid layer indices (comma-separated numbers, e.g., 0, 5, 11)');
      return;
    }

    if (hookTypes.length === 0) {
      setError('Please select at least one hook type');
      return;
    }

    if (maxSamples < 100 || maxSamples > 1000000) {
      setError('Evaluation samples must be between 100 and 1,000,000');
      return;
    }

    if (topKExamples < 1 || topKExamples > 1000) {
      setError('Top-K examples must be between 1 and 1,000');
      return;
    }

    setIsSubmitting(true);

    try {
      const data = {
        name: name.trim(),
        description: description.trim() || undefined,
        layer_indices: layerIndices,
        hook_types: hookTypes,
        max_samples: maxSamples,
        batch_size: batchSize,
        top_k_examples: topKExamples,
        is_favorite: isFavorite,
        context_prefix_tokens: contextPrefixTokens,
        context_suffix_tokens: contextSuffixTokens,
        extra_metadata: {
          filter_special: filterSpecial,
          filter_single_char: filterSingleChar,
          filter_punctuation: filterPunctuation,
          filter_numbers: filterNumbers,
          filter_fragments: filterFragments,
          filter_stop_words: filterStopWords,
          min_activation_frequency: minActivationFrequency,
          auto_nlp: autoNlp,
        },
      };

      await onSubmit(data);

      // Reset form on success (only in create mode)
      if (!isEditMode) {
        setName('');
        setDescription('');
        setLayerIndicesInput('');
        setHookTypes(['residual']);
        setMaxSamples(10000);
        setBatchSize(8);
        setTopKExamples(100);
        setIsFavorite(false);
        setContextPrefixTokens(25);
        setContextSuffixTokens(25);
        setFilterSpecial(true);
        setFilterSingleChar(true);
        setFilterPunctuation(true);
        setFilterNumbers(true);
        setFilterFragments(true);
        setFilterStopWords(false);
        setMinActivationFrequency(0.001);
        setAutoNlp(true);
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to save template';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={`bg-white dark:bg-slate-900/50 border border-slate-200 dark:border-slate-800 rounded-lg p-6 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">
          {isEditMode ? 'Edit Template' : 'Create Extraction Template'}
        </h2>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-5 h-5 text-slate-600 dark:text-slate-400" />
          </button>
        )}
      </div>

      <div className="space-y-4">
        {/* Name */}
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Template Name <span className="text-red-400">*</span>
          </label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Extraction Template"
            className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            disabled={isSubmitting}
            required
          />
        </div>

        {/* Description */}
        <div>
          <label htmlFor="description" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Description
          </label>
          <textarea
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Brief description of this template's purpose..."
            rows={2}
            className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
            disabled={isSubmitting}
          />
        </div>

        {/* Layer Indices */}
        <div>
          <label htmlFor="layer-indices" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Layer Indices <span className="text-red-400">*</span>
          </label>
          <input
            id="layer-indices"
            type="text"
            value={layerIndicesInput}
            onChange={(e) => setLayerIndicesInput(e.target.value)}
            placeholder="0, 5, 11, 23"
            className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            disabled={isSubmitting}
            required
          />
          <p className="text-xs text-slate-500 mt-1">
            Comma-separated layer indices (e.g., 0, 5, 11, 23)
          </p>
        </div>

        {/* Hook Types */}
        <div>
          <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Hook Types <span className="text-red-400">*</span>
          </label>
          <div className="flex flex-wrap gap-3">
            {Object.values(HookType).map((hookType) => (
              <label
                key={hookType}
                className="flex items-center gap-2 px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded cursor-pointer hover:border-slate-300 dark:hover:border-slate-600 transition-colors"
              >
                <input
                  type="checkbox"
                  checked={hookTypes.includes(hookType)}
                  onChange={() => handleHookTypeToggle(hookType)}
                  disabled={isSubmitting}
                  className="w-4 h-4 text-emerald-600 bg-slate-100 dark:bg-slate-700 border-slate-300 dark:border-slate-600 rounded focus:ring-2 focus:ring-emerald-500"
                />
                <span className="text-sm text-slate-700 dark:text-slate-300 capitalize">{hookType}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Favorite Toggle */}
        <div>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={isFavorite}
              onChange={(e) => setIsFavorite(e.target.checked)}
              disabled={isSubmitting}
              className="w-4 h-4 text-emerald-600 bg-slate-700 border-slate-600 rounded focus:ring-2 focus:ring-emerald-500"
            />
            <span className="text-sm text-slate-700 dark:text-slate-300">Mark as favorite</span>
          </label>
        </div>

        {/* Extraction Configuration Section */}
        <div className="border-t border-slate-300 dark:border-slate-700 pt-4">
          <h3 className="text-sm font-medium text-slate-800 dark:text-slate-200 mb-4">Extraction Configuration</h3>

          {/* Evaluation Samples + Top-K Grid */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Evaluation Samples</label>
              <input
                type="number"
                value={maxSamples}
                onChange={(e) => setMaxSamples(Number(e.target.value))}
                min={100}
                max={1000000}
                step={100}
                className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500"
                disabled={isSubmitting}
              />
              <p className="text-xs text-slate-500 mt-1">Max: 1,000,000</p>
            </div>
            <div>
              <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Top-K Examples per Feature</label>
              <input
                type="number"
                value={topKExamples}
                onChange={(e) => setTopKExamples(Number(e.target.value))}
                min={10}
                max={1000}
                step={10}
                className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500"
                disabled={isSubmitting}
              />
            </div>
          </div>

          {/* Dead Neuron Filtering */}
          <div className="mt-4 p-4 bg-slate-100 dark:bg-slate-800/50 border border-slate-300 dark:border-slate-700 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-slate-700 dark:text-slate-300">Dead Neuron Filtering</label>
              <span className="text-xs text-emerald-500">
                {(minActivationFrequency * 100).toFixed(2)}% min frequency
              </span>
            </div>
            <p className="text-xs text-slate-600 dark:text-slate-400 mb-3">
              Neurons firing less than this threshold are considered &quot;dead&quot; and will be filtered out.
            </p>
            <div className="flex items-center gap-3">
              <input
                type="range"
                value={minActivationFrequency * 1000}
                onChange={(e) => setMinActivationFrequency(Number(e.target.value) / 1000)}
                min={0}
                max={10}
                step={0.1}
                className="flex-1 h-2 bg-slate-100 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                disabled={isSubmitting}
              />
              <input
                type="number"
                value={(minActivationFrequency * 100).toFixed(2)}
                onChange={(e) => setMinActivationFrequency(Number(e.target.value) / 100)}
                min={0}
                max={10}
                step={0.01}
                className="w-20 px-2 py-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white text-sm focus:outline-none focus:border-emerald-500"
                disabled={isSubmitting}
              />
              <span className="text-xs text-slate-600 dark:text-slate-400">%</span>
            </div>
          </div>

          {/* Context Window Configuration (collapsible) */}
          <div className="mt-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-300 dark:border-slate-700 p-4">
            <button
              type="button"
              onClick={() => setShowContextWindow(!showContextWindow)}
              className="flex items-center justify-between w-full text-left"
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Context Window</span>
                <span className="text-xs text-emerald-500">
                  ({contextPrefixTokens} prefix + prime + {contextSuffixTokens} suffix)
                </span>
              </div>
              <ChevronDown className={`w-4 h-4 text-slate-600 dark:text-slate-400 transition-transform ${showContextWindow ? 'rotate-180' : ''}`} />
            </button>

            {showContextWindow && (
              <div className="mt-4 space-y-3">
                <p className="text-xs text-slate-600 dark:text-slate-400">
                  Capture tokens before and after the prime token (max activation) to provide context for interpretation.
                </p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Prefix Tokens</label>
                    <input
                      type="number"
                      value={contextPrefixTokens}
                      onChange={(e) => setContextPrefixTokens(Number(e.target.value))}
                      min={0}
                      max={50}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500"
                      disabled={isSubmitting}
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Suffix Tokens</label>
                    <input
                      type="number"
                      value={contextSuffixTokens}
                      onChange={(e) => setContextSuffixTokens(Number(e.target.value))}
                      min={0}
                      max={50}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:outline-none focus:border-emerald-500"
                      disabled={isSubmitting}
                    />
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Token Filtering (collapsible, 6 individual checkboxes) */}
          <div className="mt-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-300 dark:border-slate-700 p-4">
            <button
              type="button"
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center justify-between w-full text-left"
            >
              <div className="flex items-center gap-2">
                <span className="text-sm font-medium text-slate-700 dark:text-slate-300">Token Filtering</span>
                <span className="text-xs text-slate-500">
                  ({[filterSpecial, filterSingleChar, filterPunctuation, filterNumbers, filterFragments, filterStopWords].filter(Boolean).length}/6 enabled)
                </span>
              </div>
              <ChevronDown className={`w-4 h-4 text-slate-600 dark:text-slate-400 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
            </button>

            {showFilters && (
              <div className="mt-4 grid grid-cols-2 gap-3">
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterSpecial}
                    onChange={(e) => setFilterSpecial(e.target.checked)}
                    disabled={isSubmitting}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>Special tokens</span>
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterSingleChar}
                    onChange={(e) => setFilterSingleChar(e.target.checked)}
                    disabled={isSubmitting}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>Single characters</span>
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterPunctuation}
                    onChange={(e) => setFilterPunctuation(e.target.checked)}
                    disabled={isSubmitting}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>Punctuation</span>
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterNumbers}
                    onChange={(e) => setFilterNumbers(e.target.checked)}
                    disabled={isSubmitting}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>Numbers</span>
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterFragments}
                    onChange={(e) => setFilterFragments(e.target.checked)}
                    disabled={isSubmitting}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>Word fragments</span>
                </label>
                <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={filterStopWords}
                    onChange={(e) => setFilterStopWords(e.target.checked)}
                    disabled={isSubmitting}
                    className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                  />
                  <span>Stop words</span>
                </label>
              </div>
            )}
          </div>

          {/* NLP Processing Configuration */}
          <div className="mt-4 bg-white dark:bg-slate-900 rounded-lg border border-slate-300 dark:border-slate-700 p-4">
            <label className="flex items-center gap-3 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
              <input
                type="checkbox"
                checked={autoNlp}
                onChange={(e) => setAutoNlp(e.target.checked)}
                disabled={isSubmitting}
                className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500"
              />
              <div>
                <span className="font-medium">Auto-run NLP Analysis</span>
                <p className="text-xs text-slate-500 mt-0.5">
                  Automatically compute POS tags, NER, patterns, and clusters for feature labels
                </p>
              </div>
            </label>
          </div>
        </div>

        {/* Error Message */}
        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded text-sm text-red-400">
            {error}
          </div>
        )}

        {/* Submit Button */}
        <div className="flex gap-3">
          <button
            type="submit"
            disabled={isSubmitting || !name.trim() || !layerIndicesInput.trim()}
            className="flex-1 flex items-center justify-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed text-white font-medium rounded transition-colors"
          >
            <Save className="w-4 h-4" />
            {isSubmitting ? 'Saving...' : isEditMode ? 'Update Template' : 'Create Template'}
          </button>
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              disabled={isSubmitting}
              className="px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-slate-700 dark:text-slate-300 font-medium rounded transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </form>
  );
}

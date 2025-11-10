/**
 * ExtractionTemplateForm component for creating and editing extraction templates.
 *
 * This component renders a comprehensive form with all template fields.
 */

import React, { useState, useEffect } from 'react';
import { Save, X } from 'lucide-react';
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

  // Form state
  const [name, setName] = useState(template?.name || '');
  const [description, setDescription] = useState(template?.description || '');
  const [layerIndicesInput, setLayerIndicesInput] = useState(
    template?.layer_indices.join(', ') || ''
  );
  const [hookTypes, setHookTypes] = useState<string[]>(template?.hook_types || ['residual']);
  const [maxSamples, setMaxSamples] = useState(template?.max_samples || 1000);
  const [batchSize, setBatchSize] = useState(template?.batch_size || 32);
  const [topKExamples, setTopKExamples] = useState(template?.top_k_examples || 10);
  const [isFavorite, setIsFavorite] = useState(template?.is_favorite || false);
  const [metadataJson, setMetadataJson] = useState(
    template?.extra_metadata ? JSON.stringify(template.extra_metadata, null, 2) : '{}'
  );

  // Filtering configuration state
  const [filterEnabled, setFilterEnabled] = useState(template?.extraction_filter_enabled || false);
  const [filterMode, setFilterMode] = useState<'minimal' | 'conservative' | 'standard' | 'aggressive'>(
    template?.extraction_filter_mode || 'standard'
  );

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset form when template changes
  useEffect(() => {
    if (template) {
      setName(template.name);
      setDescription(template.description || '');
      setLayerIndicesInput(template.layer_indices.join(', '));
      setHookTypes(template.hook_types);
      setMaxSamples(template.max_samples);
      setBatchSize(template.batch_size);
      setTopKExamples(template.top_k_examples);
      setIsFavorite(template.is_favorite);
      setMetadataJson(template.extra_metadata ? JSON.stringify(template.extra_metadata, null, 2) : '{}');
      setFilterEnabled(template.extraction_filter_enabled || false);
      setFilterMode(template.extraction_filter_mode || 'standard');
    }
  }, [template]);

  const parseLayerIndices = (input: string): number[] | null => {
    try {
      const trimmed = input.trim();
      if (!trimmed) return null;

      // Parse comma-separated numbers
      const indices = trimmed
        .split(',')
        .map((s) => s.trim())
        .filter((s) => s.length > 0)
        .map((s) => parseInt(s, 10));

      // Validate all are numbers
      if (indices.some(isNaN)) {
        return null;
      }

      // Remove duplicates and sort
      return Array.from(new Set(indices)).sort((a, b) => a - b);
    } catch {
      return null;
    }
  };

  const validateMetadata = (json: string): Record<string, any> | null => {
    try {
      if (!json.trim()) return {};
      return JSON.parse(json);
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

    // Validation
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

    if (maxSamples < 1 || maxSamples > 1000000) {
      setError('Max samples must be between 1 and 1,000,000');
      return;
    }

    if (batchSize < 1 || batchSize > 1024) {
      setError('Batch size must be between 1 and 1024');
      return;
    }

    if (topKExamples < 1 || topKExamples > 100) {
      setError('Top-K examples must be between 1 and 100');
      return;
    }

    const metadata = validateMetadata(metadataJson);
    if (metadata === null) {
      setError('Invalid JSON in metadata field');
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
        extraction_filter_enabled: filterEnabled,
        extraction_filter_mode: filterMode,
        extra_metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
      };

      await onSubmit(data);

      // Reset form on success (only in create mode)
      if (!isEditMode) {
        setName('');
        setDescription('');
        setLayerIndicesInput('');
        setHookTypes(['residual']);
        setMaxSamples(1000);
        setBatchSize(32);
        setTopKExamples(10);
        setIsFavorite(false);
        setFilterEnabled(false);
        setFilterMode('standard');
        setMetadataJson('{}');
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
      className={`bg-slate-900/50 border border-slate-800 rounded-lg p-6 ${className}`}
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-slate-100">
          {isEditMode ? 'Edit Template' : 'Create Extraction Template'}
        </h2>
        {onCancel && (
          <button
            type="button"
            onClick={onCancel}
            className="p-1 hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-5 h-5 text-slate-400" />
          </button>
        )}
      </div>

      <div className="space-y-4">
        {/* Name */}
        <div>
          <label htmlFor="name" className="block text-sm font-medium text-slate-300 mb-2">
            Template Name <span className="text-red-400">*</span>
          </label>
          <input
            id="name"
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="My Extraction Template"
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            disabled={isSubmitting}
            required
          />
        </div>

        {/* Description */}
        <div>
          <label htmlFor="description" className="block text-sm font-medium text-slate-300 mb-2">
            Description
          </label>
          <textarea
            id="description"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            placeholder="Brief description of this template's purpose..."
            rows={2}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
            disabled={isSubmitting}
          />
        </div>

        {/* Layer Indices */}
        <div>
          <label
            htmlFor="layer-indices"
            className="block text-sm font-medium text-slate-300 mb-2"
          >
            Layer Indices <span className="text-red-400">*</span>
          </label>
          <input
            id="layer-indices"
            type="text"
            value={layerIndicesInput}
            onChange={(e) => setLayerIndicesInput(e.target.value)}
            placeholder="0, 5, 11, 23"
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            disabled={isSubmitting}
            required
          />
          <p className="text-xs text-slate-500 mt-1">
            Comma-separated layer indices (e.g., 0, 5, 11, 23)
          </p>
        </div>

        {/* Hook Types */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            Hook Types <span className="text-red-400">*</span>
          </label>
          <div className="flex flex-wrap gap-3">
            {Object.values(HookType).map((hookType) => (
              <label
                key={hookType}
                className="flex items-center gap-2 px-3 py-2 bg-slate-800 border border-slate-700 rounded cursor-pointer hover:border-slate-600 transition-colors"
              >
                <input
                  type="checkbox"
                  checked={hookTypes.includes(hookType)}
                  onChange={() => handleHookTypeToggle(hookType)}
                  disabled={isSubmitting}
                  className="w-4 h-4 text-emerald-600 bg-slate-700 border-slate-600 rounded focus:ring-2 focus:ring-emerald-500"
                />
                <span className="text-sm text-slate-300 capitalize">{hookType}</span>
              </label>
            ))}
          </div>
        </div>

        {/* Numeric Fields Grid */}
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label htmlFor="max-samples" className="block text-sm font-medium text-slate-300 mb-2">
              Max Samples
            </label>
            <input
              id="max-samples"
              type="number"
              value={maxSamples}
              onChange={(e) => setMaxSamples(parseInt(e.target.value, 10))}
              min="1"
              max="1000000"
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={isSubmitting}
              required
            />
          </div>

          <div>
            <label htmlFor="batch-size" className="block text-sm font-medium text-slate-300 mb-2">
              Batch Size
            </label>
            <input
              id="batch-size"
              type="number"
              value={batchSize}
              onChange={(e) => setBatchSize(parseInt(e.target.value, 10))}
              min="1"
              max="1024"
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={isSubmitting}
              required
            />
          </div>

          <div>
            <label htmlFor="top-k" className="block text-sm font-medium text-slate-300 mb-2">
              Top-K Examples
            </label>
            <input
              id="top-k"
              type="number"
              value={topKExamples}
              onChange={(e) => setTopKExamples(parseInt(e.target.value, 10))}
              min="1"
              max="100"
              className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={isSubmitting}
              required
            />
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
            <span className="text-sm text-slate-300">Mark as favorite</span>
          </label>
        </div>

        {/* Token Filtering Settings */}
        <div className="border-t border-slate-700 pt-4 space-y-3">
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="filter-enabled-template"
              checked={filterEnabled}
              onChange={(e) => setFilterEnabled(e.target.checked)}
              disabled={isSubmitting}
              className="w-4 h-4 text-emerald-600 bg-slate-700 border-slate-600 rounded focus:ring-2 focus:ring-emerald-500"
            />
            <label htmlFor="filter-enabled-template" className="text-sm font-medium text-slate-100">
              Enable Token Filtering
            </label>
          </div>

          {filterEnabled && (
            <div className="ml-6 space-y-3 bg-slate-800/50 p-3 rounded border border-slate-700/50">
              <div className="space-y-2">
                <label className="text-xs font-medium text-slate-300">Filter Mode</label>
                <div className="grid grid-cols-2 gap-2">
                  {(['minimal', 'conservative', 'standard', 'aggressive'] as const).map((mode) => (
                    <label key={mode} className="flex items-center gap-2 cursor-pointer">
                      <input
                        type="radio"
                        value={mode}
                        checked={filterMode === mode}
                        onChange={(e) => setFilterMode(e.target.value as typeof filterMode)}
                        disabled={isSubmitting}
                        className="w-3.5 h-3.5 text-emerald-600 bg-slate-700 border-slate-600 focus:ring-emerald-500"
                      />
                      <span className="text-sm text-slate-200 capitalize">{mode}</span>
                    </label>
                  ))}
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  {filterMode === 'minimal' && 'Only filter control characters'}
                  {filterMode === 'conservative' && 'Filter control characters + whitespace'}
                  {filterMode === 'standard' && 'Balanced filtering (recommended)'}
                  {filterMode === 'aggressive' && 'Maximum filtering (punctuation, numbers, etc.)'}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* Extra Metadata */}
        <div>
          <label htmlFor="metadata" className="block text-sm font-medium text-slate-300 mb-2">
            Extra Metadata (JSON)
          </label>
          <textarea
            id="metadata"
            value={metadataJson}
            onChange={(e) => setMetadataJson(e.target.value)}
            placeholder='{"author": "user", "version": "1.0"}'
            rows={4}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 font-mono text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent resize-none"
            disabled={isSubmitting}
          />
          <p className="text-xs text-slate-500 mt-1">Optional JSON metadata</p>
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
              className="px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-slate-300 font-medium rounded transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    </form>
  );
}

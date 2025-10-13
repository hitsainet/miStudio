/**
 * ActivationExtractionConfig - Modal for configuring activation extraction.
 *
 * Features:
 * - Dataset selection
 * - Layer selection with visual grid
 * - Activation type selection (residual, MLP, attention)
 * - Batch size and max samples configuration
 * - Real-time extraction progress tracking
 * - Support for extraction templates (future enhancement)
 */

import { useState, useEffect } from 'react';
import { X, Play, Star } from 'lucide-react';
import { Model, ActivationExtractionConfig as ExtractionConfig } from '../../types/model';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useExtractionTemplatesStore } from '../../stores/extractionTemplatesStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useModelExtractionProgress } from '../../hooks/useModelProgress';

interface ActivationExtractionConfigProps {
  model: Model;
  onClose: () => void;
  onExtract: (modelId: string, config: ExtractionConfig) => Promise<void>;
}

export function ActivationExtractionConfig({
  model,
  onClose,
  onExtract
}: ActivationExtractionConfigProps) {
  const { datasets, fetchDatasets } = useDatasetsStore();
  const { templates, favorites, fetchTemplates, fetchFavorites } = useExtractionTemplatesStore();
  const { models } = useModelsStore();

  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedLayers, setSelectedLayers] = useState<number[]>([0, 5, 11]);
  const [hookTypes, setHookTypes] = useState<('residual' | 'mlp' | 'attention')[]>(['residual']);
  const [batchSize, setBatchSize] = useState(32);
  const [maxSamples, setMaxSamples] = useState(1000);
  const [topKExamples, setTopKExamples] = useState(10);
  const [extracting, setExtracting] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [extractionStarted, setExtractionStarted] = useState(false);

  // Get the latest model data from store
  const latestModel = models.find(m => m.id === model.id) || model;

  // Subscribe to extraction progress updates for this model
  useModelExtractionProgress(extractionStarted ? model.id : undefined);

  // Determine number of layers from model architecture
  const numLayers = model.architecture_config?.num_layers ||
                   model.architecture_config?.num_hidden_layers ||
                   12;
  const layers = Array.from({ length: numLayers }, (_, i) => i);

  // Fetch datasets and templates on mount
  useEffect(() => {
    fetchDatasets();
    fetchTemplates();
    fetchFavorites();
  }, [fetchDatasets, fetchTemplates, fetchFavorites]);

  // Set first dataset as default
  useEffect(() => {
    if (datasets.length > 0 && !selectedDataset) {
      setSelectedDataset(datasets[0].id);
    }
  }, [datasets, selectedDataset]);

  // Monitor extraction progress from store and display errors
  useEffect(() => {
    if (extractionStarted && latestModel.extraction_status === 'error') {
      setValidationError(latestModel.extraction_message || 'Extraction failed');
      setExtracting(false);
      setExtractionStarted(false);
    }
  }, [extractionStarted, latestModel.extraction_status, latestModel.extraction_message]);

  // Load template configuration when selected
  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId);

    if (!templateId) {
      // Reset to defaults if "None" is selected
      setSelectedLayers([0, 5, 11]);
      setHookTypes(['residual']);
      setBatchSize(32);
      setMaxSamples(1000);
      setTopKExamples(10);
      return;
    }

    const template = templates.find(t => t.id === templateId);
    if (template) {
      setSelectedLayers([...template.layer_indices]);
      setHookTypes([...template.hook_types] as any);
      setBatchSize(template.batch_size);
      setMaxSamples(template.max_samples);
      setTopKExamples(template.top_k_examples);
    }
  };

  const toggleLayer = (layer: number) => {
    if (selectedLayers.includes(layer)) {
      setSelectedLayers(selectedLayers.filter((l) => l !== layer));
    } else {
      setSelectedLayers([...selectedLayers, layer].sort((a, b) => a - b));
    }
  };

  const selectAllLayers = () => {
    setSelectedLayers(layers);
  };

  const deselectAllLayers = () => {
    setSelectedLayers([]);
  };

  const toggleHookType = (type: 'residual' | 'mlp' | 'attention') => {
    if (hookTypes.includes(type)) {
      setHookTypes(hookTypes.filter((t) => t !== type));
    } else {
      setHookTypes([...hookTypes, type]);
    }
  };

  const validate = (): boolean => {
    if (!selectedDataset) {
      setValidationError('Please select a dataset');
      return false;
    }

    if (selectedLayers.length === 0) {
      setValidationError('Please select at least one layer');
      return false;
    }

    if (hookTypes.length === 0) {
      setValidationError('Please select at least one hook type');
      return false;
    }

    if (batchSize < 1 || batchSize > 256) {
      setValidationError('Batch size must be between 1 and 256');
      return false;
    }

    if (maxSamples < 1 || maxSamples > 100000) {
      setValidationError('Max samples must be between 1 and 100,000');
      return false;
    }

    setValidationError(null);
    return true;
  };

  const handleExtract = async () => {
    if (!validate()) {
      return;
    }

    setExtracting(true);
    setExtractionStarted(true);
    try {
      const config: ExtractionConfig = {
        dataset_id: selectedDataset,
        layer_indices: selectedLayers,
        hook_types: hookTypes,
        max_samples: maxSamples,
        batch_size: batchSize,
        top_k_examples: topKExamples,
      };

      await onExtract(model.id, config);

      // Keep modal open so user can see progress or errors
      // WebSocket will update the model state, which will trigger error display if needed
      // User can manually close when ready
      setExtracting(false);
    } catch (error) {
      console.error('[ActivationExtractionConfig] Extraction failed:', error);
      setValidationError(error instanceof Error ? error.message : 'Extraction failed');
      setExtracting(false);
      setExtractionStarted(false);
    }
  };

  const readyDatasets = datasets.filter((ds) => ds.status === 'ready');

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-3xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-2xl font-semibold text-emerald-400">Extract Activations</h2>
            <p className="text-sm text-slate-400 mt-1">
              Extract activations from {model.name}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            disabled={extracting}
            className="text-slate-400 hover:text-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Close"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Template Selection */}
          {templates.length > 0 && (
            <div>
              <label htmlFor="extraction-template" className="block text-sm font-medium text-slate-300 mb-2">
                Load Template (Optional)
              </label>
              <select
                id="extraction-template"
                value={selectedTemplate}
                onChange={(e) => handleTemplateSelect(e.target.value)}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                <option value="">None (Manual Configuration)</option>
                {favorites.length > 0 && (
                  <optgroup label="⭐ Favorites">
                    {favorites.map((template) => (
                      <option key={template.id} value={template.id}>
                        {template.name}
                      </option>
                    ))}
                  </optgroup>
                )}
                {templates.length > 0 && (
                  <optgroup label="All Templates">
                    {templates.map((template) => (
                      <option key={template.id} value={template.id}>
                        {template.name}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
              {selectedTemplate && (
                <p className="text-xs text-slate-500 mt-2">
                  Template loaded. You can still modify settings below.
                </p>
              )}
            </div>
          )}

          {/* Dataset Selection */}
          <div>
            <label htmlFor="extraction-dataset" className="block text-sm font-medium text-slate-300 mb-2">
              Select Dataset
            </label>
            {readyDatasets.length === 0 ? (
              <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg text-yellow-400 text-sm">
                No ready datasets available. Please download and prepare a dataset first.
              </div>
            ) : (
              <select
                id="extraction-dataset"
                value={selectedDataset}
                onChange={(e) => setSelectedDataset(e.target.value)}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {readyDatasets.map((ds) => (
                  <option key={ds.id} value={ds.id}>
                    {ds.name} ({ds.num_samples?.toLocaleString() || 0} samples)
                  </option>
                ))}
              </select>
            )}
          </div>

          {/* Layer Selection */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-slate-300">
                Select Layers ({selectedLayers.length} selected)
              </label>
              <div className="flex gap-2">
                <button
                  type="button"
                  onClick={selectAllLayers}
                  disabled={extracting}
                  className="text-xs px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Select All
                </button>
                <button
                  type="button"
                  onClick={deselectAllLayers}
                  disabled={extracting}
                  className="text-xs px-2 py-1 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Deselect All
                </button>
              </div>
            </div>
            <div className="grid grid-cols-6 gap-2">
              {layers.map((layer) => (
                <button
                  type="button"
                  key={layer}
                  onClick={() => toggleLayer(layer)}
                  disabled={extracting}
                  className={`px-3 py-2 rounded font-mono text-sm transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
                    selectedLayers.includes(layer)
                      ? 'bg-emerald-600 hover:bg-emerald-700 text-white'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                  }`}
                >
                  L{layer}
                </button>
              ))}
            </div>
          </div>

          {/* Hook Type Selection */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Hook Types
            </label>
            <div className="flex gap-3">
              {(['residual', 'mlp', 'attention'] as const).map((type) => (
                <button
                  key={type}
                  type="button"
                  onClick={() => toggleHookType(type)}
                  disabled={extracting}
                  className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed capitalize ${
                    hookTypes.includes(type)
                      ? 'bg-purple-600 hover:bg-purple-700 text-white'
                      : 'bg-slate-800 hover:bg-slate-700 text-slate-300'
                  }`}
                >
                  {type}
                </button>
              ))}
            </div>
          </div>

          {/* Extraction Settings */}
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label htmlFor="extraction-batch-size" className="block text-sm font-medium text-slate-300 mb-2">
                Batch Size
              </label>
              <input
                id="extraction-batch-size"
                type="number"
                min="1"
                max="256"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value) || 1)}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
            </div>
            <div>
              <label htmlFor="extraction-max-samples" className="block text-sm font-medium text-slate-300 mb-2">
                Max Samples
              </label>
              <input
                id="extraction-max-samples"
                type="number"
                min="1"
                max="100000"
                value={maxSamples}
                onChange={(e) => setMaxSamples(parseInt(e.target.value) || 1)}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
            </div>
            <div>
              <label htmlFor="extraction-top-k" className="block text-sm font-medium text-slate-300 mb-2">
                Top K Examples
              </label>
              <input
                id="extraction-top-k"
                type="number"
                min="1"
                max="100"
                value={topKExamples}
                onChange={(e) => setTopKExamples(parseInt(e.target.value) || 1)}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
            </div>
          </div>

          {/* Validation Error */}
          {validationError && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
              {validationError}
            </div>
          )}

          {/* Extraction Info */}
          {!extracting && selectedLayers.length > 0 && hookTypes.length > 0 && (
            <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-slate-300">
              <div className="font-medium mb-2">Extraction Summary:</div>
              <ul className="space-y-1 text-slate-400">
                <li>• Will extract from {selectedLayers.length} layer(s)</li>
                <li>• Using {hookTypes.length} hook type(s): {hookTypes.join(', ')}</li>
                <li>• Processing up to {maxSamples.toLocaleString()} samples</li>
                <li>• Batch size: {batchSize}</li>
              </ul>
            </div>
          )}

          {/* Extract Button */}
          <button
            type="button"
            onClick={handleExtract}
            disabled={extracting || readyDatasets.length === 0 || selectedLayers.length === 0 || hookTypes.length === 0}
            className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium text-white"
          >
            {extracting ? (
              <>
                <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
                Starting Extraction...
              </>
            ) : (
              <>
                <Play className="w-5 h-5" />
                Start Extraction
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

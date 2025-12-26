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
import { X, Play, Save, AlertCircle, Info, Cpu } from 'lucide-react';
import { Model, ActivationExtractionConfig as ExtractionConfig } from '../../types/model';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useExtractionTemplatesStore } from '../../stores/extractionTemplatesStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useSystemMonitorStore } from '../../stores/systemMonitorStore';
import { useModelExtractionProgress } from '../../hooks/useModelProgress';
import { estimateExtractionResources } from '../../api/models';

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
  const { datasets, fetchDatasets, tokenizations, fetchTokenizations } = useDatasetsStore();
  const { templates, favorites, fetchTemplates, fetchFavorites, createTemplate } = useExtractionTemplatesStore();
  const { models } = useModelsStore();
  const { gpuList, fetchGPUList } = useSystemMonitorStore();

  const [selectedTemplate, setSelectedTemplate] = useState('');
  const [selectedDataset, setSelectedDataset] = useState('');
  const [selectedLayers, setSelectedLayers] = useState<number[]>([0, 5, 11]);
  const [hookTypes, setHookTypes] = useState<('residual' | 'mlp' | 'attention')[]>(['residual']);
  const [batchSize, setBatchSize] = useState(32);
  const [microBatchSize, setMicroBatchSize] = useState<number | ''>(8); // GPU micro-batch size for memory efficiency
  const [maxSamples, setMaxSamples] = useState(100000);
  const [topKExamples, setTopKExamples] = useState(10);
  const [gpuId, setGpuId] = useState(0);
  const [extracting, setExtracting] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [extractionStarted, setExtractionStarted] = useState(false);
  const [showSaveTemplate, setShowSaveTemplate] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [templateDescription, setTemplateDescription] = useState('');
  const [savingTemplate, setSavingTemplate] = useState(false);
  const [resourceEstimates, setResourceEstimates] = useState<any>(null);
  const [loadingEstimates, setLoadingEstimates] = useState(false);

  // Get the latest model data from store
  const latestModel = models.find(m => m.id === model.id) || model;

  // Subscribe to extraction progress updates for this model
  useModelExtractionProgress(extractionStarted ? model.id : undefined);

  // Determine number of layers from model architecture
  const numLayers = model.architecture_config?.num_layers ||
                   model.architecture_config?.num_hidden_layers ||
                   12;
  const layers = Array.from({ length: numLayers }, (_, i) => i);

  // Fetch datasets, templates, and GPU list on mount
  useEffect(() => {
    fetchDatasets();
    fetchTemplates();
    fetchFavorites();
    fetchGPUList();
  }, [fetchDatasets, fetchTemplates, fetchFavorites, fetchGPUList]);

  // Set first dataset as default
  useEffect(() => {
    if (datasets.length > 0 && !selectedDataset) {
      setSelectedDataset(datasets[0].id);
    }
  }, [datasets, selectedDataset]);

  // Fetch tokenizations when dataset changes
  useEffect(() => {
    if (selectedDataset) {
      fetchTokenizations(selectedDataset);
    }
  }, [selectedDataset, fetchTokenizations]);

  // Check if selected dataset has tokenizations for this model
  const datasetTokenizations = selectedDataset ? tokenizations[selectedDataset] || [] : [];
  const modelTokenization = datasetTokenizations.find(t => t.model_id === model.id && t.status === 'ready');
  const hasTokenization = !!modelTokenization;

  // Monitor extraction progress from store and display errors
  useEffect(() => {
    if (extractionStarted && latestModel.extraction_status === 'error') {
      setValidationError(latestModel.extraction_message || 'Extraction failed');
      setExtracting(false);
      setExtractionStarted(false);
    }
  }, [extractionStarted, latestModel.extraction_status, latestModel.extraction_message]);

  // Fetch resource estimates whenever configuration changes
  useEffect(() => {
    // Only fetch if we have valid configuration
    if (!selectedDataset || selectedLayers.length === 0 || hookTypes.length === 0) {
      setResourceEstimates(null);
      setValidationError(null);
      return;
    }

    // Debounce the estimate fetch
    const timeoutId = setTimeout(async () => {
      setLoadingEstimates(true);
      setValidationError(null); // Clear any previous errors
      try {
        const config: ExtractionConfig = {
          dataset_id: selectedDataset,
          layer_indices: selectedLayers,
          hook_types: hookTypes,
          max_samples: maxSamples,
          batch_size: batchSize,
          micro_batch_size: microBatchSize === '' ? undefined : microBatchSize,
          top_k_examples: topKExamples,
        };

        const result = await estimateExtractionResources(model.id, config);
        console.log('[ActivationExtractionConfig] Resource estimates received:', result);
        setResourceEstimates(result.estimates);
      } catch (error) {
        console.error('[ActivationExtractionConfig] Failed to fetch resource estimates:', error);

        // Extract meaningful error message from API error
        if (error && typeof error === 'object') {
          let errorMessage = 'Failed to calculate resource estimates';

          // Check for Pydantic validation errors (status 422)
          if ('detail' in error && Array.isArray((error as any).detail) && (error as any).detail.length > 0) {
            const firstError = (error as any).detail[0];
            if (firstError.msg) {
              errorMessage = firstError.msg;
            }
          }
          // Check for standard error message
          else if ('message' in error && typeof (error as any).message === 'string') {
            errorMessage = (error as any).message;
          }

          setValidationError(errorMessage);
        }
        setResourceEstimates(null);
      } finally {
        setLoadingEstimates(false);
      }
    }, 500); // 500ms debounce

    return () => clearTimeout(timeoutId);
  }, [selectedDataset, selectedLayers, hookTypes, maxSamples, batchSize, topKExamples, model.id]);

  // Load template configuration when selected
  const handleTemplateSelect = (templateId: string) => {
    setSelectedTemplate(templateId);

    if (!templateId) {
      // Reset to defaults if "None" is selected
      setSelectedLayers([0, 5, 11]);
      setHookTypes(['residual']);
      setBatchSize(32);
      setMicroBatchSize(8);
      setMaxSamples(100000);
      setTopKExamples(10);
      return;
    }

    const template = templates.find(t => t.id === templateId);
    if (template) {
      setSelectedLayers([...template.layer_indices]);
      setHookTypes([...template.hook_types] as any);
      setBatchSize(template.batch_size);
      setMicroBatchSize(template.micro_batch_size || 8);
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

    if (!hasTokenization) {
      const selectedDs = datasets.find(d => d.id === selectedDataset);
      setValidationError(`Dataset "${selectedDs?.name || selectedDataset}" has not been tokenized for model "${model.name}". Please tokenize the dataset first.`);
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

    if (maxSamples < 1 || maxSamples > 1000000) {
      setValidationError('Max samples must be between 1 and 1,000,000');
      return false;
    }

    setValidationError(null);
    return true;
  };

  // Generate a smart default template name based on configuration
  const generateTemplateName = (): string => {
    // Format layers as ranges or individual values
    const layerStr = selectedLayers.length <= 3
      ? selectedLayers.join(',')
      : `${selectedLayers[0]}-${selectedLayers[selectedLayers.length - 1]}`;

    // Format hook types
    const hookStr = hookTypes.join(',');

    // Get model name (truncate if too long)
    const modelName = model.name.length > 30 ? model.name.substring(0, 30) + '...' : model.name;

    return `${modelName} - L${layerStr} - ${hookStr}`;
  };

  // Open save template dialog with generated name
  const handleOpenSaveTemplate = () => {
    setTemplateName(generateTemplateName());
    setTemplateDescription(`Extraction config for ${model.name} with ${selectedLayers.length} layers and ${hookTypes.length} hook types`);
    setShowSaveTemplate(true);
  };

  // Save the current configuration as a template
  const handleSaveTemplate = async () => {
    if (!templateName.trim()) {
      setValidationError('Template name is required');
      return;
    }

    setSavingTemplate(true);
    try {
      await createTemplate({
        name: templateName,
        description: templateDescription || undefined,
        layer_indices: selectedLayers,
        hook_types: hookTypes,
        max_samples: maxSamples,
        batch_size: batchSize,
        micro_batch_size: microBatchSize === '' ? undefined : microBatchSize,
        top_k_examples: topKExamples,
        is_favorite: false,
      });

      // Refresh templates list
      await fetchTemplates();
      await fetchFavorites();

      // Close dialog
      setShowSaveTemplate(false);
      setTemplateName('');
      setTemplateDescription('');
      setSavingTemplate(false);

      // Show success (could add a toast notification here)
      console.log('[ActivationExtractionConfig] Template saved successfully');
    } catch (error) {
      console.error('[ActivationExtractionConfig] Failed to save template:', error);
      setValidationError(error instanceof Error ? error.message : 'Failed to save template');
      setSavingTemplate(false);
    }
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
        micro_batch_size: microBatchSize === '' ? undefined : microBatchSize,
        top_k_examples: topKExamples,
        gpu_id: gpuId,
      };

      await onExtract(model.id, config);

      // Close modal immediately after extraction starts successfully
      // User can see progress on the model card
      onClose();
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
              <>
                <select
                  id="extraction-dataset"
                  value={selectedDataset}
                  onChange={(e) => setSelectedDataset(e.target.value)}
                  disabled={extracting}
                  className={`w-full px-4 py-2 bg-slate-900 border rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${
                    selectedDataset && !hasTokenization ? 'border-yellow-500/50' : 'border-slate-700'
                  }`}
                >
                  {readyDatasets.map((ds) => (
                    <option key={ds.id} value={ds.id}>
                      {ds.name} ({ds.num_samples?.toLocaleString() || 0} samples)
                    </option>
                  ))}
                </select>
                {/* Tokenization Warning */}
                {selectedDataset && !hasTokenization && (
                  <div className="mt-2 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg flex items-start gap-2">
                    <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                    <div className="text-sm text-yellow-400">
                      <span className="font-medium">Tokenization required:</span> This dataset has not been tokenized for {model.name}.
                      Go to the Datasets panel to tokenize it first.
                    </div>
                  </div>
                )}
                {/* Show tokenization info when available */}
                {selectedDataset && hasTokenization && modelTokenization && (
                  <p className="text-xs text-emerald-400/70 mt-1">
                    ✓ Tokenized: {modelTokenization.num_tokens?.toLocaleString() || '?'} tokens
                  </p>
                )}
              </>
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
              <label htmlFor="extraction-micro-batch-size" className="block text-sm font-medium text-slate-300 mb-2">
                Micro-Batch Size
                <span className="text-xs text-slate-500 ml-2">(GPU memory)</span>
              </label>
              <input
                id="extraction-micro-batch-size"
                type="number"
                min="1"
                max="256"
                value={microBatchSize}
                onChange={(e) => setMicroBatchSize(parseInt(e.target.value) || 1)}
                disabled={extracting}
                placeholder="Auto (same as batch)"
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
                max="1000000"
                value={maxSamples}
                onChange={(e) => setMaxSamples(parseInt(e.target.value) || 1)}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
              <p className="text-xs text-slate-500 mt-1">Max: 1,000,000</p>
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
            <div>
              <label htmlFor="extraction-gpu" className="block text-sm font-medium text-slate-300 mb-2">
                <Cpu className="w-4 h-4 inline mr-1" />
                GPU Device
              </label>
              <select
                id="extraction-gpu"
                value={gpuId}
                onChange={(e) => setGpuId(parseInt(e.target.value))}
                disabled={extracting}
                className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {gpuList?.gpus && gpuList.gpus.length > 0 ? (
                  gpuList.gpus.map((gpu) => (
                    <option key={gpu.gpu_id} value={gpu.gpu_id}>
                      GPU {gpu.gpu_id}: {gpu.name}
                    </option>
                  ))
                ) : (
                  <option value={0}>GPU 0 (Default)</option>
                )}
              </select>
              {gpuList?.gpus && gpuList.gpus.length > 1 && (
                <p className="text-xs text-slate-500 mt-1">Select GPU with most free VRAM</p>
              )}
            </div>
          </div>

          {/* Resource Estimates */}
          {!extracting && selectedLayers.length > 0 && hookTypes.length > 0 && selectedDataset && (
            <div className="space-y-4">
              {/* Extraction Summary */}
              <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-slate-300">
                <div className="font-medium mb-2">Extraction Summary:</div>
                <ul className="space-y-1 text-slate-400">
                  <li>• Will extract from {selectedLayers.length} layer(s)</li>
                  <li>• Using {hookTypes.length} hook type(s): {hookTypes.join(', ')}</li>
                  <li>• Processing up to {maxSamples.toLocaleString()} samples</li>
                  <li>• Batch size: {batchSize} (micro-batch: {microBatchSize || 'auto'})</li>
                </ul>
              </div>

              {/* Resource Requirements */}
              {loadingEstimates ? (
                <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                  <div className="flex items-center gap-2 text-slate-400">
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-slate-400 border-t-transparent"></div>
                    <span className="text-sm">Calculating resource requirements...</span>
                  </div>
                </div>
              ) : resourceEstimates ? (
                <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg text-sm">
                  <div className="flex items-center gap-2 font-medium mb-3 text-slate-300">
                    <Info className="w-4 h-4 text-blue-400" />
                    <span>Resource Requirements:</span>
                  </div>

                  <div className="space-y-3 text-slate-400">
                    {/* GPU Memory */}
                    <div className="flex justify-between items-center">
                      <span>GPU Memory:</span>
                      <span className={`font-medium ${
                        resourceEstimates.gpu_memory.warning === 'high' ? 'text-red-400' :
                        resourceEstimates.gpu_memory.warning === 'medium' ? 'text-yellow-400' :
                        'text-emerald-400'
                      }`}>
                        {resourceEstimates.gpu_memory.total_gb.toFixed(2)} GB
                      </span>
                    </div>

                    {/* Disk Space */}
                    <div className="flex justify-between items-center">
                      <span>Disk Space:</span>
                      <span className={`font-medium ${
                        resourceEstimates.disk_space.warning === 'high' ? 'text-red-400' :
                        resourceEstimates.disk_space.warning === 'medium' ? 'text-yellow-400' :
                        'text-emerald-400'
                      }`}>
                        {resourceEstimates.disk_space.total_gb.toFixed(2)} GB
                      </span>
                    </div>

                    {/* Processing Time */}
                    <div className="flex justify-between items-center">
                      <span>Estimated Time:</span>
                      <span className={`font-medium ${
                        resourceEstimates.processing_time.warning === 'long' ? 'text-yellow-400' :
                        'text-emerald-400'
                      }`}>
                        {resourceEstimates.processing_time.time_str}
                      </span>
                    </div>

                    {/* Warnings */}
                    {resourceEstimates.warnings && resourceEstimates.warnings.length > 0 && (
                      <div className="pt-2 mt-2 border-t border-slate-700">
                        <div className="flex items-start gap-2">
                          <AlertCircle className="w-4 h-4 text-yellow-400 mt-0.5 flex-shrink-0" />
                          <div className="space-y-1">
                            {resourceEstimates.warnings.map((warning: string, idx: number) => (
                              <div key={idx} className="text-yellow-400 text-xs">
                                {warning}
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ) : null}
            </div>
          )}

          {/* Validation Error */}
          {validationError && (
            <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
              {validationError}
            </div>
          )}

          {/* Action Buttons */}
          <div className="grid grid-cols-2 gap-3">
            {/* Save as Template Button */}
            <button
              type="button"
              onClick={handleOpenSaveTemplate}
              disabled={extracting || selectedLayers.length === 0 || hookTypes.length === 0}
              className="px-6 py-3 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium text-white"
            >
              <Save className="w-5 h-5" />
              Save as Template
            </button>

            {/* Extract Button */}
            <button
              type="button"
              onClick={handleExtract}
              disabled={extracting || readyDatasets.length === 0 || selectedLayers.length === 0 || hookTypes.length === 0 || !hasTokenization}
              className="px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium text-white"
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

      {/* Save Template Dialog */}
      {showSaveTemplate && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[60] flex items-center justify-center p-4">
          <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-md w-full">
            {/* Dialog Header */}
            <div className="flex items-center justify-between p-6 border-b border-slate-800">
              <div>
                <h3 className="text-xl font-semibold text-emerald-400">Save as Template</h3>
                <p className="text-sm text-slate-400 mt-1">Save current configuration for reuse</p>
              </div>
              <button
                type="button"
                onClick={() => setShowSaveTemplate(false)}
                disabled={savingTemplate}
                className="text-slate-400 hover:text-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                aria-label="Close"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Dialog Content */}
            <div className="p-6 space-y-4">
              {/* Template Name */}
              <div>
                <label htmlFor="template-name" className="block text-sm font-medium text-slate-300 mb-2">
                  Template Name *
                </label>
                <input
                  id="template-name"
                  type="text"
                  autoComplete="off"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  disabled={savingTemplate}
                  placeholder="e.g., GPT-2 Small - L0,5,11 - residual,mlp"
                  className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                />
              </div>

              {/* Template Description */}
              <div>
                <label htmlFor="template-description" className="block text-sm font-medium text-slate-300 mb-2">
                  Description (Optional)
                </label>
                <textarea
                  id="template-description"
                  value={templateDescription}
                  onChange={(e) => setTemplateDescription(e.target.value)}
                  disabled={savingTemplate}
                  rows={3}
                  placeholder="Describe what this template is for..."
                  className="w-full px-4 py-2 bg-slate-950 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors resize-none"
                />
              </div>

              {/* Configuration Summary */}
              <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg text-sm text-slate-300">
                <div className="font-medium mb-2">Template Configuration:</div>
                <ul className="space-y-1 text-slate-400">
                  <li>• Layer(s): {selectedLayers.join(', ')}</li>
                  <li>• Hook Types: {hookTypes.join(', ')}</li>
                  <li>• Batch Size: {batchSize} (micro-batch: {microBatchSize || 'auto'})</li>
                  <li>• Max Samples: {maxSamples.toLocaleString()}</li>
                  <li>• Top K Examples: {topKExamples}</li>
                </ul>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => setShowSaveTemplate(false)}
                  disabled={savingTemplate}
                  className="flex-1 px-4 py-2 bg-slate-700 hover:bg-slate-600 rounded-lg transition-colors text-slate-100 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleSaveTemplate}
                  disabled={savingTemplate || !templateName.trim()}
                  className="flex-1 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium flex items-center justify-center gap-2"
                >
                  {savingTemplate ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent"></div>
                      Saving...
                    </>
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save Template
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * StartExtractionModal Component
 *
 * Modal dialog for configuring and starting feature extraction from a training job.
 * Replaces the inline FeaturesPanel configuration form in TrainingCard.
 *
 * Workflow:
 * 1. User clicks "Start Extraction" button on completed training
 * 2. This modal opens with extraction configuration options
 * 3. User configures extraction parameters
 * 4. User clicks "Start Extraction" to begin
 * 5. Modal shows success message directing to Extractions tab
 * 6. User navigates to Extractions tab to view progress and results
 */

import React, { useState } from 'react';
import { Zap, X } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import type { Training } from '../../types/training';
import { ResourceConfigPanel } from './ResourceConfigPanel';

interface StartExtractionModalProps {
  training: Training;
  isOpen: boolean;
  onClose: () => void;
}

/**
 * StartExtractionModal Component.
 */
export const StartExtractionModal: React.FC<StartExtractionModalProps> = ({
  training,
  isOpen,
  onClose,
}) => {
  const {
    isLoadingExtraction,
    extractionError,
    startExtraction,
  } = useFeaturesStore();

  // Local state for extraction config
  const [evaluationSamples, setEvaluationSamples] = useState(10000);
  const [topKExamples, setTopKExamples] = useState(100);
  const [labelingMethod, setLabelingMethod] = useState<'pattern' | 'local' | 'openai'>('pattern');
  const [ollamaEndpointUrl, setOllamaEndpointUrl] = useState<string>('http://mistudio.mcslab.io/ollama/');
  const [localLabelingModel, setLocalLabelingModel] = useState<string>('gemma2:2b');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [openaiModel, setOpenaiModel] = useState<'gpt4-mini' | 'gpt4' | 'gpt35'>('gpt4-mini');
  const [vectorizationBatchSize, setVectorizationBatchSize] = useState<string>('auto');
  const [softTimeLimit, setSoftTimeLimit] = useState<number>(40); // hours
  const [hardTimeLimit, setHardTimeLimit] = useState<number>(48); // hours
  const [resourceConfig, setResourceConfig] = useState<{
    batch_size?: number;
    num_workers?: number;
    db_commit_batch?: number;
  }>({});
  const [availableOllamaModels, setAvailableOllamaModels] = useState<Array<{name: string; display_name: string}>>([]);
  const [isLoadingOllamaModels, setIsLoadingOllamaModels] = useState(false);
  const [showSuccessMessage, setShowSuccessMessage] = useState(false);

  // Token filtering configuration
  const [filterSpecial, setFilterSpecial] = useState(true);
  const [filterSingleChar, setFilterSingleChar] = useState(true);
  const [filterPunctuation, setFilterPunctuation] = useState(true);
  const [filterNumbers, setFilterNumbers] = useState(true);
  const [filterFragments, setFilterFragments] = useState(true);
  const [filterStopWords, setFilterStopWords] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  /**
   * Manually fetch Ollama models from the specified endpoint.
   */
  const handleFetchOllamaModels = async () => {
    setIsLoadingOllamaModels(true);
    try {
      const response = await fetch('/api/v1/labeling/models/available');
      if (response.ok) {
        const data = await response.json();
        setAvailableOllamaModels(data.models || []);
        // Set first model as default if available
        if (data.models && data.models.length > 0) {
          setLocalLabelingModel(data.models[0].name);
        }
      } else {
        console.error('Failed to fetch Ollama models');
        setAvailableOllamaModels([]);
      }
    } catch (error) {
      console.error('Error fetching Ollama models:', error);
      setAvailableOllamaModels([]);
    } finally {
      setIsLoadingOllamaModels(false);
    }
  };

  /**
   * Handle start extraction button click.
   */
  const handleStartExtraction = async () => {
    try {
      await startExtraction(training.id, {
        evaluation_samples: evaluationSamples,
        top_k_examples: topKExamples,
        vectorization_batch_size: vectorizationBatchSize === 'auto' ? 'auto' : parseInt(vectorizationBatchSize),
        soft_time_limit: softTimeLimit * 3600, // Convert hours to seconds
        time_limit: hardTimeLimit * 3600, // Convert hours to seconds
        // Token filtering configuration
        filter_special: filterSpecial,
        filter_single_char: filterSingleChar,
        filter_punctuation: filterPunctuation,
        filter_numbers: filterNumbers,
        filter_fragments: filterFragments,
        filter_stop_words: filterStopWords,
        ...resourceConfig,
      } as any);

      // Show success message
      setShowSuccessMessage(true);
    } catch (error) {
      console.error('Failed to start extraction:', error);
    }
  };

  /**
   * Handle modal close.
   */
  const handleClose = () => {
    setShowSuccessMessage(false);
    onClose();
  };

  // Don't render if not open
  if (!isOpen) {
    return null;
  }

  return (
    <>
      {/* Modal Overlay */}
      <div
        className="fixed inset-0 bg-black/70 z-40"
        onClick={handleClose}
      />

      {/* Modal Dialog */}
      <div className="fixed inset-0 flex items-center justify-center z-50 p-4">
        <div className="bg-slate-900 rounded-lg border border-slate-700 shadow-2xl w-full max-w-3xl max-h-[90vh] overflow-y-auto">
          {/* Modal Header */}
          <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white">Start Feature Extraction</h2>
              <p className="text-sm text-slate-400 mt-1">
                Configure and extract interpretable features from trained SAE
              </p>
            </div>
            <button
              onClick={handleClose}
              className="text-slate-400 hover:text-white transition-colors"
              aria-label="Close modal"
            >
              <X className="w-6 h-6" />
            </button>
          </div>

          {/* Modal Body */}
          <div className="p-6 space-y-4">
            {/* Success Message */}
            {showSuccessMessage && (
              <div className="p-4 bg-emerald-900/20 border border-emerald-700 rounded-lg">
                <p className="text-emerald-400 font-medium mb-2">
                  ✓ Extraction started successfully!
                </p>
                <p className="text-sm text-slate-300">
                  Navigate to the <span className="font-semibold text-emerald-400">Extractions</span> tab to view progress and browse discovered features.
                </p>
              </div>
            )}

            {/* Configuration Form (only show if not successful) */}
            {!showSuccessMessage && (
              <>
                {/* Basic Configuration */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Evaluation Samples</label>
                    <input
                      type="number"
                      value={evaluationSamples}
                      onChange={(e) => setEvaluationSamples(Number(e.target.value))}
                      min={1000}
                      max={100000}
                      step={1000}
                      className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                  <div>
                    <label className="block text-xs text-slate-400 mb-1">Top-K Examples per Feature</label>
                    <input
                      type="number"
                      value={topKExamples}
                      onChange={(e) => setTopKExamples(Number(e.target.value))}
                      min={10}
                      max={1000}
                      step={10}
                      className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                    />
                  </div>
                </div>

                {/* Token Filtering Configuration */}
                <div className="bg-slate-900 rounded-lg border border-slate-700 p-4">
                  <button
                    type="button"
                    onClick={() => setShowFilters(!showFilters)}
                    className="flex items-center justify-between w-full text-left"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-slate-300">Token Filtering</span>
                      <span className="text-xs text-slate-500">
                        ({[filterSpecial, filterSingleChar, filterPunctuation, filterNumbers, filterFragments, filterStopWords].filter(Boolean).length}/6 enabled)
                      </span>
                    </div>
                    <span className="text-slate-400">{showFilters ? '▼' : '▶'}</span>
                  </button>

                  {showFilters && (
                    <div className="mt-4 grid grid-cols-2 gap-3">
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterSpecial}
                          onChange={(e) => setFilterSpecial(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span>Special tokens</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterSingleChar}
                          onChange={(e) => setFilterSingleChar(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span>Single characters</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterPunctuation}
                          onChange={(e) => setFilterPunctuation(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span>Punctuation</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterNumbers}
                          onChange={(e) => setFilterNumbers(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span>Numbers</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterFragments}
                          onChange={(e) => setFilterFragments(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span>Word fragments</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterStopWords}
                          onChange={(e) => setFilterStopWords(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                        />
                        <span>Stop words</span>
                      </label>
                    </div>
                  )}
                </div>

                {/* Labeling Configuration */}
                <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">Feature Labeling</h4>

                  {/* Labeling Method Selector */}
                  <div className="mb-3">
                    <label className="block text-xs text-slate-400 mb-1">Labeling Method</label>
                    <select
                      value={labelingMethod}
                      onChange={(e) => setLabelingMethod(e.target.value as 'pattern' | 'local' | 'openai')}
                      className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                    >
                      <option value="pattern">Pattern Matching (fast, simple patterns)</option>
                      <option value="local">OpenAI-Compatible (Ollama, vLLM, etc.)</option>
                      <option value="openai">OpenAI API (fast, high quality, costs money)</option>
                    </select>
                    <p className="text-xs text-slate-500 mt-1">
                      {labelingMethod === 'pattern' && 'Uses 8 hardcoded patterns for quick labeling'}
                      {labelingMethod === 'local' && `Uses ${localLabelingModel || 'local LLM'} via Ollama for high-quality labeling (~5.5 hours for 16K features)`}
                      {labelingMethod === 'openai' && 'Uses GPT-4o-mini API (~55 minutes for 16K features, ~$1.64 cost)'}
                    </p>
                  </div>

                  {/* Local Model Configuration (shown when labeling_method=local) */}
                  {labelingMethod === 'local' && (
                    <div className="space-y-3">
                      {/* Endpoint URL */}
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">Endpoint URL</label>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={ollamaEndpointUrl}
                            onChange={(e) => setOllamaEndpointUrl(e.target.value)}
                            placeholder="http://mistudio.mcslab.io/ollama/"
                            className="flex-1 px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                          />
                          <button
                            type="button"
                            onClick={handleFetchOllamaModels}
                            disabled={isLoadingOllamaModels}
                            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 disabled:cursor-not-allowed text-white rounded transition-colors"
                          >
                            {isLoadingOllamaModels ? 'Fetching...' : 'Fetch Models'}
                          </button>
                        </div>
                        <p className="text-xs text-slate-500 mt-1">
                          OpenAI-compatible API endpoint (Ollama, vLLM, etc.)
                        </p>
                      </div>

                      {/* Model Name */}
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">Model Name</label>
                        {availableOllamaModels.length > 0 ? (
                          <>
                            <select
                              value={localLabelingModel}
                              onChange={(e) => setLocalLabelingModel(e.target.value)}
                              className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                            >
                              {availableOllamaModels.map((model) => (
                                <option key={model.name} value={model.name}>
                                  {model.name}
                                </option>
                              ))}
                            </select>
                            <p className="text-xs text-slate-500 mt-1">
                              {availableOllamaModels.length} model(s) available
                            </p>
                          </>
                        ) : (
                          <div className="text-xs text-slate-400 p-2 bg-slate-800 border border-slate-700 rounded">
                            Click "Fetch Models" to load available models
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* OpenAI Configuration (shown when labeling_method=openai) */}
                  {labelingMethod === 'openai' && (
                    <div className="space-y-3">
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">OpenAI API Key</label>
                        <input
                          type="password"
                          value={openaiApiKey}
                          onChange={(e) => setOpenaiApiKey(e.target.value)}
                          placeholder="sk-..."
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                        />
                        <p className="text-xs text-slate-500 mt-1">
                          Leave blank to use OPENAI_API_KEY environment variable
                        </p>
                      </div>
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">OpenAI Model</label>
                        <select
                          value={openaiModel}
                          onChange={(e) => setOpenaiModel(e.target.value as 'gpt4-mini' | 'gpt4' | 'gpt35')}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                        >
                          <option value="gpt4-mini">GPT-4o-mini (recommended, $0.0001/feature)</option>
                          <option value="gpt4">GPT-4 Turbo (higher quality, more expensive)</option>
                          <option value="gpt35">GPT-3.5 Turbo (faster, lower quality)</option>
                        </select>
                      </div>
                    </div>
                  )}
                </div>

                {/* Performance & Timeout Configuration */}
                <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">Performance & Timeout</h4>

                  <div className="space-y-3">
                    {/* Vectorization Batch Size */}
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Vectorization Batch Size</label>
                      <select
                        value={vectorizationBatchSize}
                        onChange={(e) => setVectorizationBatchSize(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                      >
                        <option value="auto">Auto (Recommended - GPU memory optimized)</option>
                        <option value="1">Sequential (1 sample at a time - slowest)</option>
                        <option value="32">32 samples (Low memory)</option>
                        <option value="64">64 samples (Balanced)</option>
                        <option value="128">128 samples (High memory)</option>
                        <option value="256">256 samples (Maximum - requires high memory)</option>
                      </select>
                      <p className="text-xs text-slate-500 mt-1">
                        Process multiple samples simultaneously for 10-15x speedup. Auto adjusts based on GPU memory.
                      </p>
                    </div>

                    {/* Time Limits */}
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">Soft Time Limit (hours)</label>
                        <input
                          type="number"
                          value={softTimeLimit}
                          onChange={(e) => setSoftTimeLimit(Number(e.target.value))}
                          min={1}
                          max={100}
                          step={1}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                        />
                        <p className="text-xs text-slate-500 mt-1">
                          Task receives warning after this time
                        </p>
                      </div>
                      <div>
                        <label className="block text-xs text-slate-400 mb-1">Hard Time Limit (hours)</label>
                        <input
                          type="number"
                          value={hardTimeLimit}
                          onChange={(e) => setHardTimeLimit(Number(e.target.value))}
                          min={1}
                          max={100}
                          step={1}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                        />
                        <p className="text-xs text-slate-500 mt-1">
                          Task terminates after this time
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Resource Configuration Panel */}
                <ResourceConfigPanel
                  trainingId={training.id}
                  evaluationSamples={evaluationSamples}
                  topKExamples={topKExamples}
                  onConfigChange={setResourceConfig}
                />

                {/* Error Message */}
                {extractionError && (
                  <div className="p-3 bg-red-900/20 border border-red-700 rounded text-red-400 text-sm">
                    {extractionError}
                  </div>
                )}
              </>
            )}
          </div>

          {/* Modal Footer */}
          <div className="sticky bottom-0 bg-slate-900 border-t border-slate-700 px-6 py-4 flex items-center justify-end gap-3">
            {showSuccessMessage ? (
              <button
                onClick={handleClose}
                className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded transition-colors"
              >
                Close
              </button>
            ) : (
              <>
                <button
                  onClick={handleClose}
                  className="px-6 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleStartExtraction}
                  disabled={isLoadingExtraction}
                  className="flex items-center gap-2 px-6 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded transition-colors"
                >
                  <Zap className="w-5 h-5" />
                  {isLoadingExtraction ? 'Starting...' : 'Start Extraction'}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

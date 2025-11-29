/**
 * StartExtractionModal Component
 *
 * Unified modal dialog for configuring and starting feature extraction.
 * Supports extraction from both:
 * - Training jobs (SAEs trained in miStudio)
 * - External SAEs (downloaded from HuggingFace)
 *
 * Workflow:
 * 1. User clicks "Start Extraction" on Extractions tab
 * 2. Selects source (Training or SAE)
 * 3. For SAE: also selects dataset
 * 4. Configures extraction parameters
 * 5. Starts extraction
 * 6. Modal shows success message
 */

import React, { useState, useEffect } from 'react';
import { Zap, X, Brain, ChevronDown } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTrainingsStore } from '../../stores/trainingsStore';
import type { Training } from '../../types/training';
import type { SAE } from '../../types/sae';
import { startSAEExtraction, SAEExtractionConfig, getReadySAEs } from '../../api/saes';

type ExtractionSourceType = 'training' | 'sae';

interface StartExtractionModalProps {
  isOpen: boolean;
  onClose: () => void;
  // Optional pre-selection
  preSelectedTraining?: Training;
  preSelectedSAE?: SAE;
}

/**
 * StartExtractionModal Component - Unified extraction modal.
 */
export const StartExtractionModal: React.FC<StartExtractionModalProps> = ({
  isOpen,
  onClose,
  preSelectedTraining,
  preSelectedSAE,
}) => {
  const {
    isLoadingExtraction,
    extractionError,
    startExtraction,
  } = useFeaturesStore();

  const { datasets, fetchDatasets } = useDatasetsStore();
  const { trainings, fetchTrainings } = useTrainingsStore();

  // Local state for SAEs (fetched directly from API to avoid affecting global store)
  const [saes, setSaes] = useState<SAE[]>([]);

  // Source selection
  const [sourceType, setSourceType] = useState<ExtractionSourceType>(
    preSelectedSAE ? 'sae' : preSelectedTraining ? 'training' : 'sae'
  );
  const [selectedTrainingId, setSelectedTrainingId] = useState<string>(preSelectedTraining?.id || '');
  const [selectedSAEId, setSelectedSAEId] = useState<string>(preSelectedSAE?.id || '');
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('');

  // Extraction config
  const [evaluationSamples, setEvaluationSamples] = useState(10000);
  const [topKExamples, setTopKExamples] = useState(100);

  // Token filtering
  const [filterSpecial, setFilterSpecial] = useState(true);
  const [filterSingleChar, setFilterSingleChar] = useState(true);
  const [filterPunctuation, setFilterPunctuation] = useState(true);
  const [filterNumbers, setFilterNumbers] = useState(true);
  const [filterFragments, setFilterFragments] = useState(true);
  const [filterStopWords, setFilterStopWords] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Context window
  const [contextPrefixTokens, setContextPrefixTokens] = useState(5);
  const [contextSuffixTokens, setContextSuffixTokens] = useState(3);
  const [showContextWindow, setShowContextWindow] = useState(false);

  // Dead neuron filtering
  const [minActivationFrequency, setMinActivationFrequency] = useState(0.001);

  // UI state
  const [showSuccessMessage, setShowSuccessMessage] = useState(false);
  const [localError, setLocalError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [saeLoadError, setSaeLoadError] = useState<string | null>(null);

  // Load data on mount
  useEffect(() => {
    if (isOpen) {
      // Fetch ready SAEs directly from API
      setSaeLoadError(null);
      getReadySAEs()
        .then((response) => {
          setSaes(response.data);
          setSaeLoadError(null);
        })
        .catch((err) => {
          console.error('Failed to fetch SAEs:', err);
          setSaes([]);
          setSaeLoadError('Failed to load SAEs. Please try again.');
        });

      // Fetch datasets and trainings from stores
      fetchDatasets();
      fetchTrainings();
    }
  }, [isOpen, fetchDatasets, fetchTrainings]);

  // Filter completed trainings
  const completedTrainings = trainings.filter(t => t.status === 'completed');

  // SAEs are already filtered to ready status by the API call
  const readySAEs = saes;

  // Filter ready datasets
  const readyDatasets = datasets.filter(d => d.status === 'ready');

  // Get selected entities
  const selectedTraining = completedTrainings.find(t => t.id === selectedTrainingId);
  const selectedSAE = readySAEs.find(s => s.id === selectedSAEId);

  // Look up human-readable names from stores (training only has IDs, not names)
  const getDatasetName = (datasetId: string) => {
    const dataset = datasets.find(d => d.id === datasetId);
    return dataset?.name || datasetId;
  };

  /**
   * Handle start extraction.
   */
  const handleStartExtraction = async () => {
    setLocalError(null);
    setIsSubmitting(true);

    try {
      const config: SAEExtractionConfig = {
        evaluation_samples: evaluationSamples,
        top_k_examples: topKExamples,
        filter_special: filterSpecial,
        filter_single_char: filterSingleChar,
        filter_punctuation: filterPunctuation,
        filter_numbers: filterNumbers,
        filter_fragments: filterFragments,
        filter_stop_words: filterStopWords,
        context_prefix_tokens: contextPrefixTokens,
        context_suffix_tokens: contextSuffixTokens,
        min_activation_frequency: minActivationFrequency,
      };

      if (sourceType === 'training') {
        if (!selectedTrainingId) {
          throw new Error('Please select a training');
        }
        await startExtraction(selectedTrainingId, config as any);
      } else {
        if (!selectedSAEId) {
          throw new Error('Please select an SAE');
        }
        if (!selectedDatasetId) {
          throw new Error('Please select a dataset');
        }
        await startSAEExtraction(selectedSAEId, selectedDatasetId, config);
      }

      setShowSuccessMessage(true);
    } catch (error: any) {
      const errorMessage = error.response?.data?.detail || error.message || 'Failed to start extraction';
      setLocalError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  /**
   * Handle modal close.
   */
  const handleClose = () => {
    setShowSuccessMessage(false);
    setLocalError(null);
    onClose();
  };

  // Don't render if not open
  if (!isOpen) {
    return null;
  }

  const displayError = localError || extractionError;

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
                Extract interpretable features from an SAE
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
                  Extraction started successfully!
                </p>
                <p className="text-sm text-slate-300">
                  The extraction job is now queued. You can monitor progress on this page.
                </p>
              </div>
            )}

            {/* Configuration Form (only show if not successful) */}
            {!showSuccessMessage && (
              <>
                {/* Source Type Selection */}
                <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                  <h4 className="text-sm font-semibold text-slate-300 mb-3">Extraction Source</h4>
                  <div className="grid grid-cols-2 gap-3">
                    <button
                      type="button"
                      onClick={() => setSourceType('sae')}
                      className={`p-4 rounded-lg border-2 transition-all flex flex-col items-center gap-2 ${
                        sourceType === 'sae'
                          ? 'border-emerald-500 bg-emerald-900/20'
                          : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                      }`}
                    >
                      <Brain className={`w-8 h-8 ${sourceType === 'sae' ? 'text-emerald-400' : 'text-slate-400'}`} />
                      <span className={`font-medium ${sourceType === 'sae' ? 'text-emerald-400' : 'text-slate-300'}`}>
                        SAE
                      </span>
                      <span className="text-xs text-slate-500 text-center">
                        External or imported SAE
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setSourceType('training')}
                      className={`p-4 rounded-lg border-2 transition-all flex flex-col items-center gap-2 ${
                        sourceType === 'training'
                          ? 'border-emerald-500 bg-emerald-900/20'
                          : 'border-slate-700 bg-slate-800/50 hover:border-slate-600'
                      }`}
                    >
                      <Zap className={`w-8 h-8 ${sourceType === 'training' ? 'text-emerald-400' : 'text-slate-400'}`} />
                      <span className={`font-medium ${sourceType === 'training' ? 'text-emerald-400' : 'text-slate-300'}`}>
                        Training
                      </span>
                      <span className="text-xs text-slate-500 text-center">
                        Completed training job
                      </span>
                    </button>
                  </div>
                </div>

                {/* Source Selection */}
                {sourceType === 'sae' ? (
                  <div className="space-y-3">
                    {/* SAE Selection */}
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Select SAE</label>
                      <select
                        value={selectedSAEId}
                        onChange={(e) => setSelectedSAEId(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                      >
                        <option value="">-- Select an SAE --</option>
                        {readySAEs.map((sae) => (
                          <option key={sae.id} value={sae.id}>
                            {sae.name} ({sae.model_name || 'Unknown model'}, Layer {sae.layer ?? '?'})
                          </option>
                        ))}
                      </select>
                      {saeLoadError && (
                        <p className="text-xs text-red-400 mt-1">{saeLoadError}</p>
                      )}
                      {!saeLoadError && readySAEs.length === 0 && (
                        <p className="text-xs text-amber-400 mt-1">
                          No ready SAEs available. Download or import an SAE first.
                        </p>
                      )}
                    </div>

                    {/* Dataset Selection (required for SAE) */}
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Select Dataset</label>
                      <select
                        value={selectedDatasetId}
                        onChange={(e) => setSelectedDatasetId(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                      >
                        <option value="">-- Select a dataset --</option>
                        {readyDatasets.map((dataset) => (
                          <option key={dataset.id} value={dataset.id}>
                            {dataset.name}
                          </option>
                        ))}
                      </select>
                      {readyDatasets.length === 0 && (
                        <p className="text-xs text-amber-400 mt-1">
                          No ready datasets available. Download a dataset first.
                        </p>
                      )}
                    </div>

                    {/* Selected SAE Info */}
                    {selectedSAE && (
                      <div className="p-3 bg-slate-800/30 border border-slate-700 rounded text-sm">
                        <div className="grid grid-cols-2 gap-2 text-slate-400">
                          <span>Model:</span>
                          <span className="text-slate-200">{selectedSAE.model_name || 'Unknown'}</span>
                          <span>Layer:</span>
                          <span className="text-slate-200">{selectedSAE.layer ?? 'Unknown'}</span>
                          <span>Features:</span>
                          <span className="text-slate-200">{selectedSAE.n_features?.toLocaleString() ?? 'Unknown'}</span>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="space-y-3">
                    {/* Training Selection */}
                    <div>
                      <label className="block text-xs text-slate-400 mb-1">Select Training</label>
                      <select
                        value={selectedTrainingId}
                        onChange={(e) => setSelectedTrainingId(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                      >
                        <option value="">-- Select a training --</option>
                        {completedTrainings.map((training) => {
                          const datasetName = getDatasetName(training.dataset_id);
                          const modelId = training.model_id?.slice(0, 8) || 'Unknown';
                          return (
                            <option key={training.id} value={training.id}>
                              {`${training.id.slice(0, 8)} - ${modelId} / ${datasetName}`}
                            </option>
                          );
                        })}
                      </select>
                      {completedTrainings.length === 0 && (
                        <p className="text-xs text-amber-400 mt-1">
                          No completed trainings available.
                        </p>
                      )}
                    </div>

                    {/* Selected Training Info */}
                    {selectedTraining && (
                      <div className="p-3 bg-slate-800/30 border border-slate-700 rounded text-sm">
                        <div className="grid grid-cols-2 gap-2 text-slate-400">
                          <span>Model:</span>
                          <span className="text-slate-200">{selectedTraining.model_id}</span>
                          <span>Dataset:</span>
                          <span className="text-slate-200">{getDatasetName(selectedTraining.dataset_id)}</span>
                          <span>Latent Dim:</span>
                          <span className="text-slate-200">{selectedTraining.hyperparameters?.latent_dim?.toLocaleString() ?? 'Unknown'}</span>
                        </div>
                      </div>
                    )}
                  </div>
                )}

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

                {/* Dead Neuron Filtering */}
                <div className="p-4 bg-slate-800/50 border border-slate-700 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <label className="text-sm font-medium text-slate-300">Dead Neuron Filtering</label>
                    <span className="text-xs text-emerald-500">
                      {(minActivationFrequency * 100).toFixed(2)}% min frequency
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 mb-3">
                    Neurons firing less than this threshold are considered "dead" and will be filtered out.
                  </p>
                  <div className="flex items-center gap-3">
                    <input
                      type="range"
                      value={minActivationFrequency * 1000}
                      onChange={(e) => setMinActivationFrequency(Number(e.target.value) / 1000)}
                      min={0}
                      max={10}
                      step={0.1}
                      className="flex-1 h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-500"
                    />
                    <input
                      type="number"
                      value={(minActivationFrequency * 100).toFixed(2)}
                      onChange={(e) => setMinActivationFrequency(Number(e.target.value) / 100)}
                      min={0}
                      max={10}
                      step={0.01}
                      className="w-20 px-2 py-1 bg-slate-800 border border-slate-700 rounded text-white text-sm focus:outline-none focus:border-emerald-500"
                    />
                    <span className="text-xs text-slate-400">%</span>
                  </div>
                </div>

                {/* Context Window Configuration */}
                <div className="bg-slate-900 rounded-lg border border-slate-700 p-4">
                  <button
                    type="button"
                    onClick={() => setShowContextWindow(!showContextWindow)}
                    className="flex items-center justify-between w-full text-left"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium text-slate-300">Context Window</span>
                      <span className="text-xs text-emerald-500">
                        ({contextPrefixTokens} prefix + prime + {contextSuffixTokens} suffix)
                      </span>
                    </div>
                    <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${showContextWindow ? 'rotate-180' : ''}`} />
                  </button>

                  {showContextWindow && (
                    <div className="mt-4 space-y-3">
                      <p className="text-xs text-slate-400">
                        Capture tokens before and after the prime token (max activation) to provide context for interpretation.
                      </p>
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">Prefix Tokens</label>
                          <input
                            type="number"
                            value={contextPrefixTokens}
                            onChange={(e) => setContextPrefixTokens(Number(e.target.value))}
                            min={0}
                            max={20}
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                        <div>
                          <label className="block text-xs text-slate-400 mb-1">Suffix Tokens</label>
                          <input
                            type="number"
                            value={contextSuffixTokens}
                            onChange={(e) => setContextSuffixTokens(Number(e.target.value))}
                            min={0}
                            max={20}
                            className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:outline-none focus:border-emerald-500"
                          />
                        </div>
                      </div>
                    </div>
                  )}
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
                    <ChevronDown className={`w-4 h-4 text-slate-400 transition-transform ${showFilters ? 'rotate-180' : ''}`} />
                  </button>

                  {showFilters && (
                    <div className="mt-4 grid grid-cols-2 gap-3">
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterSpecial}
                          onChange={(e) => setFilterSpecial(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span>Special tokens</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterSingleChar}
                          onChange={(e) => setFilterSingleChar(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span>Single characters</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterPunctuation}
                          onChange={(e) => setFilterPunctuation(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span>Punctuation</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterNumbers}
                          onChange={(e) => setFilterNumbers(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span>Numbers</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterFragments}
                          onChange={(e) => setFilterFragments(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span>Word fragments</span>
                      </label>
                      <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={filterStopWords}
                          onChange={(e) => setFilterStopWords(e.target.checked)}
                          className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500"
                        />
                        <span>Stop words</span>
                      </label>
                    </div>
                  )}
                </div>

                {/* Error Message */}
                {displayError && (
                  <div className="p-3 bg-red-900/20 border border-red-700 rounded text-red-400 text-sm">
                    {displayError}
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
                  disabled={isSubmitting || isLoadingExtraction || (sourceType === 'sae' ? !selectedSAEId || !selectedDatasetId : !selectedTrainingId)}
                  className="flex items-center gap-2 px-6 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 text-white rounded transition-colors"
                >
                  <Zap className="w-5 h-5" />
                  {isSubmitting || isLoadingExtraction ? 'Starting...' : 'Start Extraction'}
                </button>
              </>
            )}
          </div>
        </div>
      </div>
    </>
  );
};

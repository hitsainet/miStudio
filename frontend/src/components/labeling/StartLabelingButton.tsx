/**
 * StartLabelingButton Component
 *
 * Button to trigger semantic labeling for an extraction job.
 * Opens a modal to configure labeling options.
 */

import React, { useState, useEffect } from 'react';
import { Tag, AlertCircle } from 'lucide-react';
import { useLabelingStore } from '../../stores/labelingStore';
import { useLabelingPromptTemplatesStore } from '../../stores/labelingPromptTemplatesStore';
import { LabelingMethod } from '../../types/labeling';
import { getLocalModels } from '../../api/models';

interface StartLabelingButtonProps {
  extractionId: string;
  disabled?: boolean;
  onSuccess?: () => void;
}

export const StartLabelingButton: React.FC<StartLabelingButtonProps> = ({
  extractionId,
  disabled = false,
  onSuccess,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [labelingMethod, setLabelingMethod] = useState<LabelingMethod>(
    LabelingMethod.OPENAI
  );
  const [openaiModel, setOpenaiModel] = useState('gpt-4o-mini');
  const [openaiApiKey, setOpenaiApiKey] = useState('');
  const [openaiCompatibleEndpoint, setOpenaiCompatibleEndpoint] = useState('http://mistudio.mcslab.io/ollama/v1');
  const [openaiCompatibleModel, setOpenaiCompatibleModel] = useState('gemma2:2b');
  const [localModel, setLocalModel] = useState('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [compatibleModels, setCompatibleModels] = useState<string[]>([]);
  const [loadingCompatibleModels, setLoadingCompatibleModels] = useState(false);
  const [compatibleModelsError, setCompatibleModelsError] = useState<string | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [maxExamples, setMaxExamples] = useState<number>(25);  // Number of examples per feature (10-50)
  const [apiTimeout, setApiTimeout] = useState<number>(120);  // API timeout in seconds (30-600)

  // Token filtering configuration
  const [filterSpecial, setFilterSpecial] = useState(true);
  const [filterSingleChar, setFilterSingleChar] = useState(true);
  const [filterPunctuation, setFilterPunctuation] = useState(true);
  const [filterNumbers, setFilterNumbers] = useState(true);
  const [filterFragments, setFilterFragments] = useState(true);
  const [filterStopWords, setFilterStopWords] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Debugging configuration
  const [saveRequestsForTesting, setSaveRequestsForTesting] = useState(false);
  const [saveRequestsSampleRate, setSaveRequestsSampleRate] = useState(100);  // Store as percentage 0-100
  const [exportFormat, setExportFormat] = useState<'postman' | 'curl' | 'both'>('both');
  const [savePoorQualityLabels, setSavePoorQualityLabels] = useState(false);
  const [poorQualitySampleRate, setPoorQualitySampleRate] = useState(100);  // Store as percentage 0-100

  const { startLabeling, isLoading, error, clearError } = useLabelingStore();
  const { templates, fetchTemplates } = useLabelingPromptTemplatesStore();

  // Fetch templates when modal opens
  useEffect(() => {
    if (isOpen && templates.length === 0) {
      fetchTemplates();
    }
  }, [isOpen]);

  // Fetch locally cached models when modal opens
  useEffect(() => {
    if (isOpen && labelingMethod === LabelingMethod.LOCAL && availableModels.length === 0) {
      setLoadingModels(true);
      getLocalModels()
        .then((response) => {
          setAvailableModels(response.models);
          // Set first model as default if none selected
          if (!localModel && response.models.length > 0) {
            setLocalModel(response.models[0]);
          }
        })
        .catch((err) => {
          console.error('Failed to fetch local models:', err);
        })
        .finally(() => {
          setLoadingModels(false);
        });
    }
  }, [isOpen, labelingMethod, availableModels.length, localModel]);

  // Fetch models from OpenAI-compatible endpoint
  const fetchCompatibleModels = async () => {
    if (!openaiCompatibleEndpoint) {
      setCompatibleModelsError('Please enter an endpoint URL first');
      return;
    }

    setLoadingCompatibleModels(true);
    setCompatibleModelsError(null);

    try {
      // Normalize endpoint URL (remove trailing slash, ensure /v1 suffix for consistency)
      let baseUrl = openaiCompatibleEndpoint.trim();
      if (baseUrl.endsWith('/')) {
        baseUrl = baseUrl.slice(0, -1);
      }
      // Ensure URL ends with /v1 for OpenAI-compatible API standard
      if (!baseUrl.endsWith('/v1')) {
        baseUrl = `${baseUrl}/v1`;
      }

      // Query the /v1/models endpoint
      const response = await fetch(`${baseUrl}/models`);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      // OpenAI API format: { "data": [{ "id": "model-name", ... }] }
      // Ollama format: { "models": [{ "name": "model-name", ... }] }
      let models: string[] = [];

      if (data.data && Array.isArray(data.data)) {
        // OpenAI/vLLM format
        models = data.data.map((m: any) => m.id || m.name).filter(Boolean);
      } else if (data.models && Array.isArray(data.models)) {
        // Ollama format
        models = data.models.map((m: any) => m.name || m.id).filter(Boolean);
      } else {
        throw new Error('Unexpected response format from endpoint');
      }

      if (models.length === 0) {
        setCompatibleModelsError('No models found on this endpoint');
      } else {
        setCompatibleModels(models);
        // Auto-select first model if current value is default or empty
        if (!openaiCompatibleModel || openaiCompatibleModel === 'gemma2:2b' || openaiCompatibleModel === 'llama3.2') {
          setOpenaiCompatibleModel(models[0]);
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch models';
      setCompatibleModelsError(errorMessage);
      console.error('Failed to fetch compatible models:', err);
    } finally {
      setLoadingCompatibleModels(false);
    }
  };

  const handleStartLabeling = async () => {
    try {
      clearError();

      await startLabeling({
        extraction_job_id: extractionId,
        labeling_method: labelingMethod,
        openai_model: labelingMethod === LabelingMethod.OPENAI ? openaiModel : undefined,
        openai_api_key:
          labelingMethod === LabelingMethod.OPENAI && openaiApiKey
            ? openaiApiKey
            : undefined,
        openai_compatible_endpoint:
          labelingMethod === LabelingMethod.OPENAI_COMPATIBLE ? openaiCompatibleEndpoint : undefined,
        openai_compatible_model:
          labelingMethod === LabelingMethod.OPENAI_COMPATIBLE ? openaiCompatibleModel : undefined,
        local_model: labelingMethod === LabelingMethod.LOCAL ? localModel : undefined,
        prompt_template_id: selectedTemplateId || undefined,
        batch_size: 10,
        max_examples: maxExamples,  // Number of examples per feature
        api_timeout: apiTimeout,  // API request timeout in seconds
        // Token filtering configuration
        filter_special: filterSpecial,
        filter_single_char: filterSingleChar,
        filter_punctuation: filterPunctuation,
        filter_numbers: filterNumbers,
        filter_fragments: filterFragments,
        filter_stop_words: filterStopWords,
        // Debugging configuration
        save_requests_for_testing: saveRequestsForTesting,
        save_requests_sample_rate: saveRequestsForTesting ? saveRequestsSampleRate / 100 : undefined,  // Convert 0-100 to 0.0-1.0
        export_format: saveRequestsForTesting ? exportFormat : undefined,
        save_poor_quality_labels: savePoorQualityLabels,
        poor_quality_sample_rate: savePoorQualityLabels ? poorQualitySampleRate / 100 : undefined,  // Convert 0-100 to 0.0-1.0
      });

      setIsOpen(false);
      onSuccess?.();
    } catch (err) {
      console.error('Failed to start labeling:', err);
    }
  };

  return (
    <>
      <button
        onClick={() => setIsOpen(true)}
        disabled={disabled || isLoading}
        className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
      >
        <Tag className="w-4 h-4" />
        Label Features
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-slate-800 rounded-lg p-6 w-full max-w-5xl border border-slate-700 max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-semibold text-white mb-4">
              Start Semantic Labeling
            </h2>

            {error && (
              <div className="mb-4 p-3 bg-red-900/20 border border-red-800 rounded-lg flex items-start gap-2">
                <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                <div className="text-sm text-red-200">{error}</div>
              </div>
            )}

            <div className="grid grid-cols-2 gap-6">
              {/* LEFT COLUMN */}
              <div className="flex flex-col">
                <div className="space-y-4">
                  {/* Prompt Template */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Prompt Template
                    </label>
                    <select
                      value={selectedTemplateId || ''}
                      onChange={(e) => setSelectedTemplateId(e.target.value || null)}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      <option value="">Use Default Template</option>
                      {templates.map((template) => (
                        <option key={template.id} value={template.id}>
                          {template.name}
                          {template.is_default && ' (Default)'}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-slate-400">
                      Select a prompt template for labeling features
                    </p>
                  </div>

                  {/* Labeling Method */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Labeling Method
                    </label>
                    <select
                      value={labelingMethod}
                      onChange={(e) => setLabelingMethod(e.target.value as LabelingMethod)}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      <option value={LabelingMethod.OPENAI}>OpenAI (requires api-key)</option>
                      <option value={LabelingMethod.OPENAI_COMPATIBLE}>OpenAI-Compatible (Ollama, vLLM, etc.)</option>
                      <option value={LabelingMethod.LOCAL}>Local Model</option>
                    </select>
                  </div>

                  {/* OpenAI Model (when OpenAI selected) */}
                  {labelingMethod === LabelingMethod.OPENAI && (
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">
                        OpenAI Model
                      </label>
                      <select
                        value={openaiModel}
                        onChange={(e) => setOpenaiModel(e.target.value)}
                        className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      >
                        <option value="gpt-4o-mini">gpt-4o-mini (recommended)</option>
                        <option value="gpt-4o">gpt-4o</option>
                        <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                      </select>
                    </div>
                  )}

                  {/* OpenAI-Compatible Endpoint + Model (when OpenAI-Compatible selected) */}
                  {labelingMethod === LabelingMethod.OPENAI_COMPATIBLE && (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                          Endpoint URL
                        </label>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={openaiCompatibleEndpoint}
                            onChange={(e) => {
                              setOpenaiCompatibleEndpoint(e.target.value);
                              setCompatibleModels([]);
                              setCompatibleModelsError(null);
                            }}
                            placeholder="http://ollama.mcslab.io/v1"
                            className="flex-1 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                          />
                          <button
                            type="button"
                            onClick={fetchCompatibleModels}
                            disabled={loadingCompatibleModels || !openaiCompatibleEndpoint}
                            className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                          >
                            {loadingCompatibleModels ? 'Loading...' : 'Fetch Models'}
                          </button>
                        </div>
                        <p className="mt-1 text-xs text-slate-400">
                          OpenAI-compatible API endpoint (Ollama, vLLM, etc.)
                        </p>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                          Model Name
                        </label>
                        {compatibleModels.length > 0 ? (
                          <select
                            value={openaiCompatibleModel}
                            onChange={(e) => setOpenaiCompatibleModel(e.target.value)}
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                          >
                            {compatibleModels.map((model) => (
                              <option key={model} value={model}>
                                {model}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <input
                            type="text"
                            value={openaiCompatibleModel}
                            onChange={(e) => setOpenaiCompatibleModel(e.target.value)}
                            placeholder="llama3.2"
                            className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                          />
                        )}
                        {compatibleModelsError && (
                          <p className="mt-1 text-xs text-red-400">{compatibleModelsError}</p>
                        )}
                        {!compatibleModelsError && (
                          <p className="mt-1 text-xs text-slate-400">
                            {compatibleModels.length > 0
                              ? `${compatibleModels.length} model(s) available`
                              : 'Click "Fetch Models" or enter model name manually'}
                          </p>
                        )}
                      </div>
                    </>
                  )}

                  {/* Local Model (when Local selected) */}
                  {labelingMethod === LabelingMethod.LOCAL && (
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">
                        Local Model
                      </label>
                      {loadingModels ? (
                        <div className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-400">
                          Loading models...
                        </div>
                      ) : availableModels.length > 0 ? (
                        <select
                          value={localModel}
                          onChange={(e) => setLocalModel(e.target.value)}
                          className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        >
                          {availableModels.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <div className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-slate-400">
                          No local models found
                        </div>
                      )}
                      <p className="mt-1 text-xs text-slate-400">
                        Locally cached HuggingFace models
                      </p>
                    </div>
                  )}
                </div>

                {/* Token Filtering - pushed to bottom */}
                <div className="mt-auto pt-4">
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
                </div>
              </div>

              {/* RIGHT COLUMN */}
              <div className="flex flex-col">
                <div className="space-y-4">
                  {/* Examples Per Feature */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      Examples Per Feature
                    </label>
                    <input
                      type="number"
                      min={10}
                      max={50}
                      value={maxExamples}
                      onChange={(e) => setMaxExamples(parseInt(e.target.value) || 25)}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-400">
                      Number of high-activation examples to show the LLM (10-50, default: 25)
                    </p>
                  </div>

                  {/* OpenAI API Key (when OpenAI selected) */}
                  {labelingMethod === LabelingMethod.OPENAI && (
                    <div>
                      <label className="block text-sm font-medium text-slate-300 mb-2">
                        OpenAI API Key (optional)
                      </label>
                      <input
                        type="password"
                        value={openaiApiKey}
                        onChange={(e) => setOpenaiApiKey(e.target.value)}
                        placeholder="Uses server default if not provided"
                        className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                      <p className="mt-1 text-xs text-slate-400">
                        Leave blank to use the server's configured API key
                      </p>
                    </div>
                  )}

                  {/* API Timeout */}
                  <div>
                    <label className="block text-sm font-medium text-slate-300 mb-2">
                      API Timeout (seconds)
                    </label>
                    <input
                      type="number"
                      min={30}
                      max={600}
                      value={apiTimeout}
                      onChange={(e) => setApiTimeout(parseInt(e.target.value) || 120)}
                      className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-400">
                      Maximum wait time for API responses (30-600 seconds, default: 120). Increase for larger models.
                    </p>
                  </div>
                </div>

                {/* Save API Requests - pushed to bottom */}
                <div className="mt-auto pt-4">
                  <div className="bg-slate-900 rounded-lg border border-slate-700 p-4 space-y-3">
                    <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={saveRequestsForTesting}
                        onChange={(e) => setSaveRequestsForTesting(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                      />
                      <div className="flex flex-col">
                        <span className="font-medium">Save API requests for testing</span>
                        <span className="text-xs text-slate-400">Save requests to /tmp/ for debugging with Postman/cURL</span>
                      </div>
                    </label>

                    {/* Export Format - only shown when Save API requests is enabled */}
                    {saveRequestsForTesting && (
                      <div className="ml-6 pt-2 border-t border-slate-700">
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                          Export Format
                        </label>
                        <select
                          value={exportFormat}
                          onChange={(e) => setExportFormat(e.target.value as 'postman' | 'curl' | 'both')}
                          className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        >
                          <option value="both">Both (Postman + cURL)</option>
                          <option value="postman">Postman Collection Only</option>
                          <option value="curl">cURL Command Only</option>
                        </select>
                        <p className="mt-1 text-xs text-slate-400">
                          Choose which format(s) to save for debugging
                        </p>
                      </div>
                    )}

                    {/* Sample Rate Slider - only shown when Save API requests is enabled */}
                    {saveRequestsForTesting && (
                      <div className="ml-6 pt-2 border-t border-slate-700">
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                          Sample Rate: {saveRequestsSampleRate}%
                        </label>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={5}
                          value={saveRequestsSampleRate}
                          onChange={(e) => setSaveRequestsSampleRate(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                        />
                        <p className="mt-1 text-xs text-slate-400">
                          Percentage of all requests to save (0% = none, 100% = all). Lower values reduce disk usage.
                        </p>
                      </div>
                    )}

                    {/* Poor Quality Detection - in same debugging card */}
                    <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer pt-3 border-t border-slate-700">
                      <input
                        type="checkbox"
                        checked={savePoorQualityLabels}
                        onChange={(e) => setSavePoorQualityLabels(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
                      />
                      <div className="flex flex-col">
                        <span className="font-medium">Save poor quality labels for debugging</span>
                        <span className="text-xs text-slate-400">Detect and save ineffective labels like "uncategorized" or generic responses</span>
                      </div>
                    </label>

                    {/* Sample Rate Slider - only shown when Save poor quality labels is enabled */}
                    {savePoorQualityLabels && (
                      <div className="ml-6 pt-2 border-t border-slate-700">
                        <label className="block text-sm font-medium text-slate-300 mb-2">
                          Sample Rate: {poorQualitySampleRate}%
                        </label>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={5}
                          value={poorQualitySampleRate}
                          onChange={(e) => setPoorQualitySampleRate(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                        />
                        <p className="mt-1 text-xs text-slate-400">
                          Percentage of poor quality labels to save (0% = none, 100% = all). Lower values reduce disk usage.
                        </p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 flex gap-3">
              <button
                onClick={handleStartLabeling}
                disabled={isLoading}
                className="flex-1 px-4 py-2 bg-emerald-600 text-white rounded-lg hover:bg-emerald-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors"
              >
                {isLoading ? 'Starting...' : 'Start Labeling'}
              </button>
              <button
                onClick={() => {
                  setIsOpen(false);
                  clearError();
                }}
                disabled={isLoading}
                className="px-4 py-2 bg-slate-700 text-white rounded-lg hover:bg-slate-600 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

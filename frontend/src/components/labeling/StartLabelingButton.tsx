/**
 * StartLabelingButton Component
 *
 * Button to trigger semantic labeling for an extraction job.
 * Opens a modal to configure labeling options.
 */

import React, { useState, useEffect, useRef } from 'react';
import { Tag, AlertCircle, Save, Trash2, ChevronDown } from 'lucide-react';
import { useLabelingStore } from '../../stores/labelingStore';
import { useLabelingPromptTemplatesStore } from '../../stores/labelingPromptTemplatesStore';
import { useSettingsStore } from '../../stores/settingsStore';
import { LabelingMethod } from '../../types/labeling';
import { getLocalModels } from '../../api/models';
import { fetchAPI } from '../../api/client';

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
  const [openaiCompatibleEndpoint, setOpenaiCompatibleEndpoint] = useState('/ollama/v1');
  const [openaiCompatibleModel, setOpenaiCompatibleModel] = useState('gemma2:2b');
  const [localModel, setLocalModel] = useState('');
  const [availableModels, setAvailableModels] = useState<string[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [compatibleModels, setCompatibleModels] = useState<string[]>([]);
  const [loadingCompatibleModels, setLoadingCompatibleModels] = useState(false);
  const [compatibleModelsError, setCompatibleModelsError] = useState<string | null>(null);
  const [openaiModels, setOpenaiModels] = useState<string[]>([]);
  const [loadingOpenaiModels, setLoadingOpenaiModels] = useState(false);
  const [openaiModelsError, setOpenaiModelsError] = useState<string | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [batchSize, setBatchSize] = useState<number>(10);  // Features per batch (1-100)
  const [maxExamples, setMaxExamples] = useState<number>(25);  // Number of examples per feature (10-50)
  const [maxTokens, setMaxTokens] = useState<number>(300);  // Max tokens in LLM response (50-8000)
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

  const [showSavedEndpoints, setShowSavedEndpoints] = useState(false);
  const savedEndpointsRef = useRef<HTMLDivElement>(null);

  const { startLabeling, isLoading, error, clearError } = useLabelingStore();
  const { templates, fetchTemplates } = useLabelingPromptTemplatesStore();
  const { settings, upsert: upsertSetting, remove: removeSetting, fetchAll: fetchSettings } = useSettingsStore();

  // Sync openai-compatible endpoint/model from shared settings keys (same as Settings → Endpoints)
  const compatEndpointFromSettings = settings.find((s) => s.key === 'openai_compatible_endpoint')?.value;
  const compatModelFromSettings = settings.find((s) => s.key === 'openai_compatible_model')?.value;
  useEffect(() => {
    if (compatEndpointFromSettings) setOpenaiCompatibleEndpoint(compatEndpointFromSettings);
  }, [compatEndpointFromSettings]);
  useEffect(() => {
    if (compatModelFromSettings) setOpenaiCompatibleModel(compatModelFromSettings);
  }, [compatModelFromSettings]);

  // Derive saved endpoints from DB-backed settings store
  const savedEndpoints = settings
    .filter((s) => s.category === 'endpoints')
    .map((s) => {
      try {
        const parsed = JSON.parse(s.value) as { url: string; label?: string; lastUsed?: number };
        return { ...parsed, _key: s.key };
      } catch {
        return { url: s.value, _key: s.key };
      }
    });

  const addEndpoint = (url: string, label?: string) => {
    const key = `endpoint:${url}`;
    upsertSetting({
      key,
      value: JSON.stringify({ url, label: label || undefined, lastUsed: Date.now() }),
      is_sensitive: false,
      category: 'endpoints',
    });
  };

  const removeEndpoint = (url: string) => {
    const key = `endpoint:${url}`;
    removeSetting(key);
  };

  const touchEndpoint = (url: string) => {
    const ep = savedEndpoints.find((e) => e.url === url);
    if (ep) {
      addEndpoint(url, ep.label);
    }
  };

  // Close saved endpoints dropdown on outside click
  useEffect(() => {
    if (!showSavedEndpoints) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (savedEndpointsRef.current && !savedEndpointsRef.current.contains(e.target as Node)) {
        setShowSavedEndpoints(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showSavedEndpoints]);

  // Fetch templates and settings when modal opens
  useEffect(() => {
    if (isOpen) {
      if (templates.length === 0) fetchTemplates();
      fetchSettings();
    }
  }, [isOpen]);

  // Initialize batch size from DB settings default
  useEffect(() => {
    if (isOpen) {
      const defaultBatchSizeSetting = settings.find(s => s.key === 'labeling_default_batch_size');
      if (defaultBatchSizeSetting) {
        const parsed = parseInt(defaultBatchSizeSetting.value);
        if (!isNaN(parsed) && parsed >= 1 && parsed <= 100) {
          setBatchSize(parsed);
        }
      }
    }
  }, [isOpen, settings]);

  // Auto-select default template when templates are loaded
  useEffect(() => {
    if (templates.length > 0 && selectedTemplateId === null) {
      const defaultTemplate = templates.find(t => t.is_default);
      if (defaultTemplate) {
        setSelectedTemplateId(defaultTemplate.id);
      }
    }
  }, [templates]);

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

  // Fetch models from OpenAI-compatible endpoint (always proxied through backend)
  const fetchCompatibleModels = async () => {
    if (!openaiCompatibleEndpoint) {
      setCompatibleModelsError('Please enter an endpoint URL first');
      return;
    }

    setLoadingCompatibleModels(true);
    setCompatibleModelsError(null);

    try {
      const endpoint = openaiCompatibleEndpoint.trim();

      // Resolve relative URLs to absolute using current origin
      // e.g. "/ollama/v1" → "<origin>/ollama/v1"
      let resolvedUrl = endpoint;
      if (!endpoint.startsWith('http://') && !endpoint.startsWith('https://')) {
        resolvedUrl = `${window.location.origin}${endpoint.startsWith('/') ? '' : '/'}${endpoint}`;
      }

      // All fetching goes through backend to avoid CORS and K8s DNS issues
      const data = await fetchAPI<{ models: { id: string }[]; total: number }>('/labeling/models/openai', {
        method: 'POST',
        body: JSON.stringify({ endpoint_url: resolvedUrl }),
      });

      const models = (data.models || []).map((m) => m.id).filter(Boolean);

      if (models.length === 0) {
        setCompatibleModelsError('No models found on this endpoint.');
      } else {
        setCompatibleModels(models);
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

  // Fetch models from OpenAI API (proxied through backend to avoid CORS)
  const fetchOpenAIModels = async () => {
    setLoadingOpenaiModels(true);
    setOpenaiModelsError(null);

    try {
      const data = await fetchAPI<{ models: { id: string }[]; total: number }>('/labeling/models/openai', {
        method: 'POST',
        body: JSON.stringify({
          api_key: openaiApiKey || undefined,
          endpoint_url: 'https://api.openai.com/v1',
        }),
      });

      const models = (data.models || []).map((m) => m.id).filter(Boolean);

      if (models.length === 0) {
        setOpenaiModelsError('No models returned from OpenAI API.');
      } else {
        setOpenaiModels(models);
        // Keep current selection if it's in the list, otherwise select first
        if (!models.includes(openaiModel)) {
          const preferred = models.find((m) => m === 'gpt-4o-mini') || models[0];
          setOpenaiModel(preferred);
        }
      }
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to fetch models';
      setOpenaiModelsError(errorMessage);
      console.error('Failed to fetch OpenAI models:', err);
    } finally {
      setLoadingOpenaiModels(false);
    }
  };

  const handleStartLabeling = async () => {
    try {
      clearError();

      // Auto-save endpoint URL when starting a job
      if (labelingMethod === LabelingMethod.OPENAI_COMPATIBLE && openaiCompatibleEndpoint.trim()) {
        addEndpoint(openaiCompatibleEndpoint.trim());
        // Keep shared settings keys in sync so Settings → Endpoints reflects the latest values
        upsertSetting({ key: 'openai_compatible_endpoint', value: openaiCompatibleEndpoint.trim(), is_sensitive: false, category: 'endpoints' });
        upsertSetting({ key: 'openai_compatible_model', value: openaiCompatibleModel.trim(), is_sensitive: false, category: 'endpoints' });
      }

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
        batch_size: batchSize,
        max_examples: maxExamples,  // Number of examples per feature
        max_tokens: maxTokens,  // Max tokens in LLM response
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
          <div className="bg-white dark:bg-slate-800 rounded-lg p-6 w-full max-w-5xl border border-slate-200 dark:border-slate-700 max-h-[90vh] overflow-y-auto">
            <h2 className="text-xl font-semibold text-slate-900 dark:text-white mb-4">
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
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Prompt Template
                    </label>
                    <select
                      value={selectedTemplateId || ''}
                      onChange={(e) => setSelectedTemplateId(e.target.value || null)}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      <option value="">Use Default Template</option>
                      {templates.map((template) => (
                        <option key={template.id} value={template.id}>
                          {template.name}
                          {template.is_default && ' (Default)'}
                        </option>
                      ))}
                    </select>
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                      Select a prompt template for labeling features
                    </p>
                  </div>

                  {/* Labeling Method */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Labeling Method
                    </label>
                    <select
                      value={labelingMethod}
                      onChange={(e) => setLabelingMethod(e.target.value as LabelingMethod)}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    >
                      <option value={LabelingMethod.OPENAI}>OpenAI (requires api-key)</option>
                      <option value={LabelingMethod.OPENAI_COMPATIBLE}>OpenAI-Compatible (miLLM, Ollama, vLLM, etc.)</option>
                      <option value={LabelingMethod.LOCAL}>Local Model</option>
                    </select>
                  </div>

                  {/* OpenAI Model (when OpenAI selected) */}
                  {labelingMethod === LabelingMethod.OPENAI && (
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        OpenAI Model
                      </label>
                      <div className="flex gap-2">
                        {openaiModels.length > 0 ? (
                          <select
                            value={openaiModel}
                            onChange={(e) => setOpenaiModel(e.target.value)}
                            className="flex-1 px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                          >
                            {openaiModels.map((model) => (
                              <option key={model} value={model}>
                                {model}{model === 'gpt-4o-mini' ? ' (recommended)' : ''}
                              </option>
                            ))}
                          </select>
                        ) : (
                          <select
                            value={openaiModel}
                            onChange={(e) => setOpenaiModel(e.target.value)}
                            className="flex-1 px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                          >
                            <option value="gpt-4o-mini">gpt-4o-mini (recommended)</option>
                            <option value="gpt-4o">gpt-4o</option>
                            <option value="gpt-3.5-turbo">gpt-3.5-turbo</option>
                          </select>
                        )}
                        <button
                          type="button"
                          onClick={fetchOpenAIModels}
                          disabled={loadingOpenaiModels}
                          className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                        >
                          {loadingOpenaiModels ? 'Loading...' : 'Fetch Models'}
                        </button>
                      </div>
                      {openaiModelsError && (
                        <p className="mt-1 text-xs text-red-400">{openaiModelsError}</p>
                      )}
                      {!openaiModelsError && openaiModels.length > 0 && (
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                          {openaiModels.length} model(s) available from OpenAI
                        </p>
                      )}
                    </div>
                  )}

                  {/* OpenAI-Compatible Endpoint + Model (when OpenAI-Compatible selected) */}
                  {labelingMethod === LabelingMethod.OPENAI_COMPATIBLE && (
                    <>
                      <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                          Endpoint URL
                        </label>
                        <div className="flex gap-2">
                          <div className="flex-1 relative" ref={savedEndpointsRef}>
                            <div className="flex">
                              <input
                                type="text"
                                value={openaiCompatibleEndpoint}
                                onChange={(e) => {
                                  setOpenaiCompatibleEndpoint(e.target.value);
                                  setCompatibleModels([]);
                                  setCompatibleModelsError(null);
                                }}
                                placeholder="/ollama/v1"
                                className="flex-1 px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-l-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                              />
                              <button
                                type="button"
                                onClick={() => setShowSavedEndpoints(!showSavedEndpoints)}
                                className="px-2 py-2 bg-white dark:bg-slate-900 border border-l-0 border-slate-300 dark:border-slate-700 rounded-r-lg text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition-colors"
                                title="Saved endpoints"
                              >
                                <ChevronDown className="w-4 h-4" />
                              </button>
                            </div>
                            {showSavedEndpoints && (
                              <div className="absolute z-50 top-full left-0 right-0 mt-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg shadow-xl max-h-48 overflow-y-auto">
                                {savedEndpoints.length === 0 ? (
                                  <div className="px-3 py-2 text-xs text-slate-500">No saved endpoints</div>
                                ) : (
                                  savedEndpoints
                                    .sort((a, b) => (b.lastUsed || 0) - (a.lastUsed || 0))
                                    .map((ep) => (
                                      <div
                                        key={ep.url}
                                        className="flex items-center gap-1 px-3 py-1.5 hover:bg-slate-100 dark:hover:bg-slate-700 group"
                                      >
                                        <button
                                          type="button"
                                          className="flex-1 text-left text-sm text-slate-800 dark:text-slate-200 truncate"
                                          onClick={() => {
                                            setOpenaiCompatibleEndpoint(ep.url);
                                            setCompatibleModels([]);
                                            setCompatibleModelsError(null);
                                            touchEndpoint(ep.url);
                                            setShowSavedEndpoints(false);
                                          }}
                                          title={ep.url}
                                        >
                                          {ep.label ? (
                                            <>
                                              <span className="text-slate-700 dark:text-slate-300">{ep.label}</span>
                                              <span className="text-slate-500 ml-1 text-xs">({ep.url})</span>
                                            </>
                                          ) : (
                                            ep.url
                                          )}
                                        </button>
                                        <button
                                          type="button"
                                          onClick={(e) => {
                                            e.stopPropagation();
                                            removeEndpoint(ep.url);
                                          }}
                                          className="p-0.5 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                                          title="Delete saved endpoint"
                                        >
                                          <Trash2 className="w-3.5 h-3.5" />
                                        </button>
                                      </div>
                                    ))
                                )}
                              </div>
                            )}
                          </div>
                          <button
                            type="button"
                            onClick={() => {
                              if (openaiCompatibleEndpoint.trim()) {
                                addEndpoint(openaiCompatibleEndpoint.trim());
                              }
                            }}
                            disabled={!openaiCompatibleEndpoint.trim()}
                            className="px-2 py-2 bg-slate-100 dark:bg-slate-700 text-slate-700 dark:text-slate-300 rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 hover:text-slate-900 dark:hover:text-white disabled:bg-slate-50 dark:disabled:bg-slate-800 disabled:text-slate-400 dark:disabled:text-slate-600 disabled:cursor-not-allowed transition-colors"
                            title="Save endpoint URL"
                          >
                            <Save className="w-4 h-4" />
                          </button>
                          <button
                            type="button"
                            onClick={fetchCompatibleModels}
                            disabled={loadingCompatibleModels || !openaiCompatibleEndpoint}
                            className="px-3 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
                          >
                            {loadingCompatibleModels ? 'Loading...' : 'Fetch Models'}
                          </button>
                        </div>
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                          OpenAI-compatible API endpoint (miLLM, Ollama, vLLM, etc.)
                        </p>
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                          Model Name
                        </label>
                        {compatibleModels.length > 0 ? (
                          <select
                            value={openaiCompatibleModel}
                            onChange={(e) => setOpenaiCompatibleModel(e.target.value)}
                            className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
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
                            className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                          />
                        )}
                        {compatibleModelsError && (
                          <p className="mt-1 text-xs text-red-400">{compatibleModelsError}</p>
                        )}
                        {!compatibleModelsError && (
                          <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
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
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        Local Model
                      </label>
                      {loadingModels ? (
                        <div className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-600 dark:text-slate-400">
                          Loading models...
                        </div>
                      ) : availableModels.length > 0 ? (
                        <select
                          value={localModel}
                          onChange={(e) => setLocalModel(e.target.value)}
                          className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        >
                          {availableModels.map((model) => (
                            <option key={model} value={model}>
                              {model}
                            </option>
                          ))}
                        </select>
                      ) : (
                        <div className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-600 dark:text-slate-400">
                          No local models found
                        </div>
                      )}
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        Locally cached HuggingFace models
                      </p>
                    </div>
                  )}
                </div>

                {/* Token Filtering - pushed to bottom */}
                <div className="mt-auto pt-4">
                  <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-300 dark:border-slate-700 p-4">
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
                      <span className="text-slate-600 dark:text-slate-400">{showFilters ? '▼' : '▶'}</span>
                    </button>

                    {showFilters && (
                      <div className="mt-4 grid grid-cols-2 gap-3">
                        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={filterSpecial}
                            onChange={(e) => setFilterSpecial(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                          />
                          <span>Special tokens</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={filterSingleChar}
                            onChange={(e) => setFilterSingleChar(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                          />
                          <span>Single characters</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={filterPunctuation}
                            onChange={(e) => setFilterPunctuation(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                          />
                          <span>Punctuation</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={filterNumbers}
                            onChange={(e) => setFilterNumbers(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                          />
                          <span>Numbers</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={filterFragments}
                            onChange={(e) => setFilterFragments(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                          />
                          <span>Word fragments</span>
                        </label>
                        <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                          <input
                            type="checkbox"
                            checked={filterStopWords}
                            onChange={(e) => setFilterStopWords(e.target.checked)}
                            className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
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
                  {/* Batch Size */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Batch Size
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={batchSize}
                      onChange={(e) => setBatchSize(Math.max(1, Math.min(100, parseInt(e.target.value) || 10)))}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                      Features to label per batch (1-100, default from Settings)
                    </p>
                  </div>

                  {/* Examples Per Feature */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Examples Per Feature
                    </label>
                    <input
                      type="number"
                      min={10}
                      max={50}
                      value={maxExamples}
                      onChange={(e) => setMaxExamples(parseInt(e.target.value) || 25)}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                      Number of high-activation examples to show the LLM (10-50, default: 25)
                    </p>
                  </div>

                  {/* Max Tokens */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Max Response Tokens
                    </label>
                    <input
                      type="number"
                      min={50}
                      max={8000}
                      value={maxTokens}
                      onChange={(e) => setMaxTokens(parseInt(e.target.value) || 300)}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                      Maximum tokens in the LLM response (50-8000, default: 300). Increase for longer descriptions.
                    </p>
                  </div>

                  {/* OpenAI API Key (when OpenAI selected) */}
                  {labelingMethod === LabelingMethod.OPENAI && (
                    <div>
                      <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                        OpenAI API Key (optional)
                      </label>
                      <input
                        type="password"
                        value={openaiApiKey}
                        onChange={(e) => setOpenaiApiKey(e.target.value)}
                        placeholder="Uses server default if not provided"
                        className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500"
                      />
                      <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                        Leave blank to use the server's configured API key
                      </p>
                    </div>
                  )}

                  {/* API Timeout */}
                  <div>
                    <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                      API Timeout (seconds)
                    </label>
                    <input
                      type="number"
                      min={30}
                      max={600}
                      value={apiTimeout}
                      onChange={(e) => setApiTimeout(parseInt(e.target.value) || 120)}
                      className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                    />
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                      Maximum wait time for API responses (30-600 seconds, default: 120). Increase for larger models.
                    </p>
                  </div>
                </div>

                {/* Save API Requests - pushed to bottom */}
                <div className="mt-auto pt-4">
                  <div className="bg-white dark:bg-slate-900 rounded-lg border border-slate-300 dark:border-slate-700 p-4 space-y-3">
                    <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer">
                      <input
                        type="checkbox"
                        checked={saveRequestsForTesting}
                        onChange={(e) => setSaveRequestsForTesting(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                      />
                      <div className="flex flex-col">
                        <span className="font-medium">Save API requests for testing</span>
                        <span className="text-xs text-slate-600 dark:text-slate-400">Save requests to /tmp/ for debugging with Postman/cURL</span>
                      </div>
                    </label>

                    {/* Export Format - only shown when Save API requests is enabled */}
                    {saveRequestsForTesting && (
                      <div className="ml-6 pt-2 border-t border-slate-200 dark:border-slate-700">
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                          Export Format
                        </label>
                        <select
                          value={exportFormat}
                          onChange={(e) => setExportFormat(e.target.value as 'postman' | 'curl' | 'both')}
                          className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-600 rounded-lg text-slate-900 dark:text-white focus:outline-none focus:ring-2 focus:ring-emerald-500"
                        >
                          <option value="both">Both (Postman + cURL)</option>
                          <option value="postman">Postman Collection Only</option>
                          <option value="curl">cURL Command Only</option>
                        </select>
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                          Choose which format(s) to save for debugging
                        </p>
                      </div>
                    )}

                    {/* Sample Rate Slider - only shown when Save API requests is enabled */}
                    {saveRequestsForTesting && (
                      <div className="ml-6 pt-2 border-t border-slate-200 dark:border-slate-700">
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                          Sample Rate: {saveRequestsSampleRate}%
                        </label>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={5}
                          value={saveRequestsSampleRate}
                          onChange={(e) => setSaveRequestsSampleRate(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-100 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                        />
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
                          Percentage of all requests to save (0% = none, 100% = all). Lower values reduce disk usage.
                        </p>
                      </div>
                    )}

                    {/* Poor Quality Detection - in same debugging card */}
                    <label className="flex items-center gap-2 text-sm text-slate-700 dark:text-slate-300 cursor-pointer pt-3 border-t border-slate-200 dark:border-slate-700">
                      <input
                        type="checkbox"
                        checked={savePoorQualityLabels}
                        onChange={(e) => setSavePoorQualityLabels(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-300 dark:border-slate-600 bg-white dark:bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-white dark:focus:ring-offset-slate-900"
                      />
                      <div className="flex flex-col">
                        <span className="font-medium">Save poor quality labels for debugging</span>
                        <span className="text-xs text-slate-600 dark:text-slate-400">Detect and save ineffective labels like "uncategorized" or generic responses</span>
                      </div>
                    </label>

                    {/* Sample Rate Slider - only shown when Save poor quality labels is enabled */}
                    {savePoorQualityLabels && (
                      <div className="ml-6 pt-2 border-t border-slate-200 dark:border-slate-700">
                        <label className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                          Sample Rate: {poorQualitySampleRate}%
                        </label>
                        <input
                          type="range"
                          min={0}
                          max={100}
                          step={5}
                          value={poorQualitySampleRate}
                          onChange={(e) => setPoorQualitySampleRate(parseInt(e.target.value))}
                          className="w-full h-2 bg-slate-100 dark:bg-slate-700 rounded-lg appearance-none cursor-pointer accent-emerald-600"
                        />
                        <p className="mt-1 text-xs text-slate-600 dark:text-slate-400">
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
                className="px-4 py-2 bg-slate-100 dark:bg-slate-700 text-slate-900 dark:text-white rounded-lg hover:bg-slate-200 dark:hover:bg-slate-600 transition-colors"
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

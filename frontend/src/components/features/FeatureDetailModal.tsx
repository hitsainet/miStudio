/**
 * FeatureDetailModal Component
 *
 * Displays detailed information about a discovered feature with tabs for:
 * - Max-activating examples with token highlighting
 * - Logit lens analysis (predicted tokens)
 * - Feature correlations
 * - Ablation impact analysis
 */

import React, { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { X, Save, Star, Info, Activity, GitBranch, Zap, Hash, Copy, Check, ChevronDown, Brain, Sparkles, Loader2 } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import { useEnhancedLabeling } from '../../hooks/useEnhancedLabeling';
import { ExampleRow } from './ExampleRow';
import { LogitLensView } from './LogitLensView';
import { JSpaceAnnotation } from './JSpaceAnnotation';
import { FeatureCorrelations } from './FeatureCorrelations';
import { AblationAnalysis } from './AblationAnalysis';
import { FeatureTokenAnalysis } from './FeatureTokenAnalysis';
import { NLPAnalysisView } from './NLPAnalysisView';
import { formatActivation } from '../../utils/formatters';
import { formatExamplesForClipboard, copyToClipboard, ExportFormat } from '../../utils/featureExampleFormatter';
import { fireAndForget } from '../../utils/fireAndForget';

interface FeatureDetailModalProps {
  featureId: string;
  trainingId: string | null;
  onClose: () => void;
}

type TabType = 'examples' | 'logit-lens' | 'j-space' | 'token-analysis' | 'nlp-analysis' | 'correlations' | 'ablation';

export const FeatureDetailModal: React.FC<FeatureDetailModalProps> = ({
  featureId,
  trainingId: _trainingId,
  onClose,
}) => {
  const {
    selectedFeature,
    featureExamples,
    isLoadingFeatureDetail,
    isLoadingExamples,
    fetchFeatureDetail,
    fetchFeatureExamples,
    updateFeature,
    setStarColor,
    clearSelectedFeature,
  } = useFeaturesStore();

  const [activeTab, setActiveTab] = useState<TabType>('examples');
  const [isEditing, setIsEditing] = useState(false);
  const [editedName, setEditedName] = useState('');
  const [editedDescription, setEditedDescription] = useState('');
  const [editedNotes, setEditedNotes] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Enhanced labeling
  const { job: enhancedJob, progressPhrase, completedLabel, startError: enhancedStartError, start: startEnhanced } =
    useEnhancedLabeling(featureId);

  const [notesExpanded, setNotesExpanded] = useState(false);

  // Copy examples state
  const [copyCount, setCopyCount] = useState<number | 'all'>(10);
  const [copyFormat, setCopyFormat] = useState<ExportFormat>('text');
  const [copyStatus, setCopyStatus] = useState<'idle' | 'success' | 'error'>('idle');
  const [showCopyDropdown, setShowCopyDropdown] = useState(false);

  // Load feature details on mount
  useEffect(() => {
    fireAndForget(fetchFeatureDetail(featureId));
    fireAndForget(fetchFeatureExamples(featureId, 100));

    return () => {
      clearSelectedFeature();
    };
  }, [featureId]);

  // Update edit fields when feature loads
  useEffect(() => {
    if (selectedFeature) {
      setEditedName(selectedFeature.name);
      setEditedDescription(selectedFeature.description || '');
      setEditedNotes(selectedFeature.notes || '');
    }
  }, [selectedFeature]);

  // Auto-populate edit fields when enhanced labeling completes.
  // Skip if user was already editing to avoid clobbering unsaved changes.
  useEffect(() => {
    if (completedLabel && !isEditing) {
      setEditedName(completedLabel.name);
      setEditedDescription(completedLabel.description);
      setEditedNotes(completedLabel.notes);
      setIsEditing(true);
    }
    // isEditing intentionally omitted — only react when completedLabel changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [completedLabel]);

  // Handle save
  const handleSave = async () => {
    if (!selectedFeature) return;

    setIsSaving(true);
    try {
      await updateFeature(featureId, {
        name: editedName || null,
        description: editedDescription || null,
        notes: editedNotes || null,
      });
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to update feature:', error);
    } finally {
      setIsSaving(false);
    }
  };

  // Handle favorite toggle — aqua is never downgraded by manual starring
  const handleToggleFavorite = async () => {
    if (!selectedFeature) return;
    try {
      if (selectedFeature.is_favorite) {
        await setStarColor(featureId, null);
      } else {
        await setStarColor(featureId, selectedFeature.star_color === 'aqua' ? 'aqua' : 'yellow');
      }
    } catch (error) {
      console.error('Failed to toggle favorite:', error);
    }
  };

  // Handle backdrop click
  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  // Handle copy examples to clipboard
  const handleCopyExamples = async () => {
    if (featureExamples.length === 0) return;

    const formattedText = formatExamplesForClipboard(
      featureExamples,
      copyCount,
      copyFormat,
      {
        featureIndex: selectedFeature?.neuron_index,
        featureName: selectedFeature?.name,
        maxActivation: selectedFeature?.max_activation,
      }
    );

    const success = await copyToClipboard(formattedText);
    setCopyStatus(success ? 'success' : 'error');

    // Reset status after 2 seconds
    setTimeout(() => setCopyStatus('idle'), 2000);
    setShowCopyDropdown(false);
  };

  if (isLoadingFeatureDetail) {
    return (
      <div
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
        onClick={handleBackdropClick}
      >
        <div className="bg-white dark:bg-slate-900 rounded-lg p-8">
          <div className="flex items-center gap-3 text-slate-600 dark:text-slate-400">
            <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
            <span>Loading feature details...</span>
          </div>
        </div>
      </div>
    );
  }

  if (!selectedFeature) return null;

  return (
    <div
      className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4"
      onClick={handleBackdropClick}
    >
      <div className="bg-white dark:bg-slate-900 rounded-lg shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-200 dark:border-slate-800 p-6 flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-bold text-slate-900 dark:text-white">
                Feature #{selectedFeature.neuron_index}
              </h2>
              <button
                onClick={handleToggleFavorite}
                className="p-1 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
              >
                <Star
                  className={`w-5 h-5 ${
                    selectedFeature.star_color === 'aqua'
                      ? 'fill-cyan-400 text-cyan-400'
                      : selectedFeature.star_color === 'purple'
                      ? 'fill-purple-400 text-purple-400'
                      : selectedFeature.is_favorite
                      ? 'fill-yellow-400 text-yellow-400'
                      : 'text-slate-500'
                  }`}
                />
              </button>
            </div>

            {/* Editable Label */}
            {isEditing ? (
              <div className="space-y-3">
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Feature Name</label>
                  <input
                    type="text"
                    value={editedName}
                    onChange={(e) => setEditedName(e.target.value)}
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:border-emerald-500 focus:outline-none"
                    placeholder="Enter feature name..."
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Description</label>
                  <textarea
                    value={editedDescription}
                    onChange={(e) => setEditedDescription(e.target.value)}
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:border-emerald-500 focus:outline-none resize-none"
                    rows={2}
                    placeholder="Enter description..."
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-600 dark:text-slate-400 mb-1">Notes</label>
                  <textarea
                    value={editedNotes}
                    onChange={(e) => setEditedNotes(e.target.value)}
                    className="w-full px-3 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-slate-900 dark:text-white focus:border-emerald-500 focus:outline-none resize-none"
                    rows={3}
                    placeholder="Add notes..."
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-lg text-slate-900 dark:text-white">{selectedFeature.name}</p>
                {selectedFeature.description && (
                  <p className="text-sm text-slate-600 dark:text-slate-400">{selectedFeature.description}</p>
                )}
                {selectedFeature.notes && (
                  <div className="mt-2 bg-slate-100 dark:bg-slate-800/50 rounded">
                    <button
                      onClick={() => setNotesExpanded((v) => !v)}
                      className="flex items-center justify-between w-full px-3 py-2 text-xs text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
                    >
                      <span>Notes</span>
                      <ChevronDown className={`w-3 h-3 transition-transform ${notesExpanded ? 'rotate-180' : ''}`} />
                    </button>
                    {notesExpanded && (
                      <div className="px-3 pb-3 max-h-96 overflow-y-auto text-sm text-slate-700 dark:text-slate-300">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          components={{
                            // Compact, dark-theme-aware markdown rendering
                            p: ({ children }) => <p className="mb-3 leading-relaxed">{children}</p>,
                            h1: ({ children }) => <h3 className="text-base font-semibold text-slate-800 dark:text-slate-200 mt-3 mb-2">{children}</h3>,
                            h2: ({ children }) => <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200 mt-3 mb-1">{children}</h4>,
                            h3: ({ children }) => <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-200 mt-2 mb-1">{children}</h4>,
                            strong: ({ children }) => <strong className="font-semibold text-slate-900 dark:text-slate-100">{children}</strong>,
                            ul: ({ children }) => <ul className="list-disc list-inside mb-3 space-y-1">{children}</ul>,
                            ol: ({ children }) => <ol className="list-decimal list-inside mb-3 space-y-1">{children}</ol>,
                            li: ({ children }) => <li className="text-slate-700 dark:text-slate-300">{children}</li>,
                            code: ({ children }) => <code className="px-1 py-0.5 rounded bg-white dark:bg-slate-900 text-slate-800 dark:text-slate-200 font-mono text-xs">{children}</code>,
                            pre: ({ children }) => <pre className="p-2 my-2 rounded bg-white dark:bg-slate-900 overflow-x-auto text-xs">{children}</pre>,
                            hr: () => <hr className="my-3 border-slate-300 dark:border-slate-700" />,
                            table: ({ children }) => (
                              <div className="my-3 overflow-x-auto">
                                <table className="text-xs border border-slate-300 dark:border-slate-700 border-collapse">{children}</table>
                              </div>
                            ),
                            thead: ({ children }) => <thead className="bg-white dark:bg-slate-900">{children}</thead>,
                            tbody: ({ children }) => <tbody>{children}</tbody>,
                            tr: ({ children }) => <tr className="border-b border-slate-200 dark:border-slate-800">{children}</tr>,
                            th: ({ children }) => <th className="px-2 py-1 text-left font-semibold text-slate-800 dark:text-slate-200 border-r border-slate-200 dark:border-slate-800 last:border-r-0">{children}</th>,
                            td: ({ children }) => <td className="px-2 py-1 text-slate-700 dark:text-slate-300 border-r border-slate-200 dark:border-slate-800 last:border-r-0 align-top">{children}</td>,
                          }}
                        >
                          {selectedFeature.notes}
                        </ReactMarkdown>
                      </div>
                    )}
                  </div>
                )}
                <p className="text-xs text-slate-500">
                  Label source: {
                    selectedFeature.label_source === 'auto' ? 'Automatic'
                    : selectedFeature.label_source === 'enhanced_llm' ? 'Enhanced LLM'
                    : selectedFeature.label_source === 'llm' || selectedFeature.label_source === 'local_llm' || selectedFeature.label_source === 'openai' ? 'LLM'
                    : 'User'
                  }
                </p>
              </div>
            )}

            {/* Edit/Save buttons */}
            <div className="flex gap-2 mt-4">
              {isEditing ? (
                <>
                  <button
                    onClick={handleSave}
                    disabled={isSaving}
                    className="flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded transition-colors disabled:opacity-50"
                  >
                    <Save className="w-4 h-4" />
                    {isSaving ? 'Saving...' : 'Save Changes'}
                  </button>
                  <button
                    onClick={() => {
                      setIsEditing(false);
                      setEditedName(selectedFeature.name);
                      setEditedDescription(selectedFeature.description || '');
                      setEditedNotes(selectedFeature.notes || '');
                    }}
                    className="px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-900 dark:text-white rounded transition-colors"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={() => setIsEditing(true)}
                    className="px-4 py-2 bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-900 dark:text-white rounded transition-colors"
                  >
                    Edit Details
                  </button>

                  {/* Enhanced Label button / in-progress indicator */}
                  {enhancedJob?.status === 'running' || enhancedJob?.status === 'queued' ? (
                    <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-400 px-2">
                      <Loader2 className="w-4 h-4 animate-spin text-violet-400" />
                      <span className="text-violet-300">{progressPhrase ?? 'Queued...'}</span>
                    </div>
                  ) : (
                    <button
                      onClick={startEnhanced}
                      title="Run two-pass LLM labeling on this feature's activation examples"
                      className="flex items-center gap-2 px-4 py-2 bg-violet-700 hover:bg-violet-600 text-white rounded transition-colors"
                    >
                      <Sparkles className="w-4 h-4" />
                      Enhanced Label
                    </button>
                  )}
                  {enhancedJob?.status === 'failed' && (
                    <p className="text-xs text-red-400 self-center ml-1">
                      {enhancedJob.error_message ?? 'Enhanced labeling failed'}
                    </p>
                  )}
                  {enhancedStartError && (
                    <p className="text-xs text-red-400 self-center ml-1">
                      {enhancedStartError}
                    </p>
                  )}
                </>
              )}
            </div>
          </div>

          {/* Close button */}
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-100 dark:hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-6 h-6 text-slate-600 dark:text-slate-400" />
          </button>
        </div>

        {/* Statistics Grid */}
        <div className="border-b border-slate-200 dark:border-slate-800 p-6">
          <div className="grid grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Activation Frequency</p>
              <p className="text-xl font-semibold text-emerald-400">
                {(selectedFeature.activation_frequency * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Interpretability</p>
              <p className="text-xl font-semibold text-blue-400">
                {(selectedFeature.interpretability_score * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Max Activation</p>
              <p className="text-xl font-semibold text-purple-400">
                {formatActivation(selectedFeature.max_activation)}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-600 dark:text-slate-400 mb-1">Active Samples</p>
              <p className="text-xl font-semibold text-orange-400">
                {selectedFeature.active_samples}
              </p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-slate-200 dark:border-slate-800 px-6">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('examples')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'examples'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <Activity className="w-4 h-4" />
              <span>Examples</span>
            </button>
            <button
              onClick={() => setActiveTab('logit-lens')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'logit-lens'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <Info className="w-4 h-4" />
              <span>Logit Lens</span>
            </button>
            <button
              onClick={() => setActiveTab('j-space')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'j-space'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <Info className="w-4 h-4" />
              <span>J-Space</span>
            </button>
            <button
              onClick={() => setActiveTab('token-analysis')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'token-analysis'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <Hash className="w-4 h-4" />
              <span>Token Analysis</span>
            </button>
            <button
              onClick={() => setActiveTab('nlp-analysis')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'nlp-analysis'
                  ? 'border-cyan-500 text-cyan-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <Brain className="w-4 h-4" />
              <span>NLP Analysis</span>
            </button>
            <button
              onClick={() => setActiveTab('correlations')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'correlations'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <GitBranch className="w-4 h-4" />
              <span>Correlations</span>
            </button>
            <button
              onClick={() => setActiveTab('ablation')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'ablation'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300'
              }`}
            >
              <Zap className="w-4 h-4" />
              <span>Ablation</span>
            </button>
          </div>
        </div>

        {/* Tab Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'examples' && (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-slate-900 dark:text-white">Max-Activating Examples</h3>
                <div className="flex items-center gap-3">
                  <span className="text-sm text-slate-600 dark:text-slate-400">
                    Showing {featureExamples.length} examples
                  </span>

                  {/* Copy Examples Controls */}
                  {featureExamples.length > 0 && (
                    <div className="relative flex items-center gap-2">
                      {/* Count Selector */}
                      <select
                        value={copyCount === 'all' ? 'all' : copyCount.toString()}
                        onChange={(e) => setCopyCount(e.target.value === 'all' ? 'all' : parseInt(e.target.value))}
                        className="px-2 py-1.5 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-sm text-slate-700 dark:text-slate-300 focus:border-emerald-500 focus:outline-none cursor-pointer"
                      >
                        <option value="5">5</option>
                        <option value="10">10</option>
                        <option value="25">25</option>
                        <option value="50">50</option>
                        <option value="100">100</option>
                        <option value="all">All ({featureExamples.length})</option>
                      </select>

                      {/* Format Selector */}
                      <div className="relative">
                        <button
                          onClick={() => setShowCopyDropdown(!showCopyDropdown)}
                          className="flex items-center gap-1 px-2 py-1.5 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded text-sm text-slate-700 dark:text-slate-300 hover:border-slate-300 dark:hover:border-slate-600 focus:border-emerald-500 focus:outline-none"
                        >
                          {copyFormat === 'text' ? 'Text' : copyFormat === 'markdown' ? 'Markdown' : 'JSON'}
                          <ChevronDown className="w-3 h-3" />
                        </button>
                        {showCopyDropdown && (
                          <div className="absolute right-0 top-full mt-1 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded shadow-lg z-10">
                            <button
                              onClick={() => { setCopyFormat('text'); setShowCopyDropdown(false); }}
                              className={`block w-full px-3 py-1.5 text-sm text-left hover:bg-slate-100 dark:hover:bg-slate-700 ${copyFormat === 'text' ? 'text-emerald-400' : 'text-slate-700 dark:text-slate-300'}`}
                            >
                              Text
                            </button>
                            <button
                              onClick={() => { setCopyFormat('markdown'); setShowCopyDropdown(false); }}
                              className={`block w-full px-3 py-1.5 text-sm text-left hover:bg-slate-100 dark:hover:bg-slate-700 ${copyFormat === 'markdown' ? 'text-emerald-400' : 'text-slate-700 dark:text-slate-300'}`}
                            >
                              Markdown
                            </button>
                            <button
                              onClick={() => { setCopyFormat('json'); setShowCopyDropdown(false); }}
                              className={`block w-full px-3 py-1.5 text-sm text-left hover:bg-slate-100 dark:hover:bg-slate-700 ${copyFormat === 'json' ? 'text-emerald-400' : 'text-slate-700 dark:text-slate-300'}`}
                            >
                              JSON
                            </button>
                          </div>
                        )}
                      </div>

                      {/* Copy Button */}
                      <button
                        onClick={handleCopyExamples}
                        disabled={copyStatus === 'success'}
                        className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-sm transition-colors ${
                          copyStatus === 'success'
                            ? 'bg-emerald-600 text-white'
                            : copyStatus === 'error'
                            ? 'bg-red-600 text-white'
                            : 'bg-slate-100 dark:bg-slate-700 hover:bg-slate-200 dark:hover:bg-slate-600 text-slate-700 dark:text-slate-300'
                        }`}
                      >
                        {copyStatus === 'success' ? (
                          <>
                            <Check className="w-4 h-4" />
                            Copied!
                          </>
                        ) : copyStatus === 'error' ? (
                          <>
                            <X className="w-4 h-4" />
                            Failed
                          </>
                        ) : (
                          <>
                            <Copy className="w-4 h-4" />
                            Copy
                          </>
                        )}
                      </button>
                    </div>
                  )}
                </div>
              </div>

              {isLoadingExamples ? (
                <div className="flex items-center justify-center py-12">
                  <div className="flex items-center gap-3 text-slate-600 dark:text-slate-400">
                    <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                    <span>Loading examples...</span>
                  </div>
                </div>
              ) : featureExamples.length === 0 ? (
                <div className="text-center py-12 text-slate-600 dark:text-slate-400">
                  No examples available for this feature.
                </div>
              ) : (
                <div className="bg-white dark:bg-slate-800 rounded-lg divide-y divide-slate-200 dark:divide-slate-700/50">
                  {featureExamples.map((example, index) => (
                    <ExampleRow
                      key={index}
                      example={example}
                      index={index}
                      maxActivation={selectedFeature?.max_activation}
                    />
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'logit-lens' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Logit Lens Analysis</h3>
              <LogitLensView featureId={featureId} />
            </div>
          )}

          {activeTab === 'j-space' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">
                J-Space Annotation
              </h3>
              <JSpaceAnnotation
                featureId={featureId}
                labelTokens={
                  selectedFeature?.name ? selectedFeature.name.split(/\s+/) : []
                }
              />
            </div>
          )}

          {activeTab === 'token-analysis' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Token Analysis</h3>
              <FeatureTokenAnalysis featureId={featureId} />
            </div>
          )}

          {activeTab === 'nlp-analysis' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">NLP Analysis</h3>
              <NLPAnalysisView
                nlpAnalysis={selectedFeature?.nlp_analysis || null}
                nlpProcessedAt={selectedFeature?.nlp_processed_at || null}
                featureId={featureId}
              />
            </div>
          )}

          {activeTab === 'correlations' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Feature Correlations</h3>
              <FeatureCorrelations featureId={featureId} />
            </div>
          )}

          {activeTab === 'ablation' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-slate-900 dark:text-white mb-4">Ablation Analysis</h3>
              <AblationAnalysis featureId={featureId} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

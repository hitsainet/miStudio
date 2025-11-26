/**
 * SteeringPanel - Main panel for feature steering operations.
 *
 * Layout:
 * - Left sidebar: Feature selection (SAE picker, selected features, browser)
 * - Main area: Prompt input, generation config, results
 *
 * Features:
 * - SAE selection with feature browser
 * - Up to 4 features with individual strength controls
 * - Prompt input with generation button
 * - Side-by-side comparison of unsteered vs steered outputs
 * - Save experiments for later reference
 */

import { useEffect, useState } from 'react';
import { Play, Loader, AlertCircle, ChevronLeft, ChevronRight, Brain } from 'lucide-react';
import { useSteeringStore, selectCanGenerate } from '../../stores/steeringStore';
import { useSAEsStore } from '../../stores/saesStore';
import { SAEStatus } from '../../types/sae';
import { FeatureSelector } from '../steering/FeatureSelector';
import { GenerationConfig } from '../steering/GenerationConfig';
import { ComparisonResults } from '../steering/ComparisonResults';
import { COMPONENTS } from '../../config/brand';

export function SteeringPanel() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [showSaveModal, setShowSaveModal] = useState(false);

  const {
    selectedSAE,
    selectedFeatures,
    prompt,
    isGenerating,
    progress,
    progressMessage,
    currentComparison,
    error,
    setPrompt,
    generateComparison,
    clearError,
  } = useSteeringStore();

  const { saes, fetchSAEs } = useSAEsStore();

  const canGenerate = selectCanGenerate(useSteeringStore.getState());

  // Load SAEs on mount
  useEffect(() => {
    fetchSAEs();
  }, [fetchSAEs]);

  const handleGenerate = async () => {
    try {
      await generateComparison(true, true);
    } catch (error) {
      console.error('[SteeringPanel] Generation failed:', error);
    }
  };

  const handleSaveExperiment = () => {
    setShowSaveModal(true);
  };

  return (
    <div className="h-full flex">
      {/* Sidebar */}
      <div
        className={`transition-all duration-300 ${
          sidebarCollapsed ? 'w-0' : 'w-80'
        } flex-shrink-0 overflow-hidden`}
      >
        <FeatureSelector />
      </div>

      {/* Sidebar toggle */}
      <button
        onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
        className="flex-shrink-0 w-6 bg-slate-900 border-r border-slate-800 hover:bg-slate-800 flex items-center justify-center"
        title={sidebarCollapsed ? 'Show sidebar' : 'Hide sidebar'}
      >
        {sidebarCollapsed ? (
          <ChevronRight className="w-4 h-4 text-slate-500" />
        ) : (
          <ChevronLeft className="w-4 h-4 text-slate-500" />
        )}
      </button>

      {/* Main content */}
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-4xl mx-auto px-6 py-6 space-y-6">
          {/* Header */}
          <div>
            <h1 className="text-xl font-semibold text-slate-100 mb-2">Feature Steering</h1>
            <p className="text-slate-400">
              Steer model outputs by adjusting feature activations during generation
            </p>
          </div>

          {/* Error message */}
          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5" />
                <span>{error}</span>
              </div>
              <button onClick={clearError} className="text-red-400 hover:text-red-300">
                Dismiss
              </button>
            </div>
          )}

          {/* No SAE selected state */}
          {!selectedSAE && (
            <div className={`${COMPONENTS.card.base} p-8 text-center`}>
              <Brain className="w-12 h-12 text-slate-600 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-slate-300 mb-2">
                Select an SAE to begin
              </h3>
              <p className="text-slate-500 max-w-md mx-auto">
                Choose a Sparse Autoencoder from the sidebar to browse its features and start steering.
                {saes.filter((s) => s.status === SAEStatus.READY).length === 0 && (
                  <span className="block mt-2 text-amber-400">
                    No SAEs ready. Go to the SAEs tab to download one.
                  </span>
                )}
              </p>
            </div>
          )}

          {/* Main steering interface */}
          {selectedSAE && (
            <>
              {/* Prompt input */}
              <div className={`${COMPONENTS.card.base} p-4`}>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Prompt
                </label>
                <textarea
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  placeholder="Enter your prompt here..."
                  rows={4}
                  className="w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 resize-none transition-colors"
                />
                <div className="flex items-center justify-between mt-3">
                  <div className="text-sm text-slate-500">
                    {selectedFeatures.length === 0 ? (
                      <span className="text-amber-400">
                        Select at least one feature from the sidebar
                      </span>
                    ) : (
                      <span>
                        {selectedFeatures.length} feature{selectedFeatures.length !== 1 ? 's' : ''} selected
                      </span>
                    )}
                  </div>
                  <button
                    onClick={handleGenerate}
                    disabled={!canGenerate || isGenerating}
                    className={`px-6 py-2 flex items-center gap-2 ${COMPONENTS.button.primary} disabled:opacity-50 disabled:cursor-not-allowed`}
                  >
                    {isGenerating ? (
                      <>
                        <Loader className="w-4 h-4 animate-spin" />
                        Generating...
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4" />
                        Generate Comparison
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Generation config */}
              <GenerationConfig />

              {/* Progress indicator */}
              {isGenerating && (
                <div className={`${COMPONENTS.card.base} p-4`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm text-slate-400">
                      {progressMessage || 'Generating...'}
                    </span>
                    <span className="text-sm text-emerald-400 font-mono">
                      {progress.toFixed(0)}%
                    </span>
                  </div>
                  <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Results */}
              {currentComparison && !isGenerating && (
                <ComparisonResults
                  comparison={currentComparison}
                  onSaveExperiment={handleSaveExperiment}
                />
              )}

              {/* Empty results state */}
              {!currentComparison && !isGenerating && selectedFeatures.length > 0 && prompt.trim() && (
                <div className={`${COMPONENTS.card.base} p-8 text-center`}>
                  <Play className="w-12 h-12 text-slate-600 mx-auto mb-4" />
                  <h3 className="text-lg font-medium text-slate-300 mb-2">
                    Ready to generate
                  </h3>
                  <p className="text-slate-500">
                    Click "Generate Comparison" to see how your selected features affect the output.
                  </p>
                </div>
              )}
            </>
          )}
        </div>
      </div>

      {/* Save Experiment Modal */}
      {showSaveModal && (
        <SaveExperimentModal
          onClose={() => setShowSaveModal(false)}
          onSaved={() => setShowSaveModal(false)}
        />
      )}
    </div>
  );
}

// Save Experiment Modal Component
interface SaveExperimentModalProps {
  onClose: () => void;
  onSaved: () => void;
}

function SaveExperimentModal({ onClose, onSaved }: SaveExperimentModalProps) {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { saveExperiment } = useSteeringStore();

  const handleSave = async () => {
    if (!name.trim()) {
      setError('Name is required');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const tagList = tags
        .split(',')
        .map((t) => t.trim())
        .filter((t) => t);
      await saveExperiment(name.trim(), description.trim() || undefined, tagList.length > 0 ? tagList : undefined);
      onSaved();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to save experiment');
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className={`${COMPONENTS.card.base} w-full max-w-md p-6 m-4`}>
        <h3 className="text-lg font-semibold text-slate-100 mb-4">Save Experiment</h3>

        {error && (
          <div className="mb-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
            {error}
          </div>
        )}

        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Name <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="My steering experiment"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Description
            </label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Optional description..."
              rows={3}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 resize-none"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-slate-300 mb-1">
              Tags <span className="text-slate-500">(comma separated)</span>
            </label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="humor, creative, testing"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500"
            />
          </div>
        </div>

        <div className="flex justify-end gap-3 mt-6">
          <button
            onClick={onClose}
            className={`px-4 py-2 ${COMPONENTS.button.ghost}`}
            disabled={saving}
          >
            Cancel
          </button>
          <button
            onClick={handleSave}
            className={`px-4 py-2 ${COMPONENTS.button.primary}`}
            disabled={saving || !name.trim()}
          >
            {saving ? 'Saving...' : 'Save Experiment'}
          </button>
        </div>
      </div>
    </div>
  );
}

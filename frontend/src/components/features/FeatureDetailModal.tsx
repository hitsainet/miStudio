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
import { X, Save, Star, Info, Activity, GitBranch, Zap } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';
import { FeatureDetail } from '../../types/features';
import { TokenHighlight } from './TokenHighlight';

interface FeatureDetailModalProps {
  featureId: string;
  trainingId: string;
  onClose: () => void;
}

type TabType = 'examples' | 'logit-lens' | 'correlations' | 'ablation';

export const FeatureDetailModal: React.FC<FeatureDetailModalProps> = ({
  featureId,
  trainingId,
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
    toggleFavorite,
    clearSelectedFeature,
  } = useFeaturesStore();

  const [activeTab, setActiveTab] = useState<TabType>('examples');
  const [isEditing, setIsEditing] = useState(false);
  const [editedName, setEditedName] = useState('');
  const [editedDescription, setEditedDescription] = useState('');
  const [editedNotes, setEditedNotes] = useState('');
  const [isSaving, setIsSaving] = useState(false);

  // Load feature details on mount
  useEffect(() => {
    fetchFeatureDetail(featureId);
    fetchFeatureExamples(featureId, 100);

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

  // Handle favorite toggle
  const handleToggleFavorite = async () => {
    if (!selectedFeature) return;
    try {
      await toggleFavorite(featureId, !selectedFeature.is_favorite);
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

  if (isLoadingFeatureDetail) {
    return (
      <div
        className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center"
        onClick={handleBackdropClick}
      >
        <div className="bg-slate-900 rounded-lg p-8">
          <div className="flex items-center gap-3 text-slate-400">
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
      <div className="bg-slate-900 rounded-lg shadow-2xl w-full max-w-6xl max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="border-b border-slate-800 p-6 flex items-start justify-between">
          <div className="flex-1">
            <div className="flex items-center gap-3 mb-2">
              <h2 className="text-2xl font-bold text-white">
                Feature #{selectedFeature.neuron_index}
              </h2>
              <button
                onClick={handleToggleFavorite}
                className="p-1 hover:bg-slate-800 rounded transition-colors"
              >
                <Star
                  className={`w-5 h-5 ${
                    selectedFeature.is_favorite
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
                  <label className="block text-xs text-slate-400 mb-1">Feature Name</label>
                  <input
                    type="text"
                    value={editedName}
                    onChange={(e) => setEditedName(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:border-emerald-500 focus:outline-none"
                    placeholder="Enter feature name..."
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Description</label>
                  <textarea
                    value={editedDescription}
                    onChange={(e) => setEditedDescription(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:border-emerald-500 focus:outline-none resize-none"
                    rows={2}
                    placeholder="Enter description..."
                  />
                </div>
                <div>
                  <label className="block text-xs text-slate-400 mb-1">Notes</label>
                  <textarea
                    value={editedNotes}
                    onChange={(e) => setEditedNotes(e.target.value)}
                    className="w-full px-3 py-2 bg-slate-800 border border-slate-700 rounded text-white focus:border-emerald-500 focus:outline-none resize-none"
                    rows={3}
                    placeholder="Add notes..."
                  />
                </div>
              </div>
            ) : (
              <div className="space-y-2">
                <p className="text-lg text-white">{selectedFeature.name}</p>
                {selectedFeature.description && (
                  <p className="text-sm text-slate-400">{selectedFeature.description}</p>
                )}
                {selectedFeature.notes && (
                  <div className="mt-2 p-3 bg-slate-800/50 rounded">
                    <p className="text-xs text-slate-400 mb-1">Notes:</p>
                    <p className="text-sm text-slate-300">{selectedFeature.notes}</p>
                  </div>
                )}
                <p className="text-xs text-slate-500">
                  Label source: {selectedFeature.label_source === 'auto' ? 'Automatic' : 'User'}
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
                    className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                  >
                    Cancel
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setIsEditing(true)}
                  className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-white rounded transition-colors"
                >
                  Edit Details
                </button>
              )}
            </div>
          </div>

          {/* Close button */}
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-800 rounded transition-colors"
          >
            <X className="w-6 h-6 text-slate-400" />
          </button>
        </div>

        {/* Statistics Grid */}
        <div className="border-b border-slate-800 p-6">
          <div className="grid grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-slate-400 mb-1">Activation Frequency</p>
              <p className="text-xl font-semibold text-emerald-400">
                {(selectedFeature.activation_frequency * 100).toFixed(2)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-400 mb-1">Interpretability</p>
              <p className="text-xl font-semibold text-blue-400">
                {(selectedFeature.interpretability_score * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-400 mb-1">Max Activation</p>
              <p className="text-xl font-semibold text-purple-400">
                {selectedFeature.max_activation.toFixed(2)}
              </p>
            </div>
            <div>
              <p className="text-xs text-slate-400 mb-1">Active Samples</p>
              <p className="text-xl font-semibold text-orange-400">
                {selectedFeature.active_samples}
              </p>
            </div>
          </div>
        </div>

        {/* Tabs */}
        <div className="border-b border-slate-800 px-6">
          <div className="flex gap-1">
            <button
              onClick={() => setActiveTab('examples')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'examples'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-400 hover:text-slate-300'
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
                  : 'border-transparent text-slate-400 hover:text-slate-300'
              }`}
            >
              <Info className="w-4 h-4" />
              <span>Logit Lens</span>
            </button>
            <button
              onClick={() => setActiveTab('correlations')}
              className={`flex items-center gap-2 px-4 py-3 border-b-2 transition-colors ${
                activeTab === 'correlations'
                  ? 'border-emerald-500 text-emerald-400'
                  : 'border-transparent text-slate-400 hover:text-slate-300'
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
                  : 'border-transparent text-slate-400 hover:text-slate-300'
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
                <h3 className="text-lg font-semibold text-white">Max-Activating Examples</h3>
                <span className="text-sm text-slate-400">
                  Showing {featureExamples.length} examples
                </span>
              </div>

              {isLoadingExamples ? (
                <div className="flex items-center justify-center py-12">
                  <div className="flex items-center gap-3 text-slate-400">
                    <div className="w-5 h-5 border-2 border-emerald-500 border-t-transparent rounded-full animate-spin" />
                    <span>Loading examples...</span>
                  </div>
                </div>
              ) : featureExamples.length === 0 ? (
                <div className="text-center py-12 text-slate-400">
                  No examples available for this feature.
                </div>
              ) : (
                <div className="space-y-4">
                  {featureExamples.map((example, index) => (
                    <div key={index} className="bg-slate-800 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs text-slate-400">
                          Sample #{example.sample_index}
                        </span>
                        <span className="text-xs font-mono text-emerald-400">
                          Max: {example.max_activation.toFixed(3)}
                        </span>
                      </div>
                      <TokenHighlight
                        tokens={example.tokens}
                        activations={example.activations}
                        maxActivation={example.max_activation}
                      />
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'logit-lens' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white mb-4">Logit Lens Analysis</h3>
              <div className="bg-blue-900/20 border border-blue-700/30 rounded-lg p-6 text-center">
                <Info className="w-12 h-12 text-blue-400 mx-auto mb-3" />
                <p className="text-blue-300 mb-2">Logit lens analysis coming soon</p>
                <p className="text-sm text-slate-400">
                  This will show the most likely tokens predicted when this feature activates.
                </p>
              </div>
            </div>
          )}

          {activeTab === 'correlations' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white mb-4">Feature Correlations</h3>
              <div className="bg-purple-900/20 border border-purple-700/30 rounded-lg p-6 text-center">
                <GitBranch className="w-12 h-12 text-purple-400 mx-auto mb-3" />
                <p className="text-purple-300 mb-2">Correlation analysis coming soon</p>
                <p className="text-sm text-slate-400">
                  This will show other features that activate in similar contexts.
                </p>
              </div>
            </div>
          )}

          {activeTab === 'ablation' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold text-white mb-4">Ablation Analysis</h3>
              <div className="bg-orange-900/20 border border-orange-700/30 rounded-lg p-6 text-center">
                <Zap className="w-12 h-12 text-orange-400 mx-auto mb-3" />
                <p className="text-orange-300 mb-2">Ablation analysis coming soon</p>
                <p className="text-sm text-slate-400">
                  This will show the impact of removing this feature on model performance.
                </p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

/**
 * TrainingTemplateForm component for creating and editing training templates.
 *
 * This component renders a comprehensive form with all template fields including
 * complex hyperparameters configuration.
 */

import React, { useState, useEffect } from 'react';
import { Save, X, ChevronDown, ChevronUp } from 'lucide-react';
import {
  TrainingTemplate,
  TrainingTemplateCreate,
  TrainingTemplateUpdate,
} from '../../types/trainingTemplate';
import { SAEArchitectureType } from '../../types/training';

interface TrainingTemplateFormProps {
  template?: TrainingTemplate;
  onSubmit: (data: TrainingTemplateCreate | TrainingTemplateUpdate) => Promise<void>;
  onCancel?: () => void;
  className?: string;
}

export function TrainingTemplateForm({
  template,
  onSubmit,
  onCancel,
  className = '',
}: TrainingTemplateFormProps) {
  const isEditMode = !!template;

  // Form state - Basic Info
  const [name, setName] = useState(template?.name || '');
  const [description, setDescription] = useState(template?.description || '');
  const [modelId, setModelId] = useState(template?.model_id || '');
  const [datasetId, setDatasetId] = useState(template?.dataset_id || '');
  const [encoderType, setEncoderType] = useState<SAEArchitectureType>(
    (template?.encoder_type as SAEArchitectureType) || SAEArchitectureType.STANDARD
  );

  // Form state - Hyperparameters
  const [hiddenDim, setHiddenDim] = useState(template?.hyperparameters.hidden_dim || 768);
  const [latentDim, setLatentDim] = useState(template?.hyperparameters.latent_dim || 16384);
  const [l1Alpha, setL1Alpha] = useState(template?.hyperparameters.l1_alpha || 0.001);
  const [targetL0, setTargetL0] = useState<string>(
    template?.hyperparameters.target_l0?.toString() || ''
  );
  const [learningRate, setLearningRate] = useState(template?.hyperparameters.learning_rate || 0.0003);
  const [batchSize, setBatchSize] = useState(template?.hyperparameters.batch_size || 4096);
  const [totalSteps, setTotalSteps] = useState(template?.hyperparameters.total_steps || 100000);
  const [warmupSteps, setWarmupSteps] = useState(template?.hyperparameters.warmup_steps || 1000);
  const [weightDecay, setWeightDecay] = useState(template?.hyperparameters.weight_decay || 0.0);
  const [gradClipNorm, setGradClipNorm] = useState<string>(
    template?.hyperparameters.grad_clip_norm?.toString() || ''
  );
  const [checkpointInterval, setCheckpointInterval] = useState(
    template?.hyperparameters.checkpoint_interval || 5000
  );
  const [logInterval, setLogInterval] = useState(template?.hyperparameters.log_interval || 100);
  const [deadNeuronThreshold, setDeadNeuronThreshold] = useState(
    template?.hyperparameters.dead_neuron_threshold || 1000
  );
  const [resampleDeadNeurons, setResampleDeadNeurons] = useState(
    template?.hyperparameters.resample_dead_neurons ?? true
  );

  const [isFavorite, setIsFavorite] = useState(template?.is_favorite || false);
  const [metadataJson, setMetadataJson] = useState(
    template?.extra_metadata ? JSON.stringify(template.extra_metadata, null, 2) : '{}'
  );

  const [showAdvanced, setShowAdvanced] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Reset form when template changes
  useEffect(() => {
    if (template) {
      setName(template.name);
      setDescription(template.description || '');
      setModelId(template.model_id || '');
      setDatasetId(template.dataset_id || '');
      setEncoderType((template.encoder_type as SAEArchitectureType) || SAEArchitectureType.STANDARD);
      setHiddenDim(template.hyperparameters.hidden_dim);
      setLatentDim(template.hyperparameters.latent_dim);
      setL1Alpha(template.hyperparameters.l1_alpha);
      setTargetL0(template.hyperparameters.target_l0?.toString() || '');
      setLearningRate(template.hyperparameters.learning_rate);
      setBatchSize(template.hyperparameters.batch_size);
      setTotalSteps(template.hyperparameters.total_steps);
      setWarmupSteps(template.hyperparameters.warmup_steps || 1000);
      setWeightDecay(template.hyperparameters.weight_decay || 0.0);
      setGradClipNorm(template.hyperparameters.grad_clip_norm?.toString() || '');
      setCheckpointInterval(template.hyperparameters.checkpoint_interval || 5000);
      setLogInterval(template.hyperparameters.log_interval || 100);
      setDeadNeuronThreshold(template.hyperparameters.dead_neuron_threshold || 1000);
      setResampleDeadNeurons(template.hyperparameters.resample_dead_neurons ?? true);
      setIsFavorite(template.is_favorite);
      setMetadataJson(
        template.extra_metadata ? JSON.stringify(template.extra_metadata, null, 2) : '{}'
      );
    }
  }, [template]);

  const validateMetadata = (json: string): Record<string, any> | null => {
    try {
      if (!json.trim()) return {};
      return JSON.parse(json);
    } catch {
      return null;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    // Validation
    if (!name.trim()) {
      setError('Template name is required');
      return;
    }

    if (hiddenDim < 1 || hiddenDim > 100000) {
      setError('Hidden dimension must be between 1 and 100,000');
      return;
    }

    if (latentDim < 1 || latentDim > 1000000) {
      setError('Latent dimension must be between 1 and 1,000,000');
      return;
    }

    if (l1Alpha <= 0) {
      setError('L1 alpha must be greater than 0');
      return;
    }

    if (targetL0 && (parseFloat(targetL0) <= 0 || parseFloat(targetL0) > 1)) {
      setError('Target L0 must be between 0 and 1');
      return;
    }

    if (learningRate <= 0) {
      setError('Learning rate must be greater than 0');
      return;
    }

    if (batchSize < 1) {
      setError('Batch size must be at least 1');
      return;
    }

    if (totalSteps < 1) {
      setError('Total steps must be at least 1');
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
        model_id: modelId.trim() || null,
        dataset_id: datasetId.trim() || null,
        encoder_type: encoderType,
        hyperparameters: {
          hidden_dim: hiddenDim,
          latent_dim: latentDim,
          architecture_type: encoderType,
          l1_alpha: l1Alpha,
          target_l0: targetL0 ? parseFloat(targetL0) : undefined,
          learning_rate: learningRate,
          batch_size: batchSize,
          total_steps: totalSteps,
          warmup_steps: warmupSteps,
          weight_decay: weightDecay,
          grad_clip_norm: gradClipNorm ? parseFloat(gradClipNorm) : undefined,
          checkpoint_interval: checkpointInterval,
          log_interval: logInterval,
          dead_neuron_threshold: deadNeuronThreshold,
          resample_dead_neurons: resampleDeadNeurons,
        },
        is_favorite: isFavorite,
        extra_metadata: Object.keys(metadata).length > 0 ? metadata : undefined,
      };

      await onSubmit(data as any);

      // Reset form on success (only in create mode)
      if (!isEditMode) {
        setName('');
        setDescription('');
        setModelId('');
        setDatasetId('');
        setEncoderType(SAEArchitectureType.STANDARD);
        setHiddenDim(768);
        setLatentDim(16384);
        setL1Alpha(0.001);
        setTargetL0('');
        setLearningRate(0.0003);
        setBatchSize(4096);
        setTotalSteps(100000);
        setWarmupSteps(1000);
        setWeightDecay(0.0);
        setGradClipNorm('');
        setCheckpointInterval(5000);
        setLogInterval(100);
        setDeadNeuronThreshold(1000);
        setResampleDeadNeurons(true);
        setIsFavorite(false);
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
          {isEditMode ? 'Edit Template' : 'Create Training Template'}
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

      <div className="space-y-6">
        {/* Basic Information */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
            Basic Information
          </h3>

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
              placeholder="My Training Template"
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

          {/* Model ID and Dataset ID */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="model-id" className="block text-sm font-medium text-slate-300 mb-2">
                Model ID (Optional)
              </label>
              <input
                id="model-id"
                type="text"
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="m_..."
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
              />
            </div>
            <div>
              <label htmlFor="dataset-id" className="block text-sm font-medium text-slate-300 mb-2">
                Dataset ID (Optional)
              </label>
              <input
                id="dataset-id"
                type="text"
                value={datasetId}
                onChange={(e) => setDatasetId(e.target.value)}
                placeholder="..."
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 font-mono focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
              />
            </div>
          </div>
        </div>

        {/* SAE Architecture */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
            SAE Architecture
          </h3>

          {/* Encoder Type */}
          <div>
            <label className="block text-sm font-medium text-slate-300 mb-2">
              Architecture Type <span className="text-red-400">*</span>
            </label>
            <div className="flex flex-wrap gap-3">
              {Object.values(SAEArchitectureType).map((type) => (
                <label
                  key={type}
                  className={`flex-1 min-w-[120px] px-4 py-3 bg-slate-800 border rounded cursor-pointer transition-all ${
                    encoderType === type
                      ? 'border-emerald-500 bg-emerald-500/10'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <input
                    type="radio"
                    name="encoder-type"
                    value={type}
                    checked={encoderType === type}
                    onChange={(e) => setEncoderType(e.target.value as SAEArchitectureType)}
                    disabled={isSubmitting}
                    className="sr-only"
                  />
                  <span
                    className={`text-sm font-medium capitalize ${
                      encoderType === type ? 'text-emerald-400' : 'text-slate-300'
                    }`}
                  >
                    {type}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Dimensions */}
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="hidden-dim" className="block text-sm font-medium text-slate-300 mb-2">
                Hidden Dimension <span className="text-red-400">*</span>
              </label>
              <input
                id="hidden-dim"
                type="number"
                value={hiddenDim}
                onChange={(e) => setHiddenDim(parseInt(e.target.value, 10))}
                min="1"
                max="100000"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
                required
              />
              <p className="text-xs text-slate-500 mt-1">Input/output size</p>
            </div>

            <div>
              <label htmlFor="latent-dim" className="block text-sm font-medium text-slate-300 mb-2">
                Latent Dimension <span className="text-red-400">*</span>
              </label>
              <input
                id="latent-dim"
                type="number"
                value={latentDim}
                onChange={(e) => setLatentDim(parseInt(e.target.value, 10))}
                min="1"
                max="1000000"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
                required
              />
              <p className="text-xs text-slate-500 mt-1">SAE width (typically 8-32x hidden)</p>
            </div>
          </div>
        </div>

        {/* Sparsity & Training */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wide">
            Sparsity & Training
          </h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label htmlFor="l1-alpha" className="block text-sm font-medium text-slate-300 mb-2">
                L1 Alpha <span className="text-red-400">*</span>
              </label>
              <input
                id="l1-alpha"
                type="number"
                value={l1Alpha}
                onChange={(e) => setL1Alpha(parseFloat(e.target.value))}
                step="0.0001"
                min="0.0001"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
                required
              />
              <p className="text-xs text-slate-500 mt-1">Sparsity penalty coefficient</p>
            </div>

            <div>
              <label htmlFor="target-l0" className="block text-sm font-medium text-slate-300 mb-2">
                Target L0 (Optional)
              </label>
              <input
                id="target-l0"
                type="number"
                value={targetL0}
                onChange={(e) => setTargetL0(e.target.value)}
                step="0.01"
                min="0"
                max="1"
                placeholder="0.05"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
              />
              <p className="text-xs text-slate-500 mt-1">Fraction of active features (0-1)</p>
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label htmlFor="learning-rate" className="block text-sm font-medium text-slate-300 mb-2">
                Learning Rate <span className="text-red-400">*</span>
              </label>
              <input
                id="learning-rate"
                type="number"
                value={learningRate}
                onChange={(e) => setLearningRate(parseFloat(e.target.value))}
                step="0.0001"
                min="0.0001"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
                required
              />
            </div>

            <div>
              <label htmlFor="batch-size" className="block text-sm font-medium text-slate-300 mb-2">
                Batch Size <span className="text-red-400">*</span>
              </label>
              <input
                id="batch-size"
                type="number"
                value={batchSize}
                onChange={(e) => setBatchSize(parseInt(e.target.value, 10))}
                min="1"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
                required
              />
            </div>

            <div>
              <label htmlFor="total-steps" className="block text-sm font-medium text-slate-300 mb-2">
                Total Steps <span className="text-red-400">*</span>
              </label>
              <input
                id="total-steps"
                type="number"
                value={totalSteps}
                onChange={(e) => setTotalSteps(parseInt(e.target.value, 10))}
                min="1"
                className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                disabled={isSubmitting}
                required
              />
            </div>
          </div>
        </div>

        {/* Advanced Settings */}
        <div>
          <button
            type="button"
            onClick={() => setShowAdvanced(!showAdvanced)}
            className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-emerald-400 transition-colors"
          >
            {showAdvanced ? (
              <ChevronUp className="w-4 h-4" />
            ) : (
              <ChevronDown className="w-4 h-4" />
            )}
            Advanced Settings
          </button>

          {showAdvanced && (
            <div className="mt-4 space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="warmup-steps" className="block text-sm font-medium text-slate-300 mb-2">
                    Warmup Steps
                  </label>
                  <input
                    id="warmup-steps"
                    type="number"
                    value={warmupSteps}
                    onChange={(e) => setWarmupSteps(parseInt(e.target.value, 10))}
                    min="0"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={isSubmitting}
                  />
                </div>

                <div>
                  <label htmlFor="weight-decay" className="block text-sm font-medium text-slate-300 mb-2">
                    Weight Decay
                  </label>
                  <input
                    id="weight-decay"
                    type="number"
                    value={weightDecay}
                    onChange={(e) => setWeightDecay(parseFloat(e.target.value))}
                    step="0.0001"
                    min="0"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={isSubmitting}
                  />
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div>
                  <label htmlFor="grad-clip" className="block text-sm font-medium text-slate-300 mb-2">
                    Gradient Clip Norm
                  </label>
                  <input
                    id="grad-clip"
                    type="number"
                    value={gradClipNorm}
                    onChange={(e) => setGradClipNorm(e.target.value)}
                    step="0.1"
                    min="0"
                    placeholder="1.0"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={isSubmitting}
                  />
                </div>

                <div>
                  <label htmlFor="checkpoint-interval" className="block text-sm font-medium text-slate-300 mb-2">
                    Checkpoint Interval
                  </label>
                  <input
                    id="checkpoint-interval"
                    type="number"
                    value={checkpointInterval}
                    onChange={(e) => setCheckpointInterval(parseInt(e.target.value, 10))}
                    min="1"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={isSubmitting}
                  />
                </div>

                <div>
                  <label htmlFor="log-interval" className="block text-sm font-medium text-slate-300 mb-2">
                    Log Interval
                  </label>
                  <input
                    id="log-interval"
                    type="number"
                    value={logInterval}
                    onChange={(e) => setLogInterval(parseInt(e.target.value, 10))}
                    min="1"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={isSubmitting}
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label htmlFor="dead-neuron-threshold" className="block text-sm font-medium text-slate-300 mb-2">
                    Dead Neuron Threshold
                  </label>
                  <input
                    id="dead-neuron-threshold"
                    type="number"
                    value={deadNeuronThreshold}
                    onChange={(e) => setDeadNeuronThreshold(parseInt(e.target.value, 10))}
                    min="1"
                    className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
                    disabled={isSubmitting}
                  />
                  <p className="text-xs text-slate-500 mt-1">Steps before neuron considered dead</p>
                </div>

                <div className="flex items-end">
                  <label className="flex items-center gap-2 cursor-pointer pb-2">
                    <input
                      type="checkbox"
                      checked={resampleDeadNeurons}
                      onChange={(e) => setResampleDeadNeurons(e.target.checked)}
                      disabled={isSubmitting}
                      className="w-4 h-4 text-emerald-600 bg-slate-700 border-slate-600 rounded focus:ring-2 focus:ring-emerald-500"
                    />
                    <span className="text-sm text-slate-300">Resample dead neurons</span>
                  </label>
                </div>
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
            </div>
          )}
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
            disabled={isSubmitting || !name.trim()}
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

/**
 * Training Card Component
 *
 * Displays individual training job with:
 * - Header (model, dataset, encoder type, status)
 * - Progress bar and metrics
 * - Live metrics charts (collapsible)
 * - Checkpoint management (collapsible)
 * - Control buttons (pause/resume/stop)
 *
 * Mock UI Reference: Lines 2730-3040
 * TID Reference: Lines 362-435
 *
 * Features:
 * - Real-time progress updates via WebSocket
 * - Expandable sections for metrics and checkpoints
 * - Color-coded status indicators
 * - Responsive layout with exact Mock UI styling
 */

import React, { useState } from 'react';
import {
  Activity,
  CheckCircle,
  Pause,
  Play,
  Loader,
  StopCircle,
  Download,
  Trash2,
  Save,
  Sliders,
  X,
} from 'lucide-react';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { TrainingStatus } from '../../types/training';
import type { Training } from '../../types/training';
import type { Model } from '../../types/model';
import type { Dataset } from '../../types/dataset';

interface TrainingCardProps {
  training: Training;
  isSelected: boolean;
  onToggleSelect: (trainingId: string) => void;
  models: Model[];
  datasets: Dataset[];
}

export const TrainingCard: React.FC<TrainingCardProps> = ({
  training,
  isSelected,
  onToggleSelect,
  models,
  datasets
}) => {
  const {
    pauseTraining,
    resumeTraining,
    stopTraining,
    retryTraining,
  } = useTrainingsStore();

  // UI state
  const [showMetrics, setShowMetrics] = useState(false);
  const [showCheckpoints, setShowCheckpoints] = useState(false);
  const [showHyperparameters, setShowHyperparameters] = useState(false);
  const [autoSave, setAutoSave] = useState(false);
  const [autoSaveInterval, setAutoSaveInterval] = useState(1000);
  const [isControlling, setIsControlling] = useState(false);

  // Calculate current metrics
  const hasMetrics = training.progress > 10;
  const currentLoss = training.current_loss ?? 0;
  const l0Sparsity = training.current_l0_sparsity ?? 0;
  const deadNeurons = training.current_dead_neurons ?? 0;
  const learningRate = training.current_learning_rate ?? 0;

  // Mock GPU util (would come from backend in production)
  const gpuUtil = training.status === TrainingStatus.RUNNING ? 75 + Math.random() * 15 : 0;

  // Mock checkpoints (would come from API in production)
  const trainingCheckpoints: any[] = [];

  // Look up human-readable names
  const modelName = models.find((m) => m.id === training.model_id)?.name || training.model_id;
  const datasetName = datasets.find((d) => d.id === training.dataset_id)?.name || training.dataset_id;

  // Calculate duration
  const calculateDuration = (startTime: string, endTime: string): string => {
    const start = new Date(startTime).getTime();
    const end = new Date(endTime).getTime();
    const durationMs = end - start;

    const seconds = Math.floor(durationMs / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      return `${days}d ${hours % 24}h`;
    } else if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    } else {
      return `${seconds}s`;
    }
  };

  // Format time
  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString();
  };

  // Handle control actions
  const handlePause = async () => {
    setIsControlling(true);
    try {
      await pauseTraining(training.id);
    } catch (error) {
      console.error('Failed to pause training:', error);
    } finally {
      setIsControlling(false);
    }
  };

  const handleResume = async () => {
    setIsControlling(true);
    try {
      await resumeTraining(training.id);
    } catch (error) {
      console.error('Failed to resume training:', error);
    } finally {
      setIsControlling(false);
    }
  };

  const handleStop = async () => {
    setIsControlling(true);
    try {
      await stopTraining(training.id);
    } catch (error) {
      console.error('Failed to stop training:', error);
    } finally {
      setIsControlling(false);
    }
  };

  const handleRetry = async () => {
    setIsControlling(true);
    try {
      await retryTraining(training.id);
    } catch (error) {
      console.error('Failed to retry training:', error);
    } finally {
      setIsControlling(false);
    }
  };

  // Status icon component
  const StatusIcon = () => {
    switch (training.status) {
      case TrainingStatus.RUNNING:
        return <Activity className="w-5 h-5 text-emerald-400 animate-pulse" />;
      case TrainingStatus.COMPLETED:
        return <CheckCircle className="w-5 h-5 text-emerald-400" />;
      case TrainingStatus.PAUSED:
        return <Pause className="w-5 h-5 text-yellow-400" />;
      case TrainingStatus.INITIALIZING:
        return <Loader className="w-5 h-5 text-blue-400 animate-spin" />;
      case TrainingStatus.FAILED:
        return <StopCircle className="w-5 h-5 text-red-400" />;
      default:
        return <Loader className="w-5 h-5 text-slate-400" />;
    }
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
      {/* Header Section */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <input
            type="checkbox"
            checked={isSelected}
            onChange={() => onToggleSelect(training.id)}
            className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900"
          />
          <div>
            <div className="flex items-center gap-3">
              <h4 className="font-semibold text-lg text-slate-100">
                Training {training.id.slice(0, 8)}
              </h4>
              <span className="text-sm text-slate-400">•</span>
              <span className="text-sm text-slate-300">
                {modelName}
              </span>
              <span className="text-sm text-slate-400">+</span>
              <span className="text-sm text-slate-300">
                {datasetName}
              </span>
            </div>
            <p className="text-sm text-slate-400">
              Encoder: {training.hyperparameters.architecture_type} •{' '}
              Started: {training.started_at ? formatTime(training.started_at) : 'Not started'}
              {training.completed_at && training.started_at && (
                <>
                  {' • '}
                  Completed: {formatTime(training.completed_at)}
                  {' • '}
                  Duration: {calculateDuration(training.started_at, training.completed_at)}
                </>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusIcon />
          <span className="capitalize px-3 py-1 bg-slate-800 rounded-full text-sm text-slate-100">
            {training.status}
          </span>
        </div>
      </div>

      {/* Key Hyperparameters Section */}
      <div className="bg-slate-800/50 rounded-lg p-3">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-slate-300">Key Parameters</span>
          <button
            onClick={() => setShowHyperparameters(!showHyperparameters)}
            className="text-slate-400 hover:text-emerald-400 transition-colors"
            title="View all hyperparameters"
          >
            <Sliders size={16} />
          </button>
        </div>
        <div className="grid grid-cols-3 gap-3 text-xs">
          <div>
            <span className="text-slate-400">Latent Dim: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters.latent_dim.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Learning Rate: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters.learning_rate}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Batch Size: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters.batch_size}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Total Steps: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters.total_steps.toLocaleString()}
            </span>
          </div>
          <div>
            <span className="text-slate-400">L1 Alpha: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters.l1_alpha}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Hidden Dim: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters.hidden_dim.toLocaleString()}
            </span>
          </div>
        </div>
      </div>

      {/* Progress Section */}
      {(training.status === TrainingStatus.RUNNING ||
        training.status === TrainingStatus.COMPLETED ||
        training.status === TrainingStatus.PAUSED) && (
        <div className="space-y-3">
          {/* Progress Label */}
          <div className="flex items-center justify-between text-sm">
            <span className="text-slate-400">Training Progress</span>
            <span className="text-emerald-400 font-medium">
              {training.progress.toFixed(1)}%
            </span>
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
              style={{ width: `${training.progress}%` }}
            />
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-4 gap-3 pt-2">
            {/* Loss */}
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Loss</div>
              <div className="text-lg font-semibold text-emerald-400">
                {hasMetrics ? currentLoss.toFixed(4) : '—'}
              </div>
            </div>

            {/* L0 Sparsity */}
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">L0 Sparsity</div>
              <div className="text-lg font-semibold text-blue-400">
                {hasMetrics ? l0Sparsity.toFixed(4) : '—'}
              </div>
            </div>

            {/* Dead Neurons */}
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">Dead Neurons</div>
              <div className="text-lg font-semibold text-red-400">
                {hasMetrics ? Math.floor(deadNeurons) : '—'}
              </div>
            </div>

            {/* GPU Util */}
            <div className="bg-slate-800/50 rounded-lg p-3">
              <div className="text-xs text-slate-400 mb-1">GPU Util</div>
              <div className="text-lg font-semibold text-purple-400">
                {training.status === TrainingStatus.RUNNING ? `${gpuUtil.toFixed(0)}%` : '—'}
              </div>
            </div>
          </div>

          {/* Toggle Buttons */}
          {training.status === TrainingStatus.RUNNING && (
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setShowMetrics(!showMetrics)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Activity className="w-4 h-4" />
                <span>{showMetrics ? 'Hide' : 'Show'} Live Metrics</span>
              </button>
              <button
                type="button"
                onClick={() => setShowCheckpoints(!showCheckpoints)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center justify-center gap-2"
              >
                <Download className="w-4 h-4" />
                <span>Checkpoints ({trainingCheckpoints.length})</span>
              </button>
            </div>
          )}

          {/* Checkpoint Management Section */}
          {showCheckpoints && (
            <div className="border-t border-slate-700 pt-4 mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <h5 className="text-sm font-medium text-slate-300">
                  Checkpoint Management
                </h5>
                <button
                  type="button"
                  onClick={() => console.log('Save checkpoint')}
                  className="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 rounded text-sm flex items-center gap-1 transition-colors"
                >
                  <Save className="w-4 h-4" />
                  Save Now
                </button>
              </div>

              {/* Checkpoint List */}
              {trainingCheckpoints.length > 0 ? (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {trainingCheckpoints.map((cp: any) => (
                    <div
                      key={cp.id}
                      className="flex items-center justify-between bg-slate-800/30 p-3 rounded"
                    >
                      <div>
                        <div className="font-medium text-sm text-slate-100">
                          Step {cp.step}
                        </div>
                        <div className="text-xs text-slate-400">
                          Loss: {cp.loss.toFixed(4)} •{' '}
                          {new Date(cp.timestamp).toLocaleTimeString()}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => console.log('Load checkpoint', cp.id)}
                          className="p-1 hover:bg-slate-700 rounded text-slate-300 hover:text-slate-100 transition-colors"
                          title="Load"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => console.log('Delete checkpoint', cp.id)}
                          className="p-1 hover:bg-red-900/30 text-red-400 rounded transition-colors"
                          title="Delete"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-slate-500 text-sm">
                  No checkpoints saved yet
                </div>
              )}

              {/* Auto-save Configuration */}
              <div className="border-t border-slate-700 pt-3 space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-sm text-slate-300">
                    Auto-save every N steps
                  </label>
                  <button
                    type="button"
                    onClick={() => setAutoSave(!autoSave)}
                    className={`relative w-12 h-6 rounded-full transition-colors ${
                      autoSave ? 'bg-emerald-600' : 'bg-slate-600'
                    }`}
                    aria-label={`Toggle auto-save ${autoSave ? 'off' : 'on'}`}
                  >
                    <div
                      className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                        autoSave ? 'translate-x-7' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
                {autoSave && (
                  <input
                    type="number"
                    value={autoSaveInterval}
                    onChange={(e) => setAutoSaveInterval(parseInt(e.target.value))}
                    min="100"
                    max="10000"
                    step="100"
                    aria-label="Auto-save interval in steps"
                    className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded text-sm text-slate-100 focus:outline-none focus:border-emerald-500 transition-colors"
                    placeholder="Interval (steps)"
                  />
                )}
              </div>
            </div>
          )}

          {/* Live Metrics Section */}
          {showMetrics && training.status === TrainingStatus.RUNNING && (
            <div className="border-t border-slate-700 pt-4 mt-4 space-y-4">
              {/* Loss Curve */}
              <div className="bg-slate-800/30 rounded-lg p-4">
                <h5 className="text-sm font-medium text-slate-300 mb-3">Loss Curve</h5>
                <div className="h-24 flex items-end gap-1">
                  {Array.from({ length: 20 }, (_, i) => {
                    const height = Math.max(10, 100 - i * 3 - Math.random() * 10);
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-emerald-500 rounded-t transition-all"
                        style={{ height: `${height}%` }}
                      />
                    );
                  })}
                </div>
              </div>

              {/* L0 Sparsity Chart */}
              <div className="bg-slate-800/30 rounded-lg p-4">
                <h5 className="text-sm font-medium text-slate-300 mb-3">L0 Sparsity</h5>
                <div className="h-24 flex items-end gap-1">
                  {Array.from({ length: 20 }, (_, i) => {
                    const height = Math.min(90, 30 + i * 2 + Math.random() * 10);
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-blue-500 rounded-t transition-all"
                        style={{ height: `${height}%` }}
                      />
                    );
                  })}
                </div>
              </div>

              {/* Training Logs */}
              <div className="bg-slate-950 rounded-lg p-4 font-mono text-xs">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-slate-400">Training Logs</span>
                  <span className="text-emerald-400 text-xs">Live</span>
                </div>
                <div className="h-32 overflow-y-auto space-y-1">
                  <div className="text-slate-300">
                    <span className="text-slate-500">
                      [{new Date().toLocaleTimeString()}]
                    </span>{' '}
                    Step {training.current_step}: loss={currentLoss.toFixed(4)}
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">
                      [{new Date().toLocaleTimeString()}]
                    </span>{' '}
                    L0 sparsity: {l0Sparsity.toFixed(4)}
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">
                      [{new Date().toLocaleTimeString()}]
                    </span>{' '}
                    Dead neurons: {Math.floor(deadNeurons)}/
                    {training.hyperparameters.latent_dim}
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">
                      [{new Date().toLocaleTimeString()}]
                    </span>{' '}
                    GPU utilization: {gpuUtil.toFixed(0)}%
                  </div>
                  <div className="text-slate-300">
                    <span className="text-slate-500">
                      [{new Date().toLocaleTimeString()}]
                    </span>{' '}
                    Learning rate: {learningRate.toExponential(2)}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Control Buttons */}
      {training.status !== TrainingStatus.COMPLETED && (
        <div className="border-t border-slate-700 pt-4 flex gap-2">
          {/* Training Status: Show Pause and Stop */}
          {training.status === TrainingStatus.RUNNING && (
            <>
              <button
                type="button"
                onClick={handlePause}
                disabled={isControlling}
                className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                {isControlling ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <Pause className="w-4 h-4" />
                )}
                Pause
              </button>
              <button
                type="button"
                onClick={handleStop}
                disabled={isControlling}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                {isControlling ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <StopCircle className="w-4 h-4" />
                )}
                Stop
              </button>
            </>
          )}

          {/* Paused Status: Show Resume and Stop */}
          {training.status === TrainingStatus.PAUSED && (
            <>
              <button
                type="button"
                onClick={handleResume}
                disabled={isControlling}
                className="flex-1 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                {isControlling ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Resume
              </button>
              <button
                type="button"
                onClick={handleStop}
                disabled={isControlling}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                {isControlling ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <StopCircle className="w-4 h-4" />
                )}
                Stop
              </button>
            </>
          )}

          {/* Failed/Cancelled Status: Show Retry */}
          {(training.status === TrainingStatus.FAILED ||
            training.status === TrainingStatus.CANCELLED) && (
            <button
              type="button"
              onClick={handleRetry}
              disabled={isControlling}
              className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              {isControlling ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Play className="w-4 h-4" />
              )}
              Retry
            </button>
          )}
        </div>
      )}

      {/* Hyperparameters Modal */}
      {showHyperparameters && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-auto">
            {/* Modal Header */}
            <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-6 py-4 flex items-center justify-between">
              <h3 className="text-lg font-semibold text-slate-100">
                Training Hyperparameters
              </h3>
              <button
                onClick={() => setShowHyperparameters(false)}
                className="text-slate-400 hover:text-slate-300 transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            {/* Modal Content */}
            <div className="px-6 py-4 space-y-6">
              {/* SAE Architecture Section */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-3">SAE Architecture</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Architecture Type</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.architecture_type}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Hidden Dimension</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.hidden_dim.toLocaleString()}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Latent Dimension</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.latent_dim.toLocaleString()}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Expansion Ratio</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {(training.hyperparameters.latent_dim / training.hyperparameters.hidden_dim).toFixed(1)}x
                    </div>
                  </div>
                </div>
              </div>

              {/* Sparsity Section */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-3">Sparsity Configuration</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">L1 Alpha</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.l1_alpha}
                    </div>
                  </div>
                  {training.hyperparameters.target_l0 && (
                    <div className="bg-slate-800/50 rounded-lg p-3">
                      <div className="text-xs text-slate-400 mb-1">Target L0</div>
                      <div className="text-sm text-slate-100 font-medium">
                        {training.hyperparameters.target_l0}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Training Configuration Section */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-3">Training Configuration</h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Learning Rate</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.learning_rate}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Batch Size</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.batch_size}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-3">
                    <div className="text-xs text-slate-400 mb-1">Total Steps</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.total_steps.toLocaleString()}
                    </div>
                  </div>
                  {training.hyperparameters.warmup_steps && (
                    <div className="bg-slate-800/50 rounded-lg p-3">
                      <div className="text-xs text-slate-400 mb-1">Warmup Steps</div>
                      <div className="text-sm text-slate-100 font-medium">
                        {training.hyperparameters.warmup_steps.toLocaleString()}
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Optimization Section */}
              {(training.hyperparameters.weight_decay || training.hyperparameters.grad_clip_norm) && (
                <div>
                  <h4 className="text-sm font-semibold text-emerald-400 mb-3">Optimization</h4>
                  <div className="grid grid-cols-2 gap-4">
                    {training.hyperparameters.weight_decay && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-xs text-slate-400 mb-1">Weight Decay</div>
                        <div className="text-sm text-slate-100 font-medium">
                          {training.hyperparameters.weight_decay}
                        </div>
                      </div>
                    )}
                    {training.hyperparameters.grad_clip_norm && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-xs text-slate-400 mb-1">Gradient Clip Norm</div>
                        <div className="text-sm text-slate-100 font-medium">
                          {training.hyperparameters.grad_clip_norm}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Checkpointing Section */}
              {(training.hyperparameters.checkpoint_interval || training.hyperparameters.log_interval) && (
                <div>
                  <h4 className="text-sm font-semibold text-emerald-400 mb-3">Checkpointing & Logging</h4>
                  <div className="grid grid-cols-2 gap-4">
                    {training.hyperparameters.checkpoint_interval && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-xs text-slate-400 mb-1">Checkpoint Interval</div>
                        <div className="text-sm text-slate-100 font-medium">
                          Every {training.hyperparameters.checkpoint_interval.toLocaleString()} steps
                        </div>
                      </div>
                    )}
                    {training.hyperparameters.log_interval && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-xs text-slate-400 mb-1">Log Interval</div>
                        <div className="text-sm text-slate-100 font-medium">
                          Every {training.hyperparameters.log_interval.toLocaleString()} steps
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Dead Neuron Handling Section */}
              {(training.hyperparameters.dead_neuron_threshold !== undefined ||
                training.hyperparameters.resample_dead_neurons !== undefined) && (
                <div>
                  <h4 className="text-sm font-semibold text-emerald-400 mb-3">Dead Neuron Handling</h4>
                  <div className="grid grid-cols-2 gap-4">
                    {training.hyperparameters.dead_neuron_threshold !== undefined && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-xs text-slate-400 mb-1">Dead Neuron Threshold</div>
                        <div className="text-sm text-slate-100 font-medium">
                          {training.hyperparameters.dead_neuron_threshold.toLocaleString()} steps
                        </div>
                      </div>
                    )}
                    {training.hyperparameters.resample_dead_neurons !== undefined && (
                      <div className="bg-slate-800/50 rounded-lg p-3">
                        <div className="text-xs text-slate-400 mb-1">Resample Dead Neurons</div>
                        <div className="text-sm text-slate-100 font-medium">
                          {training.hyperparameters.resample_dead_neurons ? 'Enabled' : 'Disabled'}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>

            {/* Modal Footer */}
            <div className="sticky bottom-0 bg-slate-900 border-t border-slate-700 px-6 py-4">
              <button
                onClick={() => setShowHyperparameters(false)}
                className="w-full px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-100 rounded-lg transition-colors"
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {training.error_message && (
        <div className="bg-red-900/20 border border-red-900/50 rounded-lg p-4">
          <div className="text-sm font-medium text-red-400 mb-1">Training Error</div>
          <div className="text-xs text-red-300">{training.error_message}</div>
        </div>
      )}
    </div>
  );
};

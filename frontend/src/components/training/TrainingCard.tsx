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
} from 'lucide-react';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { TrainingStatus } from '../../types/training';
import type { Training } from '../../types/training';

interface TrainingCardProps {
  training: Training;
}

export const TrainingCard: React.FC<TrainingCardProps> = ({ training }) => {
  const {
    pauseTraining,
    resumeTraining,
    stopTraining,
    deleteTraining,
  } = useTrainingsStore();

  // UI state
  const [showMetrics, setShowMetrics] = useState(false);
  const [showCheckpoints, setShowCheckpoints] = useState(false);
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

  const handleDelete = async () => {
    if (!confirm('Are you sure you want to delete this training job? This action cannot be undone.')) {
      return;
    }
    setIsControlling(true);
    try {
      await deleteTraining(training.id);
    } catch (error) {
      console.error('Failed to delete training:', error);
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
        <div>
          <h4 className="font-semibold text-lg text-slate-100">
            {training.model_id} + {training.dataset_id}
          </h4>
          <p className="text-sm text-slate-400">
            Encoder: {training.hyperparameters.architecture_type} •{' '}
            Started: {training.started_at ? formatTime(training.started_at) : 'Not started'}
          </p>
        </div>
        <div className="flex items-center gap-3">
          <StatusIcon />
          <span className="capitalize px-3 py-1 bg-slate-800 rounded-full text-sm text-slate-100">
            {training.status}
          </span>
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

          {/* Failed/Cancelled Status: Show Retry and Delete */}
          {(training.status === TrainingStatus.FAILED ||
            training.status === TrainingStatus.CANCELLED) && (
            <>
              <button
                type="button"
                onClick={() => console.log('Retry training')}
                className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                <Play className="w-4 h-4" />
                Retry
              </button>
              <button
                type="button"
                onClick={handleDelete}
                disabled={isControlling}
                className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
              >
                {isControlling ? (
                  <Loader className="w-4 h-4 animate-spin" />
                ) : (
                  <Trash2 className="w-4 h-4" />
                )}
                Delete
              </button>
            </>
          )}

          {/* Completed Status: Show Delete */}
          {training.status === TrainingStatus.COMPLETED && (
            <button
              type="button"
              onClick={handleDelete}
              disabled={isControlling}
              className="flex-1 px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg flex items-center justify-center gap-2 transition-colors"
            >
              {isControlling ? (
                <Loader className="w-4 h-4 animate-spin" />
              ) : (
                <Trash2 className="w-4 h-4" />
              )}
              Delete
            </button>
          )}
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

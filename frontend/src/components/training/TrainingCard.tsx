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

import React, { useState, useEffect } from 'react';
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
  Zap,
  Brain,
  Copy,
  Check,
  Clock,
} from 'lucide-react';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { TrainingStatus } from '../../types/training';
import { SAEImportFromTrainingResponse } from '../../types/sae';
import SAEImportModal from './SAEImportModal';
import type { Training } from '../../types/training';
import type { Model } from '../../types/model';
import type { Dataset } from '../../types/dataset';
import { COMPONENTS } from '../../config/brand';
import { formatL0Absolute } from '../../utils/formatters';
import { fetchTrainingMetrics } from '../../api/trainings';

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
    fetchCheckpoints,
    saveCheckpoint,
    deleteCheckpoint,
  } = useTrainingsStore();

  // UI state
  const [showMetrics, setShowMetrics] = useState(false);
  const [showCheckpoints, setShowCheckpoints] = useState(false);
  const [showHyperparameters, setShowHyperparameters] = useState(false);
  const [autoSave, setAutoSave] = useState(false);
  const [autoSaveInterval, setAutoSaveInterval] = useState(1000);
  const [isControlling, setIsControlling] = useState(false);
  const [isSavingCheckpoint, setIsSavingCheckpoint] = useState(false);
  const [checkpoints, setCheckpoints] = useState<any[]>([]);

  // SAE import modal state
  const [showSAEImportModal, setShowSAEImportModal] = useState(false);
  const [importedSAECount, setImportedSAECount] = useState(0);

  // Metrics history for charts (keep last 20 points)
  // Uses step numbers for deduplication (unique per training iteration)
  const [metricsHistory, setMetricsHistory] = useState<{
    loss: number[];
    l0_sparsity: number[];
    dead_neurons: number[];  // Historical dead neuron counts
    timestamps: string[];
    steps: number[];  // Track step numbers for deduplication
  }>({
    loss: [],
    l0_sparsity: [],
    dead_neurons: [],
    timestamps: [],
    steps: [],
  });

  // Elapsed time state for running trainings
  const [elapsedTime, setElapsedTime] = useState<string>('');

  // Track whether we've loaded historical metrics
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const [logsCopied, setLogsCopied] = useState(false);
  const [hasLoadedHistoricalMetrics, setHasLoadedHistoricalMetrics] = useState(false);

  // Calculate current metrics
  const hasMetrics = training.progress > 10;
  const currentLoss = training.current_loss ?? 0;
  const l0Sparsity = training.current_l0_sparsity ?? 0;
  const deadNeurons = training.current_dead_neurons ?? 0;
  const learningRate = training.current_learning_rate ?? 0;

  // Calculate training phase
  const getTrainingPhase = () => {
    if (training.status !== TrainingStatus.RUNNING && training.status !== TrainingStatus.PAUSED) {
      return null;
    }

    const currentStep = training.current_step;
    const totalSteps = training.total_steps;
    const warmupSteps = training.hyperparameters.warmup_steps || 0;

    if (currentStep < warmupSteps) {
      const warmupProgress = (currentStep / warmupSteps) * 100;
      return {
        name: 'Warmup',
        description: `LR ramping up (${warmupProgress.toFixed(0)}%)`,
        color: 'text-blue-400',
        bgColor: 'bg-blue-500/10',
        borderColor: 'border-blue-500/30',
      };
    } else if (currentStep < warmupSteps + 10000) {
      return {
        name: 'Post-Warmup',
        description: 'L0 convergence phase',
        color: 'text-purple-400',
        bgColor: 'bg-purple-500/10',
        borderColor: 'border-purple-500/30',
      };
    } else if (currentStep < totalSteps * 0.5) {
      return {
        name: 'Mid-Training',
        description: 'Feature learning',
        color: 'text-cyan-400',
        bgColor: 'bg-cyan-500/10',
        borderColor: 'border-cyan-500/30',
      };
    } else if (currentStep < totalSteps * 0.9) {
      return {
        name: 'Late Training',
        description: 'Refinement phase',
        color: 'text-amber-400',
        bgColor: 'bg-amber-500/10',
        borderColor: 'border-amber-500/30',
      };
    } else {
      return {
        name: 'Final Phase',
        description: 'Convergence',
        color: 'text-emerald-400',
        bgColor: 'bg-emerald-500/10',
        borderColor: 'border-emerald-500/30',
      };
    }
  };

  const trainingPhase = getTrainingPhase();

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

  // Handle opening SAE import modal
  const handleOpenImportModal = () => {
    setShowSAEImportModal(true);
  };

  // Handle SAE import completion
  const handleImportComplete = (response: SAEImportFromTrainingResponse) => {
    setImportedSAECount(response.imported_count);
    setShowSAEImportModal(false);
    console.log('[TrainingCard] SAEs imported successfully:', response.sae_ids);
  };

  // Update elapsed time for running trainings
  useEffect(() => {
    if (training.status !== TrainingStatus.RUNNING || !training.started_at) {
      return;
    }

    const updateElapsed = () => {
      const start = new Date(training.started_at!).getTime();
      const now = Date.now();
      const durationMs = now - start;

      const seconds = Math.floor(durationMs / 1000);
      const minutes = Math.floor(seconds / 60);
      const hours = Math.floor(minutes / 60);
      const days = Math.floor(hours / 24);

      let formatted: string;
      if (days > 0) {
        formatted = `${days}d ${hours % 24}h ${minutes % 60}m`;
      } else if (hours > 0) {
        formatted = `${hours}h ${minutes % 60}m ${seconds % 60}s`;
      } else if (minutes > 0) {
        formatted = `${minutes}m ${seconds % 60}s`;
      } else {
        formatted = `${seconds}s`;
      }
      setElapsedTime(formatted);
    };

    // Update immediately
    updateElapsed();

    // Update every second
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [training.status, training.started_at]);

  // Fetch checkpoints on mount to show correct count
  useEffect(() => {
    let cancelled = false;

    fetchCheckpoints(training.id)
      .then((data) => {
        if (!cancelled) {
          setCheckpoints(data);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          console.error('Failed to load checkpoints:', error);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [training.id]);

  // Fetch historical metrics when Live Metrics panel is opened
  useEffect(() => {
    const loadHistoricalMetrics = async () => {
      // Only fetch if:
      // 1. showMetrics is true (panel is open)
      // 2. We haven't already loaded historical metrics
      // 3. Training is running or has some progress
      if (!showMetrics || hasLoadedHistoricalMetrics || training.progress === 0) {
        return;
      }

      setIsLoadingMetrics(true);
      try {
        // Fetch last 20 metrics from the backend
        const metrics = await fetchTrainingMetrics(training.id, { limit: 20 });

        if (metrics.length > 0) {
          // Sort by step to ensure correct order
          const sortedMetrics = metrics.sort((a, b) => a.step - b.step);

          // Deduplicate by step number (guaranteed unique per training iteration)
          const seen = new Set<number>();
          const dedupedMetrics = sortedMetrics.filter((m) => {
            if (seen.has(m.step)) return false;
            seen.add(m.step);
            return true;
          });

          setMetricsHistory({
            loss: dedupedMetrics.map((m) => m.loss),
            l0_sparsity: dedupedMetrics.map((m) => m.l0_sparsity ?? 0),
            dead_neurons: dedupedMetrics.map((m) => m.dead_neurons ?? 0),
            timestamps: dedupedMetrics.map((m) => m.timestamp),
            steps: dedupedMetrics.map((m) => m.step),
          });
        }

        setHasLoadedHistoricalMetrics(true);
      } catch (error) {
        console.error('[TrainingCard] Failed to fetch historical metrics:', error);
        // Still mark as loaded to prevent retry loops
        setHasLoadedHistoricalMetrics(true);
      } finally {
        setIsLoadingMetrics(false);
      }
    };

    loadHistoricalMetrics();
  }, [showMetrics, hasLoadedHistoricalMetrics, training.id, training.progress]);

  // Update metrics history when training metrics change (WebSocket updates)
  // Uses step number for deduplication - each step can only appear once
  useEffect(() => {
    const currentStep = training.current_step;
    const currentLossValue = training.current_loss;

    // Only add if we have valid metrics and a step number
    if (currentLossValue !== undefined && currentLossValue !== null && currentStep !== undefined) {
      setMetricsHistory((prev) => {
        // Deduplicate by step number - if this step already exists, skip
        if (prev.steps.includes(currentStep)) {
          return prev;
        }

        // Create timestamp for this update
        const newTimestamp = new Date().toISOString();

        const newLoss = [...prev.loss, currentLossValue];
        const newL0 = [...prev.l0_sparsity, training.current_l0_sparsity ?? 0];
        const newDeadNeurons = [...prev.dead_neurons, training.current_dead_neurons ?? 0];
        const newTimestamps = [...prev.timestamps, newTimestamp];
        const newSteps = [...prev.steps, currentStep];

        // Keep only last 20 points
        return {
          loss: newLoss.slice(-20),
          l0_sparsity: newL0.slice(-20),
          dead_neurons: newDeadNeurons.slice(-20),
          timestamps: newTimestamps.slice(-20),
          steps: newSteps.slice(-20),
        };
      });
    }
  }, [training.current_loss, training.current_l0_sparsity, training.current_dead_neurons, training.current_step]);


  // Handle save checkpoint
  const handleSaveCheckpoint = async () => {
    setIsSavingCheckpoint(true);
    try {
      const newCheckpoint = await saveCheckpoint(training.id);
      setCheckpoints((prev) => [newCheckpoint, ...prev]);
    } catch (error) {
      console.error('Failed to save checkpoint:', error);
    } finally {
      setIsSavingCheckpoint(false);
    }
  };

  // Handle delete checkpoint
  const handleDeleteCheckpoint = async (checkpointId: string) => {
    if (!confirm('Are you sure you want to delete this checkpoint?')) {
      return;
    }

    try {
      await deleteCheckpoint(training.id, checkpointId);
      setCheckpoints((prev) => prev.filter((c) => c.id !== checkpointId));
    } catch (error) {
      console.error('Failed to delete checkpoint:', error);
    }
  };

  // Handle copy training logs to clipboard
  const handleCopyLogs = async () => {
    console.log('[TrainingCard] handleCopyLogs called, metricsHistory:', metricsHistory);
    if (metricsHistory.loss.length === 0) {
      console.log('[TrainingCard] No metrics to copy');
      return;
    }

    // Format all log entries (not just last 10)
    const logLines = metricsHistory.steps.map((step, idx) => {
      const loss = metricsHistory.loss[idx];
      const sparsity = metricsHistory.l0_sparsity[idx];
      const historicalDeadNeurons = metricsHistory.dead_neurons[idx] ?? deadNeurons;
      const time = new Date(metricsHistory.timestamps[idx]).toLocaleTimeString();
      const latentDim = training.hyperparameters?.latent_dim || 'N/A';

      return `[${time}] step=${step}, loss=${loss.toFixed(4)}, L0=${sparsity.toFixed(4)}, dead=${Math.floor(historicalDeadNeurons)}/${latentDim}, lr=${learningRate.toExponential(2)}`;
    });

    const logText = logLines.join('\n');

    try {
      // Try modern clipboard API first (requires HTTPS)
      if (navigator.clipboard && window.isSecureContext) {
        await navigator.clipboard.writeText(logText);
      } else {
        // Fallback for HTTP: use textarea + execCommand
        const textarea = document.createElement('textarea');
        textarea.value = logText;
        textarea.style.position = 'fixed';
        textarea.style.left = '-9999px';
        textarea.style.top = '-9999px';
        document.body.appendChild(textarea);
        textarea.focus();
        textarea.select();
        document.execCommand('copy');
        document.body.removeChild(textarea);
      }
      setLogsCopied(true);
      setTimeout(() => setLogsCopied(false), 2000);
    } catch (error) {
      console.error('Failed to copy logs:', error);
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
    <div className={`${COMPONENTS.card.base} p-4 space-y-3`}>
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
              <h4 className="font-semibold text-base text-slate-100">
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
              Encoder: {training.hyperparameters?.architecture_type || 'N/A'}
              {training.hyperparameters?.training_layers && training.hyperparameters.training_layers.length > 0 && (
                <>
                  {' • '}
                  Layer(s): {training.hyperparameters.training_layers.map(l => `L${l}`).join(', ')}
                </>
              )}
              {training.hyperparameters?.hook_types && training.hyperparameters.hook_types.length > 0 && (
                <>
                  {' • '}
                  Hook(s): {training.hyperparameters.hook_types.join(', ')}
                </>
              )}
              {' • '}
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
      <div className="bg-slate-800/50 rounded-lg p-2">
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
        <div className="grid grid-cols-3 gap-2 text-xs">
          <div>
            <span className="text-slate-400">Latent Dim: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters?.latent_dim?.toLocaleString() || 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Learning Rate: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters?.learning_rate ?? 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Batch Size: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters?.batch_size ?? 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-slate-400">Total Steps: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters?.total_steps?.toLocaleString() || 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-slate-400">L1 Alpha: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters?.l1_alpha ?? 'N/A'}
            </span>
          </div>
          {training.hyperparameters?.top_k_sparsity && (
            <div>
              <span className="text-slate-400">Top-K: </span>
              <span className="text-emerald-400 font-medium">
                {training.hyperparameters.top_k_sparsity}%
              </span>
            </div>
          )}
          <div>
            <span className="text-slate-400">Hidden Dim: </span>
            <span className="text-slate-100 font-medium">
              {training.hyperparameters?.hidden_dim?.toLocaleString() || 'N/A'}
            </span>
          </div>
        </div>
      </div>

      {/* Progress Section */}
      {(training.status === TrainingStatus.RUNNING ||
        training.status === TrainingStatus.COMPLETED ||
        training.status === TrainingStatus.PAUSED) && (
        <div className="space-y-3">
          {/* Progress Label with Phase Indicator */}
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-2">
              <span className="text-slate-400">Training Progress</span>
              {trainingPhase && (
                <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-md border ${trainingPhase.bgColor} ${trainingPhase.borderColor}`}>
                  <Zap className={`w-3 h-3 ${trainingPhase.color}`} />
                  <span className={`text-xs font-medium ${trainingPhase.color}`}>
                    {trainingPhase.name}
                  </span>
                  <span className="text-xs text-slate-500">·</span>
                  <span className="text-xs text-slate-400">
                    {trainingPhase.description}
                  </span>
                </div>
              )}
            </div>
            <div className="flex items-center gap-3">
              {training.status === TrainingStatus.RUNNING && elapsedTime && (
                <div className="flex items-center gap-1.5 text-slate-400">
                  <Clock className="w-3.5 h-3.5" />
                  <span className="text-sm">{elapsedTime}</span>
                </div>
              )}
              <span className="text-emerald-400 font-medium">
                {training.progress.toFixed(1)}%
              </span>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-emerald-400 transition-all duration-500"
              style={{ width: `${training.progress}%` }}
            />
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-4 gap-2 pt-2">
            {/* Loss */}
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 mb-1">Loss</div>
              <div className="text-lg font-semibold text-emerald-400">
                {hasMetrics ? currentLoss.toFixed(4) : '—'}
              </div>
            </div>

            {/* L0 Sparsity - Neuronpedia-aligned display */}
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 mb-1">L0 Sparsity</div>
              <div className="flex flex-col">
                <div
                  className={`text-lg font-semibold ${
                    !hasMetrics
                      ? 'text-slate-400'
                      : l0Sparsity > 0.15
                      ? 'text-red-400'
                      : l0Sparsity > 0.08
                      ? 'text-yellow-400'
                      : 'text-emerald-400'
                  }`}
                  title={hasMetrics && training.hyperparameters?.latent_dim
                    ? `~${Math.round(l0Sparsity * training.hyperparameters.latent_dim)} of ${training.hyperparameters.latent_dim} features active per token`
                    : undefined
                  }
                >
                  {hasMetrics
                    ? training.hyperparameters?.latent_dim
                      ? formatL0Absolute(l0Sparsity, training.hyperparameters.latent_dim)
                      : (l0Sparsity * 100).toFixed(2) + '%'
                    : '—'}
                </div>
                {hasMetrics && training.hyperparameters?.latent_dim && (
                  <div className="text-xs text-slate-500">
                    {(l0Sparsity * 100).toFixed(1)}% of {training.hyperparameters.latent_dim}
                  </div>
                )}
                {hasMetrics && training.hyperparameters?.target_l0 && (
                  <div className="text-xs text-slate-500">
                    Target: {training.hyperparameters.latent_dim
                      ? formatL0Absolute(training.hyperparameters.target_l0, training.hyperparameters.latent_dim)
                      : (training.hyperparameters.target_l0 * 100).toFixed(1) + '%'}
                  </div>
                )}
              </div>
            </div>

            {/* Dead Neurons */}
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 mb-1">Dead Neurons</div>
              <div className="text-lg font-semibold text-red-400">
                {hasMetrics ? Math.floor(deadNeurons) : '—'}
              </div>
            </div>

            {/* Learning Rate */}
            <div className="bg-slate-800/50 rounded-lg p-2">
              <div className="text-xs text-slate-400 mb-1">Learning Rate</div>
              <div className="text-lg font-semibold text-purple-400">
                {hasMetrics ? learningRate.toExponential(2) : '—'}
              </div>
            </div>
          </div>

          {/* Toggle Buttons - different layouts per status */}
          {training.status === TrainingStatus.COMPLETED && (
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setShowCheckpoints(!showCheckpoints)}
                className={`flex items-center justify-center gap-2 rounded-lg ${COMPONENTS.button.secondary}`}
              >
                <Download className="w-4 h-4" />
                <span>Checkpoints ({checkpoints.length})</span>
              </button>
              <button
                type="button"
                onClick={handleOpenImportModal}
                disabled={importedSAECount > 0}
                className={`flex items-center justify-center gap-2 rounded-lg ${COMPONENTS.button.secondary} ${
                  importedSAECount > 0 ? 'bg-emerald-500/20 border-emerald-500/50' : ''
                }`}
              >
                <Brain className="w-4 h-4" />
                <span>
                  {importedSAECount > 0
                    ? `Imported (${importedSAECount})`
                    : 'Import to SAEs'}
                </span>
              </button>
            </div>
          )}
          {training.status === TrainingStatus.RUNNING && (
            <div className="grid grid-cols-2 gap-2">
              <button
                type="button"
                onClick={() => setShowMetrics(!showMetrics)}
                className={`flex items-center justify-center gap-2 rounded-lg ${COMPONENTS.button.secondary}`}
              >
                <Activity className="w-4 h-4" />
                <span>{showMetrics ? 'Hide' : 'Show'} Live Metrics</span>
              </button>
              <button
                type="button"
                onClick={() => setShowCheckpoints(!showCheckpoints)}
                className={`flex items-center justify-center gap-2 rounded-lg ${COMPONENTS.button.secondary}`}
              >
                <Download className="w-4 h-4" />
                <span>Checkpoints ({checkpoints.length})</span>
              </button>
            </div>
          )}
          {training.status === TrainingStatus.PAUSED && (
            <div className="grid grid-cols-1 gap-2">
              <button
                type="button"
                onClick={() => setShowCheckpoints(!showCheckpoints)}
                className={`flex items-center justify-center gap-2 rounded-lg ${COMPONENTS.button.secondary}`}
              >
                <Download className="w-4 h-4" />
                <span>Checkpoints ({checkpoints.length})</span>
              </button>
            </div>
          )}

          {/* Checkpoint Management Section */}
          {showCheckpoints && (
            <div className="border-t border-slate-700 pt-3 mt-3 space-y-2">
              <div className="flex items-center justify-between">
                <h5 className="text-sm font-medium text-slate-300">
                  Checkpoint Management
                </h5>
                <button
                  type="button"
                  onClick={handleSaveCheckpoint}
                  disabled={isSavingCheckpoint}
                  className="px-3 py-1 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:text-slate-500 rounded text-sm flex items-center gap-1 transition-colors"
                >
                  {isSavingCheckpoint ? (
                    <Loader className="w-4 h-4 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4" />
                  )}
                  {isSavingCheckpoint ? 'Saving...' : 'Save Now'}
                </button>
              </div>

              {/* Checkpoint List */}
              {checkpoints.length > 0 ? (
                <div className="space-y-2 max-h-48 overflow-y-auto">
                  {checkpoints.map((cp: any) => (
                    <div
                      key={cp.id}
                      className="flex items-center justify-between bg-slate-800/30 p-2 rounded"
                    >
                      <div>
                        <div className="font-medium text-sm text-slate-100">
                          Step {cp.step.toLocaleString()}
                        </div>
                        <div className="text-xs text-slate-400">
                          Loss: {cp.loss.toFixed(4)}
                          {cp.l0_sparsity && ` • L0: ${cp.l0_sparsity.toFixed(3)}`}
                          {' • '}
                          {new Date(cp.created_at).toLocaleTimeString()}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={() => {
                            // Download checkpoint (future implementation)
                            console.log('Download checkpoint', cp.id);
                          }}
                          className="p-1 hover:bg-slate-700 rounded text-slate-300 hover:text-slate-100 transition-colors"
                          title="Download"
                        >
                          <Download className="w-4 h-4" />
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDeleteCheckpoint(cp.id)}
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
              <div className="border-t border-slate-700 pt-2 space-y-2">
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

          {/* Live Metrics Section - Two column layout: Logs left, Charts right */}
          {showMetrics && training.status === TrainingStatus.RUNNING && (
            <div className="border-t border-slate-700 pt-3 mt-3">
              {/* Loading State */}
              {isLoadingMetrics && (
                <div className="flex items-center justify-center py-4 text-slate-400">
                  <Loader className="w-5 h-5 animate-spin mr-2" />
                  <span className="text-sm">Loading metrics history...</span>
                </div>
              )}

              {/* Two Column Layout */}
              <div className="flex gap-3">
                {/* Training Logs - Left Column (50%) */}
                <div className="w-1/2 bg-slate-950 rounded-lg p-3 font-mono text-xs">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-slate-400">Training Logs</span>
                    <div className="flex items-center gap-2">
                      <button
                        type="button"
                        onClick={handleCopyLogs}
                        disabled={metricsHistory.loss.length === 0}
                        className="p-1 hover:bg-slate-800 rounded text-slate-400 hover:text-slate-200 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                        title={logsCopied ? 'Copied!' : 'Copy logs to clipboard'}
                      >
                        {logsCopied ? (
                          <Check className="w-3.5 h-3.5 text-emerald-400" />
                        ) : (
                          <Copy className="w-3.5 h-3.5" />
                        )}
                      </button>
                      <span className="text-emerald-400 text-xs">
                        {hasLoadedHistoricalMetrics ? 'Live' : 'Loading...'}
                      </span>
                    </div>
                  </div>
                  <div className="h-52 overflow-y-auto space-y-1">
                    {metricsHistory.loss.length > 0 ? (
                      metricsHistory.steps.slice(-15).map((step, i) => {
                        const idx = metricsHistory.steps.length - 15 + i;
                        if (idx < 0) return null;

                        const loss = metricsHistory.loss[idx];
                        const sparsity = metricsHistory.l0_sparsity[idx];
                        const historicalDeadNeurons = metricsHistory.dead_neurons[idx] ?? deadNeurons;
                        const time = new Date(metricsHistory.timestamps[idx]).toLocaleTimeString();

                        return (
                          <div key={step} className="text-slate-300">
                            <span className="text-slate-500">[{time}]</span>{' '}
                            loss={loss.toFixed(4)},
                            L0={sparsity.toFixed(4)},
                            dead={Math.floor(historicalDeadNeurons)}/{training.hyperparameters?.latent_dim || 'N/A'},
                            lr={learningRate.toExponential(2)},
                            step={step.toLocaleString()}
                          </div>
                        );
                      }).filter(Boolean).reverse()
                    ) : (
                      <div className="text-slate-500 text-center py-4">
                        {isLoadingMetrics ? 'Loading metrics history...' : 'Waiting for training metrics...'}
                      </div>
                    )}
                  </div>
                </div>

                {/* Stacked Line Charts - Right Column (50%) */}
                <div className="w-1/2 flex flex-col gap-2">
                  {/* Loss Chart */}
                  <div className="bg-slate-800/30 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-400">Loss</span>
                      <span className="text-xs text-emerald-400 font-mono">
                        {metricsHistory.loss.length > 0 ? metricsHistory.loss[metricsHistory.loss.length - 1].toFixed(2) : '—'}
                      </span>
                    </div>
                    <div className="h-16">
                      {metricsHistory.loss.length > 1 ? (
                        <svg viewBox="0 0 100 40" className="w-full h-full" preserveAspectRatio="none">
                          {/* Subtle grid lines */}
                          <line x1="0" y1="10" x2="100" y2="10" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="20" x2="100" y2="20" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="30" x2="100" y2="30" stroke="#334155" strokeWidth="0.5" />
                          {(() => {
                            const data = metricsHistory.loss;
                            const maxVal = Math.max(...data);
                            const minVal = Math.min(...data);
                            const range = maxVal - minVal || 1;
                            const points = data.map((val, i) => {
                              const x = (i / (data.length - 1)) * 100;
                              const y = 40 - ((val - minVal) / range) * 36 - 2;
                              return `${x},${y}`;
                            }).join(' ');
                            return (
                              <>
                                <polyline
                                  fill="none"
                                  stroke="#10b981"
                                  strokeWidth="1.5"
                                  points={points}
                                />
                                <circle
                                  cx={(data.length - 1) / (data.length - 1) * 100}
                                  cy={40 - ((data[data.length - 1] - minVal) / range) * 36 - 2}
                                  r="2"
                                  fill="#10b981"
                                />
                              </>
                            );
                          })()}
                        </svg>
                      ) : (
                        <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                          Waiting...
                        </div>
                      )}
                    </div>
                  </div>

                  {/* L0 Sparsity Chart */}
                  <div className="bg-slate-800/30 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-400">L0 Sparsity</span>
                      <span className="text-xs text-blue-400 font-mono">
                        {metricsHistory.l0_sparsity.length > 0
                          ? (metricsHistory.l0_sparsity[metricsHistory.l0_sparsity.length - 1] * 100).toFixed(1) + '%'
                          : '—'}
                      </span>
                    </div>
                    <div className="h-16">
                      {metricsHistory.l0_sparsity.length > 1 ? (
                        <svg viewBox="0 0 100 40" className="w-full h-full" preserveAspectRatio="none">
                          {/* Subtle grid lines */}
                          <line x1="0" y1="10" x2="100" y2="10" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="20" x2="100" y2="20" stroke="#334155" strokeWidth="0.5" />
                          <line x1="0" y1="30" x2="100" y2="30" stroke="#334155" strokeWidth="0.5" />
                          {(() => {
                            const data = metricsHistory.l0_sparsity;
                            const maxVal = Math.max(...data);
                            const minVal = Math.min(...data);
                            const range = maxVal - minVal || 1;
                            const points = data.map((val, i) => {
                              const x = (i / (data.length - 1)) * 100;
                              const y = 40 - ((val - minVal) / range) * 36 - 2;
                              return `${x},${y}`;
                            }).join(' ');
                            return (
                              <>
                                <polyline
                                  fill="none"
                                  stroke="#3b82f6"
                                  strokeWidth="1.5"
                                  points={points}
                                />
                                <circle
                                  cx={(data.length - 1) / (data.length - 1) * 100}
                                  cy={40 - ((data[data.length - 1] - minVal) / range) * 36 - 2}
                                  r="2"
                                  fill="#3b82f6"
                                />
                              </>
                            );
                          })()}
                        </svg>
                      ) : (
                        <div className="h-full flex items-center justify-center text-slate-500 text-xs">
                          Waiting...
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Dead Neurons Chart */}
                  <div className="bg-slate-800/30 rounded-lg p-2">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium text-slate-400">Dead Neurons</span>
                      <span className="text-xs text-red-400 font-mono">
                        {Math.floor(deadNeurons).toLocaleString()}/{training.hyperparameters?.latent_dim?.toLocaleString() || '?'}
                      </span>
                    </div>
                    <div className="h-10 flex items-center">
                      {/* Dead neurons bar */}
                      <div className="flex-1 h-3 bg-slate-700 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-red-500 transition-all"
                          style={{
                            width: `${training.hyperparameters?.latent_dim
                              ? Math.min(100, (deadNeurons / training.hyperparameters.latent_dim) * 100)
                              : 0}%`
                          }}
                        />
                      </div>
                      <span className="text-xs text-slate-500 ml-2 w-12 text-right">
                        {training.hyperparameters?.latent_dim
                          ? ((deadNeurons / training.hyperparameters.latent_dim) * 100).toFixed(1) + '%'
                          : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Control Buttons */}
      {training.status !== TrainingStatus.COMPLETED && (
        <div className="border-t border-slate-700 pt-3 flex gap-2">
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
                className={`flex-1 flex items-center justify-center gap-2 rounded-lg ${COMPONENTS.button.primary}`}
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
      {showHyperparameters && training.hyperparameters && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-slate-900 border border-slate-700 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[80vh] overflow-auto">
            {/* Modal Header */}
            <div className="sticky top-0 bg-slate-900 border-b border-slate-700 px-4 py-3 flex items-center justify-between">
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
            <div className="px-4 py-3 space-y-4">
              {/* SAE Architecture Section */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-2">SAE Architecture</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Architecture Type</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.architecture_type}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Training Layers</div>
                    <div className="text-sm text-slate-100 font-medium font-mono">
                      {training.hyperparameters.training_layers && training.hyperparameters.training_layers.length > 0
                        ? training.hyperparameters.training_layers.map(l => `L${l}`).join(', ')
                        : 'L0'}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Hook Types</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.hook_types && training.hyperparameters.hook_types.length > 0
                        ? training.hyperparameters.hook_types.join(', ')
                        : 'residual'}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Hidden Dimension</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.hidden_dim.toLocaleString()}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Latent Dimension</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.latent_dim.toLocaleString()}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Expansion Ratio</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {(training.hyperparameters.latent_dim / training.hyperparameters.hidden_dim).toFixed(1)}x
                    </div>
                  </div>
                </div>
              </div>

              {/* Sparsity Section */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-2">Sparsity Configuration</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">L1 Alpha</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.l1_alpha}
                    </div>
                  </div>
                  {training.hyperparameters.target_l0 && (
                    <div className="bg-slate-800/50 rounded-lg p-2">
                      <div className="text-xs text-slate-400 mb-1">Target L0</div>
                      <div className="text-sm text-slate-100 font-medium">
                        {training.hyperparameters.target_l0}
                      </div>
                    </div>
                  )}
                  {training.hyperparameters.top_k_sparsity && (
                    <div className="bg-slate-800/50 rounded-lg p-2 border border-emerald-500/30">
                      <div className="text-xs text-slate-400 mb-1">Top-K Sparsity</div>
                      <div className="text-sm text-emerald-400 font-medium">
                        {training.hyperparameters.top_k_sparsity}%
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Training Configuration Section */}
              <div>
                <h4 className="text-sm font-semibold text-emerald-400 mb-2">Training Configuration</h4>
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Learning Rate</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.learning_rate}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Batch Size</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.batch_size}
                    </div>
                  </div>
                  <div className="bg-slate-800/50 rounded-lg p-2">
                    <div className="text-xs text-slate-400 mb-1">Total Steps</div>
                    <div className="text-sm text-slate-100 font-medium">
                      {training.hyperparameters.total_steps.toLocaleString()}
                    </div>
                  </div>
                  {training.hyperparameters.warmup_steps && (
                    <div className="bg-slate-800/50 rounded-lg p-2">
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
                  <h4 className="text-sm font-semibold text-emerald-400 mb-2">Optimization</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {training.hyperparameters.weight_decay && (
                      <div className="bg-slate-800/50 rounded-lg p-2">
                        <div className="text-xs text-slate-400 mb-1">Weight Decay</div>
                        <div className="text-sm text-slate-100 font-medium">
                          {training.hyperparameters.weight_decay}
                        </div>
                      </div>
                    )}
                    {training.hyperparameters.grad_clip_norm && (
                      <div className="bg-slate-800/50 rounded-lg p-2">
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
                  <h4 className="text-sm font-semibold text-emerald-400 mb-2">Checkpointing & Logging</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {training.hyperparameters.checkpoint_interval && (
                      <div className="bg-slate-800/50 rounded-lg p-2">
                        <div className="text-xs text-slate-400 mb-1">Checkpoint Interval</div>
                        <div className="text-sm text-slate-100 font-medium">
                          Every {training.hyperparameters.checkpoint_interval.toLocaleString()} steps
                        </div>
                      </div>
                    )}
                    {training.hyperparameters.log_interval && (
                      <div className="bg-slate-800/50 rounded-lg p-2">
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
                  <h4 className="text-sm font-semibold text-emerald-400 mb-2">Dead Neuron Handling</h4>
                  <div className="grid grid-cols-2 gap-3">
                    {training.hyperparameters.dead_neuron_threshold !== undefined && (
                      <div className="bg-slate-800/50 rounded-lg p-2">
                        <div className="text-xs text-slate-400 mb-1">Dead Neuron Threshold</div>
                        <div className="text-sm text-slate-100 font-medium">
                          {training.hyperparameters.dead_neuron_threshold.toLocaleString()} steps
                        </div>
                      </div>
                    )}
                    {training.hyperparameters.resample_dead_neurons !== undefined && (
                      <div className="bg-slate-800/50 rounded-lg p-2">
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
            <div className="sticky bottom-0 bg-slate-900 border-t border-slate-700 px-4 py-3">
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
        <div className="bg-red-900/20 border border-red-900/50 rounded-lg p-3">
          <div className="text-sm font-medium text-red-400 mb-1">Training Error</div>
          <div className="text-xs text-red-300">{training.error_message}</div>
        </div>
      )}

      {/* SAE Import Modal */}
      <SAEImportModal
        training={training}
        isOpen={showSAEImportModal}
        onClose={() => setShowSAEImportModal(false)}
        onImportComplete={handleImportComplete}
        modelName={modelName}
        datasetName={datasetName}
      />
    </div>
  );
};

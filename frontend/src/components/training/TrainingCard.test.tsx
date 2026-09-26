/**
 * Unit tests for TrainingCard component.
 *
 * Tests rendering, status display, control actions, expandable sections,
 * checkpoint management, and real-time metrics updates.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrainingCard } from './TrainingCard';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { TrainingStatus, SAEArchitectureType } from '../../types/training';
import type { Training } from '../../types/training';
import type { Model } from '../../types/model';
import type { Dataset } from '../../types/dataset';

// Mock the trainings store
vi.mock('../../stores/trainingsStore');

// Mock child components
vi.mock('../training/LiveMetrics', () => ({
  LiveMetrics: ({ training }: { training: any }) => (
    <div data-testid="live-metrics">Live Metrics for {training.id}</div>
  ),
}));

vi.mock('../training/CheckpointManagement', () => ({
  CheckpointManagement: ({ trainingId }: { trainingId: string }) => (
    <div data-testid="checkpoint-management">Checkpoints for {trainingId}</div>
  ),
}));

// Mock window.confirm
global.confirm = vi.fn(() => true);

describe('TrainingCard', () => {
  const mockPauseTraining = vi.fn();
  const mockResumeTraining = vi.fn();
  const mockStopTraining = vi.fn();
  const mockRetryTraining = vi.fn();
  const mockFetchCheckpoints = vi.fn();
  const mockSaveCheckpoint = vi.fn();
  const mockDeleteCheckpoint = vi.fn();
const mockStopAndFinalizeTraining = vi.fn();
const mockFinalizeTraining = vi.fn();
const mockEvaluateTraining = vi.fn();
const mockFetchTraining = vi.fn();

  const mockModels: Model[] = [
    {
      id: 'm_model1',
      name: 'GPT-2 Small',
      repo_id: 'gpt2',
      architecture: 'gpt2',
      params_count: 124000000,
      quantization: 'int8',
      status: 'ready',
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    } as Model,
  ];

  const mockDatasets: Dataset[] = [
    {
      id: 'ds_dataset1',
      name: 'TinyStories',
      source: 'huggingface',
      repo_id: 'roneneldan/TinyStories',
      split: 'train',
      status: 'ready',
      num_samples: 10000,
      created_at: '2025-01-01T00:00:00Z',
      updated_at: '2025-01-01T00:00:00Z',
    } as Dataset,
  ];

  const baseMockTraining: Training = {
    id: 'train_123',
    model_id: 'm_model1',
    dataset_id: 'ds_dataset1',
    status: TrainingStatus.RUNNING,
    progress: 50,
    current_step: 5000,
    total_steps: 10000,
    current_loss: 0.123,
    current_l0_sparsity: 0.045,
    current_dead_neurons: 15,
    current_learning_rate: 0.0003,
    hyperparameters: {
      hidden_dim: 768,
      latent_dim: 8192,
      architecture_type: SAEArchitectureType.STANDARD_SAELENS,
      l1_alpha: 0.001,
      target_l0: 0.05,
      learning_rate: 0.0003,
      batch_size: 32,
      total_steps: 10000,
      warmup_steps: 1000,
      weight_decay: 0.01,
      grad_clip_norm: 1.0,
      checkpoint_interval: 1000,
      log_interval: 100,
      dead_neuron_threshold: 10000,
      resample_dead_neurons: true,
    },
    created_at: '2025-01-20T10:00:00Z',
    updated_at: '2025-01-20T10:30:00Z',
  };

  const mockOnToggleSelect = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();

    // Set up store mock return values
    mockPauseTraining.mockResolvedValue(undefined);
    mockResumeTraining.mockResolvedValue(undefined);
    mockStopTraining.mockResolvedValue(undefined);
    mockRetryTraining.mockResolvedValue(undefined);
    mockFetchCheckpoints.mockResolvedValue([]);
    mockSaveCheckpoint.mockResolvedValue({ id: 'ckpt_1', step: 5000 });
    mockDeleteCheckpoint.mockResolvedValue(undefined);

    // Mock the store hook
    (useTrainingsStore as any).mockReturnValue({
      pauseTraining: mockPauseTraining,
      resumeTraining: mockResumeTraining,
      stopTraining: mockStopTraining,
      retryTraining: mockRetryTraining,
      fetchCheckpoints: mockFetchCheckpoints,
      saveCheckpoint: mockSaveCheckpoint,
      deleteCheckpoint: mockDeleteCheckpoint,
      stopAndFinalizeTraining: mockStopAndFinalizeTraining,
      finalizeTraining: mockFinalizeTraining,
      evaluateTraining: mockEvaluateTraining,
      fetchTraining: mockFetchTraining,
    });
    mockEvaluateTraining.mockResolvedValue(undefined);
    mockFetchTraining.mockResolvedValue(undefined);
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  /**
   * REACHABILITY for the completion ETA.
   *
   * `utils/trainingEta` has its own unit tests, but a tested util that nothing
   * renders is the "implemented, never wired" pattern this repo keeps finding.
   * These assert the badge in the rendered card; deleting the JSX block turns
   * the first one red.
   *
   * NOTE the timestamps. `baseMockTraining` has no `started_at` and an
   * `updated_at` from 2025, so the ETA correctly does NOT render on it — which
   * is why every other test in this file passes without knowing about it, and
   * why these set both fields deliberately rather than inheriting them.
   */
  describe('Completion ETA', () => {
    const liveRun: Training = {
      ...baseMockTraining,
      status: TrainingStatus.RUNNING,
      current_step: 5000,
      total_steps: 10000,
      started_at: new Date(Date.now() - 50 * 60_000).toISOString(),
      updated_at: new Date(Date.now() - 20_000).toISOString(),
    };

    const renderCard = (training: Training) =>
      render(
        <TrainingCard
          training={training}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

    it('shows an estimated completion clock on a live run', () => {
      renderCard(liveRun);

      const eta = screen.getByTestId('training-eta');
      expect(eta).toBeInTheDocument();
      expect(eta.textContent).toMatch(/^ETA /);
      // The clock is rendered by toLocaleTimeString, i.e. the viewer's own zone.
      expect(eta.textContent).toMatch(/\d/);
    });

    it('explains how the estimate was made, and that it is local time', () => {
      renderCard(liveRun);

      const title = screen.getByTestId('training-eta').getAttribute('title') ?? '';
      expect(title).toContain('remaining');
      expect(title).toContain('steps/min');
      expect(title).toContain('local timezone');
    });

    it('shows no ETA once the run has finished', () => {
      renderCard({
        ...liveRun,
        status: TrainingStatus.COMPLETED,
        completed_at: new Date().toISOString(),
      });

      expect(screen.queryByTestId('training-eta')).not.toBeInTheDocument();
    });

    it('shows no ETA for a paused run — it is not progressing', () => {
      renderCard({ ...liveRun, status: TrainingStatus.PAUSED });

      expect(screen.queryByTestId('training-eta')).not.toBeInTheDocument();
    });

    it('shows no ETA when the heartbeat has gone stale', () => {
      renderCard({ ...liveRun, updated_at: new Date(Date.now() - 30 * 60_000).toISOString() });

      expect(screen.queryByTestId('training-eta')).not.toBeInTheDocument();
    });
  });

  describe('Basic Rendering', () => {
    it('should render training card with basic information', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('GPT-2 Small')).toBeInTheDocument();
      expect(screen.getByText('TinyStories')).toBeInTheDocument();
      // Progress is shown as a percentage next to the progress bar (the
      // redesigned card no longer renders a "Training Progress" label).
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    it('should display training ID', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // ID is truncated to first 8 characters: "train_12" from "train_123"
      expect(screen.getByText(/Training train_12/)).toBeInTheDocument();
    });

    it('should display architecture type', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText(/Standard/i)).toBeInTheDocument();
    });

    it('should use model ID as fallback when model not found', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={[]}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('m_model1')).toBeInTheDocument();
    });

    it('should use dataset ID as fallback when dataset not found', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={[]}
        />
      );

      expect(screen.getByText('ds_dataset1')).toBeInTheDocument();
    });
  });

  describe('Status Display', () => {
    it('should display RUNNING status correctly', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // Status is rendered lowercase (CSS capitalize doesn't apply in tests)
      expect(screen.getByText('running')).toBeInTheDocument();
    });

    it('should display COMPLETED status correctly', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            status: TrainingStatus.COMPLETED,
            progress: 100,
            completed_at: '2025-01-20T11:00:00Z',
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('completed')).toBeInTheDocument();
    });

    it('should display PAUSED status correctly', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.PAUSED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('paused')).toBeInTheDocument();
    });

    it('should display FAILED status correctly', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            status: TrainingStatus.FAILED,
            error_message: 'Out of memory',
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('failed')).toBeInTheDocument();
      expect(screen.getByText('Out of memory')).toBeInTheDocument();
    });

    it('should display CANCELLED status correctly', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.CANCELLED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('cancelled')).toBeInTheDocument();
    });
  });

  describe('Progress Display', () => {
    it('should display progress bar with correct percentage', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const progressBar = screen.getByText('50.0%').closest('div');
      expect(progressBar).toBeInTheDocument();
    });

    it('should display progress information', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // Check that progress percentage is displayed
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    it('should display metrics when progress > 10%', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // Loss: 0.123 → formatted as "0.123000" with toFixed(6)
      expect(screen.getByText('0.123000')).toBeInTheDocument();
      // L0 Sparsity: 0.045 * 8192 = ~369 (absolute count via formatL0Absolute)
      expect(screen.getByText('~369')).toBeInTheDocument();
      // Check inline metric labels (redesigned card uses short labels).
      expect(screen.getByText('Loss')).toBeInTheDocument();
      expect(screen.getByText('L0')).toBeInTheDocument();
    });

    it("labels the activity-estimate count as inactive latents, with both windows, not as the resampler's dead", () => {
      // R1-D L2: the count is an EMA that crosses its threshold after ~84 silent steps
      // at batch 4,096, while resampling waits dead_neuron_threshold steps. It was
      // labelled "Dead" beside the control that sets the resampler's threshold.
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            hyperparameters: { ...baseMockTraining.hyperparameters, batch_size: 4096, dead_neuron_threshold: 1000 },
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const label = screen.getByText('Inactive');
      expect(label.getAttribute('title')).toContain('about the last 84 steps');
      expect(label.getAttribute('title')).toContain('1,000 consecutive silent steps');
      expect(screen.queryByText('Dead')).not.toBeInTheDocument();
    });

    it('should not display metrics when progress <= 10%', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, progress: 5 }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.queryByText(/Loss:/)).not.toBeInTheDocument();
    });
  });

  describe('Selection Checkbox', () => {
    it('should render unchecked checkbox when not selected', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).not.toBeChecked();
    });

    it('should render checked checkbox when selected', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={true}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const checkbox = screen.getByRole('checkbox');
      expect(checkbox).toBeChecked();
    });

    it('should call onToggleSelect when checkbox is clicked', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const checkbox = screen.getByRole('checkbox');
      fireEvent.click(checkbox);

      expect(mockOnToggleSelect).toHaveBeenCalledWith('train_123');
    });
  });

  describe('Control Buttons', () => {
    describe('Pause Button', () => {
      it('should display pause button when training is RUNNING', () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.getByRole('button', { name: /pause/i })).toBeInTheDocument();
      });

      it('should call pauseTraining when pause button is clicked', async () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const pauseButton = screen.getByRole('button', { name: /pause/i });
        fireEvent.click(pauseButton);

        await waitFor(() => {
          expect(mockPauseTraining).toHaveBeenCalledWith('train_123');
        });
      });
    });

    describe('Resume Button', () => {
      it('should display resume button when training is PAUSED', () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.PAUSED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.getByRole('button', { name: /resume/i })).toBeInTheDocument();
      });

      it('should call resumeTraining when resume button is clicked', async () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.PAUSED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const resumeButton = screen.getByRole('button', { name: /resume/i });
        fireEvent.click(resumeButton);

        await waitFor(() => {
          expect(mockResumeTraining).toHaveBeenCalledWith('train_123');
        });
      });

      // A CRASHED RUN IS RESUMABLE. A worker that dies leaves the row FAILED with
      // its checkpoints intact; the resume path reads the newest COMPLETE one and
      // continues at the step after it. Before this the only options were Finalize
      // (keep the SAE as it stood, train no further) and Retry (start again at step
      // 0), so a crash at step 120,000 of 150,000 discarded every step.
      it('offers Resume on a FAILED run that has checkpoints', async () => {
        mockFetchCheckpoints.mockResolvedValue([{ id: 'ckpt_1', step: 1000 }]);
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.FAILED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        await waitFor(() => {
          expect(screen.getByRole('button', { name: /resume/i })).toBeInTheDocument();
        });
      });

      it('resumes a FAILED run through the same control a pause uses', async () => {
        mockFetchCheckpoints.mockResolvedValue([{ id: 'ckpt_1', step: 1000 }]);
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.FAILED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const button = await screen.findByRole('button', { name: /resume/i });
        fireEvent.click(button);

        await waitFor(() => {
          expect(mockResumeTraining).toHaveBeenCalledWith('train_123');
        });
      });

      it('offers no Resume on a FAILED run with nothing to resume from', async () => {
        mockFetchCheckpoints.mockResolvedValue([]);
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.FAILED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        // Retry is offered instead, so the card is not actionless.
        await waitFor(() => {
          expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
        });
        expect(screen.queryByRole('button', { name: /resume/i })).not.toBeInTheDocument();
      });

      it('offers no Resume on a CANCELLED run, which was stopped on purpose', async () => {
        mockFetchCheckpoints.mockResolvedValue([{ id: 'ckpt_1', step: 1000 }]);
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.CANCELLED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        // Finalize IS offered on a cancelled run with checkpoints — so this asserts
        // the distinction, not merely that the card rendered nothing.
        await waitFor(() => {
          expect(screen.getByRole('button', { name: /finalize/i })).toBeInTheDocument();
        });
        expect(screen.queryByRole('button', { name: /resume/i })).not.toBeInTheDocument();
      });
    });

    describe('Stop Button', () => {
      it('should display stop button when training is RUNNING or PAUSED', () => {
        const { rerender } = render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.getByRole('button', { name: /^stop training$/i })).toBeInTheDocument();

        rerender(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.PAUSED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.getByRole('button', { name: /^stop training$/i })).toBeInTheDocument();
      });

      it('should call stopTraining when stop button is clicked', async () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const stopButton = screen.getByRole('button', { name: /^stop training$/i });
        fireEvent.click(stopButton);

        await waitFor(() => {
          expect(mockStopTraining).toHaveBeenCalledWith('train_123');
        });
      });
    });

    describe('Retry Button', () => {
      it('should display retry button when training is FAILED', () => {
        render(
          <TrainingCard
            training={{
              ...baseMockTraining,
              status: TrainingStatus.FAILED,
              error_message: 'OOM error',
            }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.getByText('Retry')).toBeInTheDocument();
      });

      it('should call retryTraining when retry button is clicked', async () => {
        render(
          <TrainingCard
            training={{
              ...baseMockTraining,
              status: TrainingStatus.FAILED,
              error_message: 'OOM error',
            }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const retryButton = screen.getByText('Retry');
        fireEvent.click(retryButton);

        await waitFor(() => {
          expect(mockRetryTraining).toHaveBeenCalledWith('train_123');
        });
      });
    });
  });

  describe('Expandable Sections', () => {
    describe('Metrics Section', () => {
      it('should not show metrics initially', () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.queryByText('Training Logs')).not.toBeInTheDocument();
      });

      it('should toggle metrics section when button is clicked', () => {
        render(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const metricsButton = screen.getByRole('button', { name: /show live metrics/i });
        fireEvent.click(metricsButton);

        expect(screen.getByText('Training Logs')).toBeInTheDocument();

        // Click again to hide
        const hideButton = screen.getByRole('button', { name: /hide live metrics/i });
        fireEvent.click(hideButton);
        expect(screen.queryByText('Training Logs')).not.toBeInTheDocument();
      });
    });

    describe('Checkpoints Section', () => {
      it('should not show checkpoints initially', () => {
        render(
          <TrainingCard
            training={baseMockTraining}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.queryByText('Checkpoint Management')).not.toBeInTheDocument();
      });

      it('should toggle checkpoints section when button is clicked', () => {
        render(
          <TrainingCard
            training={baseMockTraining}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const checkpointsButton = screen.getByRole('button', { name: /checkpoints/i });
        fireEvent.click(checkpointsButton);

        expect(screen.getByText('Checkpoint Management')).toBeInTheDocument();

        // Click again to hide
        fireEvent.click(checkpointsButton);
        expect(screen.queryByText('Checkpoint Management')).not.toBeInTheDocument();
      });
    });

    describe('Hyperparameters Section', () => {
      it('should not show hyperparameters modal initially', () => {
        render(
          <TrainingCard
            training={baseMockTraining}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.queryByText('Training Hyperparameters')).not.toBeInTheDocument();
      });

      it('should toggle hyperparameters modal when icon is clicked', () => {
        render(
          <TrainingCard
            training={baseMockTraining}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        const hyperparamsButton = screen.getByTitle('View all hyperparameters');
        fireEvent.click(hyperparamsButton);

        expect(screen.getByText('Training Hyperparameters')).toBeInTheDocument();
        expect(screen.getByText('SAE Architecture')).toBeInTheDocument();
      });
    });
  });

  describe('Time Display', () => {
    it('should display started time when available', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, started_at: '2025-01-20T10:00:00Z' }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // The redesigned header shows the started time via toLocaleTimeString()
      // (no "Started:" label). Assert the formatted time is rendered.
      const expectedTime = new Date('2025-01-20T10:00:00Z').toLocaleTimeString();
      expect(screen.getByText(expectedTime)).toBeInTheDocument();
    });

    it('should display completion time for completed trainings', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            status: TrainingStatus.COMPLETED,
            started_at: '2025-01-20T10:00:00Z',
            completed_at: '2025-01-20T11:00:00Z',
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // For completed trainings the header renders the derived duration
      // (start→completed) rather than a "Completed:" label. 10:00→11:00 = 1h.
      expect(screen.getByText(/1h 0m/)).toBeInTheDocument();
    });

    it('should calculate and display duration for completed trainings', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            status: TrainingStatus.COMPLETED,
            started_at: '2025-01-20T10:00:00Z',
            completed_at: '2025-01-20T11:30:00Z',
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // Duration is rendered in the header without a "Duration:" label.
      expect(screen.getByText(/1h 30m/)).toBeInTheDocument();
    });
  });

  describe('Finalize (Feature 021)', () => {
    it('should render Stop & Finalize alongside Stop while RUNNING', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      expect(
        screen.getByRole('button', { name: /stop and finalize training/i })
      ).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /^stop training$/i })).toBeInTheDocument();
    });

    it('should dispatch stopAndFinalizeTraining when clicked', async () => {
      mockStopAndFinalizeTraining.mockResolvedValueOnce({});
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      fireEvent.click(screen.getByRole('button', { name: /stop and finalize training/i }));
      await waitFor(() => {
        expect(mockStopAndFinalizeTraining).toHaveBeenCalledWith(baseMockTraining.id);
      });
    });

    it('should offer Finalize on a cancelled run that HAS checkpoints', async () => {
      // The rescue path: runs stopped before this feature existed have intact
      // checkpoints but no community_format. Without overriding the default
      // empty mock, checkpoints.length > 0 is false in every test and the whole
      // button block can be deleted with the suite still green.
      mockFetchCheckpoints.mockResolvedValueOnce([
        { id: 'ckpt_1', step: 10000, loss: 0.29, is_best: false },
      ]);
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.CANCELLED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      const btn = await screen.findByRole('button', {
        name: /finalize training from checkpoint/i,
      });
      fireEvent.click(btn);
      await waitFor(() => {
        expect(mockFinalizeTraining).toHaveBeenCalledWith(baseMockTraining.id);
      });
      expect(mockFinalizeTraining).toHaveBeenCalledTimes(1);
    });

    it('should offer Finalize on a FAILED run that has checkpoints', async () => {
      // A crashed run previously offered only Retry (which restarts from step 0),
      // stranding perfectly good checkpoints.
      mockFetchCheckpoints.mockResolvedValueOnce([
        { id: 'ckpt_1', step: 3000, loss: 0.4, is_best: false },
      ]);
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.FAILED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      expect(
        await screen.findByRole('button', {
          name: /finalize training from checkpoint/i,
        })
      ).toBeInTheDocument();
    });

    it('should escalate with allow_failed when the API 409s a FAILED run', async () => {
      // The 409 body says "Re-send with allow_failed=true" — the UI must be able
      // to actually do that, or the message is unfollowable.
      mockFetchCheckpoints.mockResolvedValueOnce([
        { id: 'ckpt_1', step: 3000, loss: 0.4, is_best: false },
      ]);
      mockFinalizeTraining
        .mockRejectedValueOnce({
          response: { status: 409, data: { detail: 'run FAILED' } },
        })
        .mockResolvedValueOnce(undefined);
      vi.spyOn(window, 'confirm').mockReturnValue(true);

      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.FAILED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      fireEvent.click(
        await screen.findByRole('button', {
          name: /finalize training from checkpoint/i,
        })
      );

      await waitFor(() => {
        expect(mockFinalizeTraining).toHaveBeenCalledTimes(2);
      });
      // Payload AND call count: "was called" would pass on the wrong flags.
      expect(mockFinalizeTraining).toHaveBeenLastCalledWith(
        baseMockTraining.id,
        undefined,
        { allowFailed: true, force: false }
      );
    });

    it('should NOT offer Finalize on a cancelled run with no checkpoints', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.CANCELLED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      expect(
        screen.queryByRole('button', { name: /finalize training from checkpoint/i })
      ).not.toBeInTheDocument();
    });

    it('should show the finalized-early badge instead of implying a full run', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            status: TrainingStatus.COMPLETED,
            finalized_from_step: 10000,
            total_steps: 50000,
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      // status is 'completed' so the SAE imports, but the card must say the run
      // stopped early rather than presenting it as a finished 50k-step run.
      expect(screen.getByText(/finalized early/i)).toBeInTheDocument();
    });

    it('should not show the badge for a normally completed run', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.COMPLETED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      expect(screen.queryByText(/finalized early/i)).not.toBeInTheDocument();
    });
  });

  describe('FVU labels (SAE training remediation, item 5)', () => {
    const openMetrics = () =>
      fireEvent.click(screen.getByRole('button', { name: /show live metrics/i }));

    it('shows the centred FVU as "FVU", with the legacy value alongside', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, current_fvu_centred: 0.3204, current_fvu: 0.2611 }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      openMetrics();

      expect(screen.getByTestId('fvu-label')).toHaveTextContent(/^FVU$/);
      const value = screen.getByTestId('fvu-value');
      expect(value).toHaveTextContent('0.3204');
      expect(value).toHaveTextContent('legacy 0.2611');
    });

    it('labels a run recorded before the change as legacy', () => {
      render(
        <TrainingCard
          training={{ ...baseMockTraining, current_fvu_centred: null, current_fvu: 0.2611 }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );
      openMetrics();

      expect(screen.getByTestId('fvu-label')).toHaveTextContent('FVU (legacy)');
      expect(screen.getByTestId('fvu-value')).toHaveTextContent('0.2611');
      expect(screen.getByTestId('fvu-value')).not.toHaveTextContent('legacy 0.2611');
    });
  });

  describe('Evaluation panel (SAE training remediation, item 6)', () => {
    const completed = {
      ...baseMockTraining,
      status: TrainingStatus.COMPLETED,
      progress: 100,
      completed_at: '2025-01-20T11:00:00Z',
    };

    it('renders on a completed training and sends Evaluate to the store', async () => {
      render(
        <TrainingCard
          training={completed}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByTestId('training-evaluation')).toBeInTheDocument();
      fireEvent.click(screen.getByRole('button', { name: /^Evaluate$/ }));
      await waitFor(() => expect(mockEvaluateTraining).toHaveBeenCalledTimes(1));
      expect(mockEvaluateTraining).toHaveBeenCalledWith('train_123');
    });

    it('sends a forced re-run of a stale evaluation to the store with force (review R1-C, UI15)', async () => {
      /* The panel's own tests pass their own onEvaluate, so a card that dropped the
       * options on the way to the store survived every test.
       * NEGATIVE CONTROL: `(id, opts) => evaluateTraining(id)` in the card -> RED here. */
      vi.useFakeTimers({ shouldAdvanceTime: true });
      vi.setSystemTime(Date.parse('2026-09-15T12:00:00Z'));
      try {
        render(
          <TrainingCard
            training={{
              ...completed,
              evaluation: { status: 'running', task_id: 'abc', updated_at: '2026-09-15T11:00:00Z' },
            }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );
        fireEvent.click(screen.getByTestId('evaluation-force'));
        await waitFor(() => expect(mockEvaluateTraining).toHaveBeenCalledTimes(1));
        expect(mockEvaluateTraining).toHaveBeenCalledWith('train_123', { force: true });
      } finally {
        vi.useRealTimers();
      }
    });

    it('shows a recorded result', () => {
      render(
        <TrainingCard
          training={{
            ...completed,
            evaluation: {
              status: 'completed',
              ce_base: 2.1,
              tokens: 1000,
              layers: [{
                layer: 11, ce_spliced: 2.4, ce_mean_ablated: 3.5, ce_zero_ablated: 11.09, ce_delta: 0.3,
                loss_recovered_vs_mean: 0.7857, loss_recovered_vs_zero: 0.97, kl: 0.28, l0: 65, fvu_centred: 0.32,
              }],
            },
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByTestId('evaluation-layer-11')).toHaveTextContent('78.6%');
    });

    it('does not render while the training is still running', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.queryByTestId('training-evaluation')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should handle pause error gracefully', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      mockPauseTraining.mockRejectedValue(new Error('Network error'));

      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const pauseButton = screen.getByRole('button', { name: /pause/i });
      fireEvent.click(pauseButton);

      await waitFor(() => {
        expect(consoleErrorSpy).toHaveBeenCalled();
      });

      consoleErrorSpy.mockRestore();
    });

    it('should handle resume error gracefully', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      mockResumeTraining.mockRejectedValue(new Error('Network error'));

      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.PAUSED }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const resumeButton = screen.getByRole('button', { name: /resume/i });
      fireEvent.click(resumeButton);

      await waitFor(() => {
        expect(consoleErrorSpy).toHaveBeenCalled();
      });

      consoleErrorSpy.mockRestore();
    });

    it('should handle stop error gracefully', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      mockStopTraining.mockRejectedValue(new Error('Network error'));

      render(
        <TrainingCard
          training={{ ...baseMockTraining, status: TrainingStatus.RUNNING }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      const stopButton = screen.getByRole('button', { name: /^stop training$/i });
      fireEvent.click(stopButton);

      await waitFor(() => {
        expect(consoleErrorSpy).toHaveBeenCalled();
      });

      consoleErrorSpy.mockRestore();
    });
  });

  describe('Edge Cases', () => {
    it('should handle training with no metrics', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            progress: 50, // Need progress > 10% to show metrics section
            current_loss: null,
            current_l0_sparsity: null,
            current_dead_neurons: null,
            current_learning_rate: null,
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      // Component should still render without errors
      expect(screen.getByText('50.0%')).toBeInTheDocument();
    });

    it('should handle training with 0% progress', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            progress: 0,
            current_step: 0,
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('0.0%')).toBeInTheDocument();
    });

    it('should handle training with 100% progress', () => {
      render(
        <TrainingCard
          training={{
            ...baseMockTraining,
            progress: 100,
            current_step: 10000,
            status: TrainingStatus.COMPLETED,
          }}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      expect(screen.getByText('100.0%')).toBeInTheDocument();
    });

    it('should handle empty models and datasets arrays', () => {
      render(
        <TrainingCard
          training={baseMockTraining}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={[]}
          datasets={[]}
        />
      );

      // Should display IDs as fallback
      expect(screen.getByText('m_model1')).toBeInTheDocument();
      expect(screen.getByText('ds_dataset1')).toBeInTheDocument();
    });
  });

  describe('L0 is shown as a feature count, never a raw fraction', () => {
    /**
     * train_9355afa6, 2026-08-26: a healthy JumpReLU SAE (FVU 0.092, 0 dead
     * neurons, ~7 of 30,720 latents active) rendered "L0: 0.000" in the
     * checkpoint list, because the trainer stores L0 as a FRACTION --
     * `(z > 0).float().mean()` = 0.000228 -- and this list printed it with
     * toFixed(3). A good run reads as a collapsed dictionary, which is how a
     * usable SAE gets discarded. The main card row was already correct.
     */
    const REAL_L0_FRACTION = 0.00022773744422011077;

    const completed = () => ({
      ...baseMockTraining,
      status: TrainingStatus.COMPLETED,
      current_l0_sparsity: REAL_L0_FRACTION,
      hyperparameters: {
        ...baseMockTraining.hyperparameters,
        latent_dim: 30720,
      },
    });

    it('renders the checkpoint L0 as a count, not 0.000', async () => {
      mockFetchCheckpoints.mockResolvedValueOnce([
        { id: 'ckpt_1', step: 48000, loss: 0.0182,
          l0_sparsity: REAL_L0_FRACTION, is_best: true,
          created_at: '2026-08-26T17:15:00Z' },
      ]);

      render(
        <TrainingCard
          training={completed() as never}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      fireEvent.click(await screen.findByRole('button', { name: /checkpoints \(1\)/i }));

      const line = await screen.findByText(/Loss: 0\.0182/);
      expect(line.textContent).toMatch(/L0: ~7/);
      expect(line.textContent).not.toMatch(/L0: 0\.000/);
    });

    it('does not hide the L0 label when a checkpoint is genuinely dead', async () => {
      // `cp.l0_sparsity && ...` treated 0 as absent, so a dead checkpoint
      // showed no L0 at all rather than saying zero.
      mockFetchCheckpoints.mockResolvedValueOnce([
        { id: 'ckpt_1', step: 48000, loss: 0.5, l0_sparsity: 0,
          is_best: false, created_at: '2026-08-26T17:15:00Z' },
      ]);

      render(
        <TrainingCard
          training={completed() as never}
          isSelected={false}
          onToggleSelect={mockOnToggleSelect}
          models={mockModels}
          datasets={mockDatasets}
        />
      );

      fireEvent.click(await screen.findByRole('button', { name: /checkpoints \(1\)/i }));
      const line = await screen.findByText(/Loss: 0\.5000/);
      expect(line.textContent).toMatch(/L0: 0/);
    });
  });
});

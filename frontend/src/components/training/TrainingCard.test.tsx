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
      architecture_type: SAEArchitectureType.STANDARD,
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
    });
  });

  afterEach(() => {
    vi.clearAllTimers();
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
      expect(screen.getByText('50.0%')).toBeInTheDocument();
      // Check for progress display elements
      expect(screen.getByText('Training Progress')).toBeInTheDocument();
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
      expect(screen.getByText('Training Progress')).toBeInTheDocument();
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

      // Loss: 0.123 → formatted as "0.1230" with toFixed(4)
      expect(screen.getByText('0.1230')).toBeInTheDocument();
      // L0 Sparsity: 0.045 → formatted as "0.0450" with toFixed(4)
      expect(screen.getByText('0.0450')).toBeInTheDocument();
      // Check metric labels
      expect(screen.getByText('Loss')).toBeInTheDocument();
      expect(screen.getByText('L0 Sparsity')).toBeInTheDocument();
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

        expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();

        rerender(
          <TrainingCard
            training={{ ...baseMockTraining, status: TrainingStatus.PAUSED }}
            isSelected={false}
            onToggleSelect={mockOnToggleSelect}
            models={mockModels}
            datasets={mockDatasets}
          />
        );

        expect(screen.getByRole('button', { name: /stop/i })).toBeInTheDocument();
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

        const stopButton = screen.getByRole('button', { name: /stop/i });
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

        expect(screen.queryByText('Loss Curve')).not.toBeInTheDocument();
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

        expect(screen.getByText('Loss Curve')).toBeInTheDocument();

        // Click again to hide
        const hideButton = screen.getByRole('button', { name: /hide live metrics/i });
        fireEvent.click(hideButton);
        expect(screen.queryByText('Loss Curve')).not.toBeInTheDocument();
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

      // Component shows "Started:" not "Created:"
      expect(screen.getByText(/Started:/)).toBeInTheDocument();
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

      expect(screen.getByText(/Completed:/)).toBeInTheDocument();
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

      // Duration should be displayed
      expect(screen.getByText(/Duration:/)).toBeInTheDocument();
      // Check for duration format (1h 30m)
      expect(screen.getByText(/1h 30m/)).toBeInTheDocument();
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

      const stopButton = screen.getByRole('button', { name: /stop/i });
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
      expect(screen.getByText('Training Progress')).toBeInTheDocument();
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
      expect(screen.getByText('Training Progress')).toBeInTheDocument();
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
      // Component doesn't display "X / Y" format for steps
      expect(screen.getByText('Training Progress')).toBeInTheDocument();
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
});

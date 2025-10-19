/**
 * Unit tests for TrainingPanel component.
 *
 * Tests rendering, state management integration, configuration form,
 * training operations, selection/deletion, status filtering, and WebSocket integration.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { TrainingPanel } from './TrainingPanel';
import { useTrainingsStore } from '../../stores/trainingsStore';
import { useModelsStore } from '../../stores/modelsStore';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { useTrainingWebSocket } from '../../hooks/useTrainingWebSocket';
import { useWebSocketContext } from '../../contexts/WebSocketContext';
import { TrainingStatus, SAEArchitectureType } from '../../types/training';

// Mock all stores and hooks
vi.mock('../../stores/trainingsStore');
vi.mock('../../stores/modelsStore');
vi.mock('../../stores/datasetsStore');
vi.mock('../../hooks/useTrainingWebSocket');
vi.mock('../../contexts/WebSocketContext');

// Mock child components
vi.mock('../training/TrainingCard', () => ({
  TrainingCard: ({
    training,
    isSelected,
    onToggleSelect,
  }: {
    training: any;
    isSelected: boolean;
    onToggleSelect: Function;
  }) => (
    <div data-testid={`training-card-${training.id}`}>
      <span>{training.id}</span>
      <span>{training.status}</span>
      <input
        type="checkbox"
        checked={isSelected}
        onChange={() => onToggleSelect(training.id)}
        data-testid={`checkbox-${training.id}`}
      />
    </div>
  ),
}));

// Mock window.confirm
global.confirm = vi.fn(() => true);

describe('TrainingPanel', () => {
  const mockFetchTrainings = vi.fn();
  const mockCreateTraining = vi.fn();
  const mockDeleteTraining = vi.fn();
  const mockUpdateConfig = vi.fn();
  const mockSetStatusFilter = vi.fn();
  const mockFetchModels = vi.fn();
  const mockFetchDatasets = vi.fn();

  const mockConfig = {
    model_id: '',
    dataset_id: '',
    hidden_dim: 768,
    latent_dim: 8192,
    architecture_type: SAEArchitectureType.STANDARD,
    l1_alpha: 0.001,
    target_l0: 0.05,
    learning_rate: 0.0003,
    batch_size: 32,
    total_steps: 100000,
    warmup_steps: 10000,
    weight_decay: 0.01,
    grad_clip_norm: 1.0,
    checkpoint_interval: 1000,
    log_interval: 100,
    dead_neuron_threshold: 10000,
    resample_dead_neurons: true,
  };

  const mockModels = [
    {
      id: 'm_gpt2',
      name: 'GPT-2',
      status: 'ready',
      model_type: 'gpt2',
      config: { vocab_size: 50257, n_positions: 1024, n_embd: 768, n_layer: 12, n_head: 12 },
    },
    {
      id: 'm_bert',
      name: 'BERT Base',
      status: 'ready',
      model_type: 'bert',
      config: { vocab_size: 30522, hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12 },
    },
  ];

  const mockDatasets = [
    {
      id: 'ds_dataset1',
      name: 'Test Dataset 1',
      status: 'ready',
      hf_repo_id: 'test/dataset1',
      source: 'huggingface',
      created_at: '2025-10-19T10:00:00Z',
    },
    {
      id: 'ds_dataset2',
      name: 'Test Dataset 2',
      status: 'ready',
      hf_repo_id: 'test/dataset2',
      source: 'huggingface',
      created_at: '2025-10-19T10:00:00Z',
    },
  ];

  const mockTrainings = [
    {
      id: 'train_123',
      model_id: 'm_gpt2',
      dataset_id: 'ds_dataset1',
      status: TrainingStatus.RUNNING,
      progress: 45.5,
      current_step: 45500,
      total_steps: 100000,
      hyperparameters: {
        hidden_dim: 768,
        latent_dim: 8192,
        architecture_type: 'standard',
        l1_alpha: 0.001,
        learning_rate: 0.0003,
        batch_size: 4096,
        total_steps: 100000,
      },
      created_at: '2025-10-19T10:00:00Z',
      updated_at: '2025-10-19T12:00:00Z',
    },
    {
      id: 'train_456',
      model_id: 'm_bert',
      dataset_id: 'ds_dataset2',
      status: TrainingStatus.COMPLETED,
      progress: 100,
      current_step: 100000,
      total_steps: 100000,
      hyperparameters: {
        hidden_dim: 768,
        latent_dim: 8192,
        architecture_type: 'standard',
        l1_alpha: 0.001,
        learning_rate: 0.0003,
        batch_size: 4096,
        total_steps: 100000,
      },
      created_at: '2025-10-19T09:00:00Z',
      updated_at: '2025-10-19T11:00:00Z',
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();

    // Default trainingsStore mock
    (useTrainingsStore as any).mockReturnValue({
      trainings: [],
      config: mockConfig,
      updateConfig: mockUpdateConfig,
      fetchTrainings: mockFetchTrainings,
      createTraining: mockCreateTraining,
      deleteTraining: mockDeleteTraining,
      statusFilter: 'all',
      setStatusFilter: mockSetStatusFilter,
      isLoading: false,
      error: null,
    });

    // Default modelsStore mock
    (useModelsStore as any).mockReturnValue({
      models: mockModels,
      fetchModels: mockFetchModels,
    });

    // Default datasetsStore mock
    (useDatasetsStore as any).mockReturnValue({
      datasets: mockDatasets,
      fetchDatasets: mockFetchDatasets,
    });

    // Default WebSocket hook mock
    (useTrainingWebSocket as any).mockReturnValue({});

    // Default WebSocket context mock
    (useWebSocketContext as any).mockReturnValue({
      isConnected: true,
    });
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Initial Rendering', () => {
    it('should render the training panel with title', () => {
      render(<TrainingPanel />);
      expect(screen.getByText('SAE Training')).toBeInTheDocument();
      expect(
        screen.getByText('Configure and launch sparse autoencoder training jobs')
      ).toBeInTheDocument();
    });

    it('should fetch models, datasets, and trainings on mount', () => {
      render(<TrainingPanel />);
      expect(mockFetchModels).toHaveBeenCalledTimes(1);
      expect(mockFetchDatasets).toHaveBeenCalledTimes(1);
      expect(mockFetchTrainings).toHaveBeenCalledTimes(1);
    });

    it('should display WebSocket connection status (connected)', () => {
      (useWebSocketContext as any).mockReturnValue({ isConnected: true });
      render(<TrainingPanel />);
      expect(screen.getByText('Live')).toBeInTheDocument();
    });

    it('should display WebSocket connection status (disconnected)', () => {
      (useWebSocketContext as any).mockReturnValue({ isConnected: false });
      render(<TrainingPanel />);
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
    });

    it('should render configuration form', () => {
      render(<TrainingPanel />);
      expect(screen.getByText('Training Configuration')).toBeInTheDocument();
      expect(screen.getByText('Dataset')).toBeInTheDocument();
      expect(screen.getByText('Model')).toBeInTheDocument();
      expect(screen.getByText('SAE Architecture')).toBeInTheDocument();
    });

    it('should render training jobs section', () => {
      render(<TrainingPanel />);
      expect(screen.getByText('Training Jobs')).toBeInTheDocument();
    });
  });

  describe('Configuration Form', () => {
    it('should populate dataset dropdown with ready datasets', () => {
      render(<TrainingPanel />);
      const datasetSelects = screen.getAllByRole('combobox');
      const datasetSelect = datasetSelects[0] as HTMLSelectElement;
      // 1 default option + 2 datasets
      expect(datasetSelect.options.length).toBe(3);
      expect(datasetSelect.options[1].value).toBe('ds_dataset1');
      expect(datasetSelect.options[2].value).toBe('ds_dataset2');
    });

    it('should populate model dropdown with ready models', () => {
      render(<TrainingPanel />);
      const selects = screen.getAllByRole('combobox');
      const modelSelect = selects[1] as HTMLSelectElement;
      // 1 default option + 2 models
      expect(modelSelect.options.length).toBe(3);
      expect(modelSelect.options[1].value).toBe('m_gpt2');
      expect(modelSelect.options[2].value).toBe('m_bert');
    });

    it('should populate architecture dropdown with options', () => {
      render(<TrainingPanel />);
      const selects = screen.getAllByRole('combobox');
      const archSelect = selects[2] as HTMLSelectElement;
      expect(archSelect.options.length).toBe(3);
      expect(archSelect.options[0].value).toBe(SAEArchitectureType.STANDARD);
      expect(archSelect.options[1].value).toBe(SAEArchitectureType.SKIP);
      expect(archSelect.options[2].value).toBe(SAEArchitectureType.TRANSCODER);
    });

    it('should call updateConfig when dataset is selected', () => {
      render(<TrainingPanel />);
      const selects = screen.getAllByRole('combobox');
      const datasetSelect = selects[0];
      fireEvent.change(datasetSelect, { target: { value: 'ds_dataset1' } });
      expect(mockUpdateConfig).toHaveBeenCalledWith({ dataset_id: 'ds_dataset1' });
    });

    it('should call updateConfig when model is selected', () => {
      render(<TrainingPanel />);
      const selects = screen.getAllByRole('combobox');
      const modelSelect = selects[1];
      fireEvent.change(modelSelect, { target: { value: 'm_gpt2' } });
      expect(mockUpdateConfig).toHaveBeenCalledWith({ model_id: 'm_gpt2' });
    });

    it('should call updateConfig when architecture is changed', () => {
      render(<TrainingPanel />);
      const selects = screen.getAllByRole('combobox');
      const archSelect = selects[2];
      fireEvent.change(archSelect, { target: { value: SAEArchitectureType.SKIP } });
      expect(mockUpdateConfig).toHaveBeenCalledWith({
        architecture_type: SAEArchitectureType.SKIP,
      });
    });

    it('should have Start Training button disabled when form is invalid', () => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: [],
        config: { ...mockConfig, model_id: '', dataset_id: '' },
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      render(<TrainingPanel />);
      const startButton = screen.getByRole('button', { name: /Start Training/i });
      expect(startButton).toBeDisabled();
    });

    it('should have Start Training button enabled when form is valid', () => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: [],
        config: { ...mockConfig, model_id: 'm_gpt2', dataset_id: 'ds_dataset1' },
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      render(<TrainingPanel />);
      const startButton = screen.getByRole('button', { name: /Start Training/i });
      expect(startButton).not.toBeDisabled();
    });
  });

  describe('Advanced Configuration', () => {
    it('should not show advanced config by default', () => {
      render(<TrainingPanel />);
      expect(screen.queryByText('Hidden Dimension')).not.toBeInTheDocument();
    });

    it('should toggle advanced config when clicked', async () => {
      render(<TrainingPanel />);
      const toggleButton = screen.getByText('Advanced Configuration');

      // Show advanced
      fireEvent.click(toggleButton);
      await waitFor(() => {
        expect(screen.getByText('Hidden Dimension')).toBeInTheDocument();
      });

      // Hide advanced
      fireEvent.click(toggleButton);
      await waitFor(() => {
        expect(screen.queryByText('Hidden Dimension')).not.toBeInTheDocument();
      });
    });

    it('should update config when advanced fields are changed', async () => {
      render(<TrainingPanel />);
      const toggleButton = screen.getByText('Advanced Configuration');
      fireEvent.click(toggleButton);

      await waitFor(() => {
        expect(screen.getByText('Hidden Dimension')).toBeInTheDocument();
      });

      const numberInputs = screen.getAllByRole('spinbutton');
      const hiddenDimInput = numberInputs.find(
        (input) => input.getAttribute('min') === '64' && input.getAttribute('max') === '8192'
      );
      expect(hiddenDimInput).toBeDefined();
      fireEvent.change(hiddenDimInput!, { target: { value: '1024' } });
      expect(mockUpdateConfig).toHaveBeenCalledWith({ hidden_dim: 1024 });
    });
  });

  describe('Starting Training', () => {
    it('should call createTraining with correct config when Start Training is clicked', async () => {
      const validConfig = {
        ...mockConfig,
        model_id: 'm_gpt2',
        dataset_id: 'ds_dataset1',
      };

      (useTrainingsStore as any).mockReturnValue({
        trainings: [],
        config: validConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      mockCreateTraining.mockResolvedValueOnce({ id: 'train_new' });

      render(<TrainingPanel />);
      const startButton = screen.getByRole('button', { name: /Start Training/i });
      fireEvent.click(startButton);

      await waitFor(() => {
        expect(mockCreateTraining).toHaveBeenCalledWith({
          model_id: 'm_gpt2',
          dataset_id: 'ds_dataset1',
          extraction_id: undefined,
          hyperparameters: {
            hidden_dim: 768,
            latent_dim: 8192,
            architecture_type: SAEArchitectureType.STANDARD,
            l1_alpha: 0.001,
            target_l0: 0.05,
            learning_rate: 0.0003,
            batch_size: 32,
            total_steps: 100000,
            warmup_steps: 10000,
            weight_decay: 0.01,
            grad_clip_norm: 1.0,
            checkpoint_interval: 1000,
            log_interval: 100,
            dead_neuron_threshold: 10000,
            resample_dead_neurons: true,
          },
        });
      });
    });

    it('should show loading state when starting training', async () => {
      const validConfig = {
        ...mockConfig,
        model_id: 'm_gpt2',
        dataset_id: 'ds_dataset1',
      };

      (useTrainingsStore as any).mockReturnValue({
        trainings: [],
        config: validConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      // Make createTraining hang so we can check loading state
      mockCreateTraining.mockImplementation(
        () => new Promise((resolve) => setTimeout(() => resolve({ id: 'train_new' }), 1000))
      );

      render(<TrainingPanel />);
      const startButton = screen.getByRole('button', { name: /Start Training/i });
      fireEvent.click(startButton);

      // Check loading state
      await waitFor(() => {
        expect(screen.getByText('Starting Training...')).toBeInTheDocument();
      });
    });
  });

  describe('Training Jobs List', () => {
    it('should display empty state when no trainings exist', () => {
      render(<TrainingPanel />);
      expect(screen.getByText('No training jobs yet')).toBeInTheDocument();
      expect(
        screen.getByText('Configure a training job above to get started')
      ).toBeInTheDocument();
    });

    it('should display loading state when loading trainings', () => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: [],
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: true,
        error: null,
      });

      render(<TrainingPanel />);
      // Look for a loader/spinner (Lucide Loader icon is used)
      expect(screen.getByText('Training Jobs')).toBeInTheDocument();
    });

    it('should render TrainingCard for each training', () => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: mockTrainings,
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      render(<TrainingPanel />);
      expect(screen.getByTestId('training-card-train_123')).toBeInTheDocument();
      expect(screen.getByTestId('training-card-train_456')).toBeInTheDocument();
    });

    it('should display error message when error exists', () => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: [],
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: 'Failed to load trainings',
      });

      render(<TrainingPanel />);
      expect(screen.getByText('Failed to load trainings')).toBeInTheDocument();
    });
  });

  describe('Status Filtering', () => {
    beforeEach(() => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: mockTrainings,
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });
    });

    it('should display status filter tabs with counts', () => {
      render(<TrainingPanel />);
      expect(screen.getByText(/All \(2\)/)).toBeInTheDocument();
      expect(screen.getByText(/Running \(1\)/)).toBeInTheDocument();
      expect(screen.getByText(/Completed \(1\)/)).toBeInTheDocument();
      expect(screen.getByText(/Failed \(0\)/)).toBeInTheDocument();
    });

    it('should call setStatusFilter when All tab is clicked', () => {
      render(<TrainingPanel />);
      const allButton = screen.getByRole('button', { name: /All \(2\)/ });
      fireEvent.click(allButton);
      expect(mockSetStatusFilter).toHaveBeenCalledWith('all');
    });

    it('should call setStatusFilter when Running tab is clicked', () => {
      render(<TrainingPanel />);
      const runningButton = screen.getByRole('button', { name: /Running \(1\)/ });
      fireEvent.click(runningButton);
      expect(mockSetStatusFilter).toHaveBeenCalledWith(TrainingStatus.RUNNING);
    });

    it('should call setStatusFilter when Completed tab is clicked', () => {
      render(<TrainingPanel />);
      const completedButton = screen.getByRole('button', { name: /Completed \(1\)/ });
      fireEvent.click(completedButton);
      expect(mockSetStatusFilter).toHaveBeenCalledWith(TrainingStatus.COMPLETED);
    });

    it('should call setStatusFilter when Failed tab is clicked', () => {
      render(<TrainingPanel />);
      const failedButton = screen.getByRole('button', { name: /Failed \(0\)/ });
      fireEvent.click(failedButton);
      expect(mockSetStatusFilter).toHaveBeenCalledWith(TrainingStatus.FAILED);
    });
  });

  describe('Selection and Deletion', () => {
    beforeEach(() => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: mockTrainings,
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });
    });

    it('should toggle selection when training checkbox is clicked', () => {
      render(<TrainingPanel />);
      const checkbox1 = screen.getByTestId('checkbox-train_123');

      // Select
      fireEvent.click(checkbox1);
      expect(checkbox1).toBeChecked();

      // Deselect
      fireEvent.click(checkbox1);
      expect(checkbox1).not.toBeChecked();
    });

    it('should select all trainings when "Select All" is clicked', () => {
      render(<TrainingPanel />);
      const selectAllCheckbox = screen.getByLabelText('Select All');

      fireEvent.click(selectAllCheckbox);

      expect(screen.getByTestId('checkbox-train_123')).toBeChecked();
      expect(screen.getByTestId('checkbox-train_456')).toBeChecked();
    });

    it('should deselect all when "Select All" is clicked again', () => {
      render(<TrainingPanel />);
      const selectAllCheckbox = screen.getByLabelText('Select All');

      // Select all
      fireEvent.click(selectAllCheckbox);

      // Deselect all
      fireEvent.click(selectAllCheckbox);

      expect(screen.getByTestId('checkbox-train_123')).not.toBeChecked();
      expect(screen.getByTestId('checkbox-train_456')).not.toBeChecked();
    });

    it('should show Delete Selected button when trainings are selected', () => {
      render(<TrainingPanel />);
      const checkbox1 = screen.getByTestId('checkbox-train_123');

      fireEvent.click(checkbox1);

      expect(screen.getByText(/Delete Selected \(1\)/)).toBeInTheDocument();
    });

    it('should call deleteTraining for each selected training when Delete Selected is clicked', async () => {
      global.confirm = vi.fn(() => true);
      mockDeleteTraining.mockResolvedValue(undefined);

      render(<TrainingPanel />);

      // Select both trainings
      fireEvent.click(screen.getByTestId('checkbox-train_123'));
      fireEvent.click(screen.getByTestId('checkbox-train_456'));

      // Click delete
      const deleteButton = screen.getByText(/Delete Selected \(2\)/);
      fireEvent.click(deleteButton);

      await waitFor(() => {
        expect(mockDeleteTraining).toHaveBeenCalledWith('train_123');
        expect(mockDeleteTraining).toHaveBeenCalledWith('train_456');
        expect(mockDeleteTraining).toHaveBeenCalledTimes(2);
      });
    });

    it('should show confirmation dialog when deleting multiple trainings', () => {
      global.confirm = vi.fn(() => false);

      render(<TrainingPanel />);

      // Select trainings
      fireEvent.click(screen.getByTestId('checkbox-train_123'));
      fireEvent.click(screen.getByTestId('checkbox-train_456'));

      // Click delete
      const deleteButton = screen.getByText(/Delete Selected \(2\)/);
      fireEvent.click(deleteButton);

      expect(global.confirm).toHaveBeenCalledWith(
        'Are you sure you want to delete 2 training jobs? This will remove all associated data and cannot be undone.'
      );
    });

    it('should not delete if confirmation is cancelled', () => {
      global.confirm = vi.fn(() => false);

      render(<TrainingPanel />);

      // Select training
      fireEvent.click(screen.getByTestId('checkbox-train_123'));

      // Click delete
      const deleteButton = screen.getByText(/Delete Selected \(1\)/);
      fireEvent.click(deleteButton);

      expect(mockDeleteTraining).not.toHaveBeenCalled();
    });
  });

  describe('WebSocket Integration', () => {
    it('should subscribe to WebSocket updates for all trainings', () => {
      (useTrainingsStore as any).mockReturnValue({
        trainings: mockTrainings,
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      render(<TrainingPanel />);

      expect(useTrainingWebSocket).toHaveBeenCalledWith(['train_123', 'train_456']);
    });

    it('should update WebSocket subscriptions when trainings change', () => {
      const { rerender } = render(<TrainingPanel />);

      expect(useTrainingWebSocket).toHaveBeenCalledWith([]);

      // Update to have trainings
      (useTrainingsStore as any).mockReturnValue({
        trainings: mockTrainings,
        config: mockConfig,
        updateConfig: mockUpdateConfig,
        fetchTrainings: mockFetchTrainings,
        createTraining: mockCreateTraining,
        deleteTraining: mockDeleteTraining,
        statusFilter: 'all',
        setStatusFilter: mockSetStatusFilter,
        isLoading: false,
        error: null,
      });

      rerender(<TrainingPanel />);

      expect(useTrainingWebSocket).toHaveBeenCalledWith(['train_123', 'train_456']);
    });
  });
});

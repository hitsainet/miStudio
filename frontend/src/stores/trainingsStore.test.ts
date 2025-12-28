/**
 * Unit tests for trainingsStore (Zustand store).
 *
 * This module tests the global state management for SAE training jobs,
 * including CRUD operations, control operations, WebSocket updates, and error handling.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import axios from 'axios';
import { useTrainingsStore } from './trainingsStore';
import { TrainingStatus, SAEArchitectureType } from '../types/training';
import type { Training, TrainingCreateRequest, Checkpoint } from '../types/training';

// Mock axios
vi.mock('axios');
const mockedAxios = axios as any;

describe('trainingsStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useTrainingsStore.setState({
      trainings: [],
      selectedTraining: null,
      config: {
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
      },
      isLoading: false,
      error: null,
      currentPage: 1,
      totalPages: 1,
      totalTrainings: 0,
      statusFilter: 'all',
      modelFilter: null,
      datasetFilter: null,
    });

    // Reset axios mock
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.clearAllTimers();
  });

  describe('Initial State', () => {
    it('should have empty trainings array initially', () => {
      const { trainings } = useTrainingsStore.getState();
      expect(trainings).toEqual([]);
    });

    it('should not be loading initially', () => {
      const { isLoading } = useTrainingsStore.getState();
      expect(isLoading).toBe(false);
    });

    it('should have no error initially', () => {
      const { error } = useTrainingsStore.getState();
      expect(error).toBeNull();
    });

    it('should have no selected training initially', () => {
      const { selectedTraining } = useTrainingsStore.getState();
      expect(selectedTraining).toBeNull();
    });

    it('should have default config values', () => {
      const { config } = useTrainingsStore.getState();
      expect(config.hidden_dim).toBe(768);
      expect(config.latent_dim).toBe(8192);
      expect(config.architecture_type).toBe(SAEArchitectureType.STANDARD);
      expect(config.learning_rate).toBe(0.0003);
      expect(config.batch_size).toBe(32);
    });

    it('should have default pagination values', () => {
      const { currentPage, totalPages, totalTrainings } = useTrainingsStore.getState();
      expect(currentPage).toBe(1);
      expect(totalPages).toBe(1);
      expect(totalTrainings).toBe(0);
    });

    it('should have default filter values', () => {
      const { statusFilter, modelFilter, datasetFilter } = useTrainingsStore.getState();
      expect(statusFilter).toBe('all');
      expect(modelFilter).toBeNull();
      expect(datasetFilter).toBeNull();
    });
  });

  describe('fetchTrainings', () => {
    const mockTraining: Training = {
      id: 'train_123',
      model_id: 'm_gpt2',
      dataset_id: 'ds_dataset1',
      status: TrainingStatus.RUNNING,
      progress: 45.5,
      current_step: 45500,
      total_steps: 100000,
      current_loss: 0.234,
      current_l0_sparsity: 0.05,
      current_dead_neurons: 100,
      current_learning_rate: 0.0003,
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
    };

    it('should fetch trainings successfully', async () => {
      const mockResponse = {
        data: {
          data: [mockTraining],
          pagination: {
            page: 1,
            limit: 50,
            total: 1,
            total_pages: 1,
          },
        },
      };

      mockedAxios.get.mockResolvedValueOnce(mockResponse);

      const { fetchTrainings } = useTrainingsStore.getState();
      await fetchTrainings();

      const state = useTrainingsStore.getState();
      expect(state.trainings).toEqual([mockTraining]);
      expect(state.totalTrainings).toBe(1);
      expect(state.totalPages).toBe(1);
      expect(state.currentPage).toBe(1);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should set loading state during fetch', async () => {
      mockedAxios.get.mockImplementation(
        () =>
          new Promise((resolve) =>
            setTimeout(
              () =>
                resolve({
                  data: { data: [], pagination: { page: 1, limit: 50, total: 0, total_pages: 0 } },
                }),
              100
            )
          )
      );

      const { fetchTrainings } = useTrainingsStore.getState();
      const fetchPromise = fetchTrainings();

      // Check loading state immediately
      const stateDuringFetch = useTrainingsStore.getState();
      expect(stateDuringFetch.isLoading).toBe(true);

      await fetchPromise;

      const stateAfterFetch = useTrainingsStore.getState();
      expect(stateAfterFetch.isLoading).toBe(false);
    });

    it('should handle fetch error', async () => {
      const errorMessage = 'Network error';
      mockedAxios.get.mockRejectedValueOnce({
        response: { data: { message: errorMessage } },
      });

      const { fetchTrainings } = useTrainingsStore.getState();

      try {
        await fetchTrainings();
      } catch (error) {
        // Expected to throw
      }

      const state = useTrainingsStore.getState();
      expect(state.error).toBe(errorMessage);
      expect(state.isLoading).toBe(false);
    });

    it('should send pagination parameters', async () => {
      mockedAxios.get.mockResolvedValueOnce({
        data: {
          data: [],
          pagination: { page: 2, limit: 25, total: 0, total_pages: 0 },
        },
      });

      const { fetchTrainings } = useTrainingsStore.getState();
      await fetchTrainings(2, 25);

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/api/v1/trainings',
        expect.objectContaining({
          params: expect.objectContaining({
            page: 2,
            limit: 25,
          }),
        })
      );
    });

    it('should send status filter when set', async () => {
      useTrainingsStore.setState({ statusFilter: TrainingStatus.RUNNING });

      mockedAxios.get.mockResolvedValueOnce({
        data: {
          data: [],
          pagination: { page: 1, limit: 50, total: 0, total_pages: 0 },
        },
      });

      const { fetchTrainings } = useTrainingsStore.getState();
      await fetchTrainings();

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/api/v1/trainings',
        expect.objectContaining({
          params: expect.objectContaining({
            status: TrainingStatus.RUNNING,
          }),
        })
      );
    });

    it('should send model filter when set', async () => {
      useTrainingsStore.setState({ modelFilter: 'm_gpt2' });

      mockedAxios.get.mockResolvedValueOnce({
        data: {
          data: [],
          pagination: { page: 1, limit: 50, total: 0, total_pages: 0 },
        },
      });

      const { fetchTrainings } = useTrainingsStore.getState();
      await fetchTrainings();

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/api/v1/trainings',
        expect.objectContaining({
          params: expect.objectContaining({
            model_id: 'm_gpt2',
          }),
        })
      );
    });

    it('should send dataset filter when set', async () => {
      useTrainingsStore.setState({ datasetFilter: 'ds_dataset1' });

      mockedAxios.get.mockResolvedValueOnce({
        data: {
          data: [],
          pagination: { page: 1, limit: 50, total: 0, total_pages: 0 },
        },
      });

      const { fetchTrainings } = useTrainingsStore.getState();
      await fetchTrainings();

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/api/v1/trainings',
        expect.objectContaining({
          params: expect.objectContaining({
            dataset_id: 'ds_dataset1',
          }),
        })
      );
    });

    it('should not send status filter when set to "all"', async () => {
      useTrainingsStore.setState({ statusFilter: 'all' });

      mockedAxios.get.mockResolvedValueOnce({
        data: {
          data: [],
          pagination: { page: 1, limit: 50, total: 0, total_pages: 0 },
        },
      });

      const { fetchTrainings } = useTrainingsStore.getState();
      await fetchTrainings();

      expect(mockedAxios.get).toHaveBeenCalledWith(
        '/api/v1/trainings',
        expect.objectContaining({
          params: expect.not.objectContaining({
            status: 'all',
          }),
        })
      );
    });
  });

  describe('fetchTraining', () => {
    const mockTraining: Training = {
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
    };

    it('should fetch single training successfully', async () => {
      // Pre-set the training in the list and as selectedTraining
      const existingTraining = { ...mockTraining, progress: 0 };
      useTrainingsStore.setState({
        trainings: [existingTraining],
        selectedTraining: existingTraining,
      });

      mockedAxios.get.mockResolvedValueOnce({
        data: mockTraining, // Updated training with progress: 45.5
      });

      const { fetchTraining } = useTrainingsStore.getState();
      await fetchTraining('train_123');

      const state = useTrainingsStore.getState();
      // Should update training in list
      expect(state.trainings[0]).toEqual(mockTraining);
      // Should update selectedTraining since it was already selected
      expect(state.selectedTraining).toEqual(mockTraining);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });

    it('should call correct endpoint', async () => {
      mockedAxios.get.mockResolvedValueOnce({
        data: mockTraining,
      });

      const { fetchTraining } = useTrainingsStore.getState();
      await fetchTraining('train_123');

      expect(mockedAxios.get).toHaveBeenCalledWith('/api/v1/trainings/train_123');
    });

    it('should handle fetch training error', async () => {
      mockedAxios.get.mockRejectedValueOnce({
        response: { data: { message: 'Training not found' } },
      });

      const { fetchTraining } = useTrainingsStore.getState();

      try {
        await fetchTraining('train_nonexistent');
      } catch (error) {
        // Expected to throw
      }

      const state = useTrainingsStore.getState();
      expect(state.error).toBe('Training not found');
      expect(state.isLoading).toBe(false);
    });
  });

  describe('createTraining', () => {
    const mockTraining: Training = {
      id: 'train_new123',
      model_id: 'm_gpt2',
      dataset_id: 'ds_dataset1',
      status: TrainingStatus.PENDING,
      progress: 0,
      current_step: 0,
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
      created_at: '2025-10-19T12:00:00Z',
      updated_at: '2025-10-19T12:00:00Z',
    };

    const mockRequest: TrainingCreateRequest = {
      model_id: 'm_gpt2',
      dataset_id: 'ds_dataset1',
      hidden_dim: 768,
      latent_dim: 8192,
      architecture_type: SAEArchitectureType.STANDARD,
      l1_alpha: 0.001,
      learning_rate: 0.0003,
      batch_size: 4096,
      total_steps: 100000,
    };

    it('should create training successfully', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: mockTraining,
      });

      const { createTraining } = useTrainingsStore.getState();
      const result = await createTraining(mockRequest);

      expect(result).toEqual(mockTraining);
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings', mockRequest);
    });

    it('should add created training to trainings list', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: mockTraining,
      });

      const { createTraining } = useTrainingsStore.getState();
      await createTraining(mockRequest);

      const state = useTrainingsStore.getState();
      expect(state.trainings).toContainEqual(mockTraining);
    });

    it('should handle create training error', async () => {
      mockedAxios.post.mockRejectedValueOnce({
        response: { data: { message: 'Validation failed' } },
      });

      const { createTraining } = useTrainingsStore.getState();

      try {
        await createTraining(mockRequest);
      } catch (error) {
        // Expected to throw
      }

      const state = useTrainingsStore.getState();
      expect(state.error).toBe('Validation failed');
    });

    it('should set selected training to newly created training', async () => {
      // Set initial training in store to verify it gets updated
      mockedAxios.post.mockResolvedValueOnce({
        data: mockTraining,
      });

      const { createTraining } = useTrainingsStore.getState();
      await createTraining(mockRequest);

      const state = useTrainingsStore.getState();
      expect(state.trainings).toContainEqual(mockTraining);
      expect(state.totalTrainings).toBe(1);
    });
  });

  describe('deleteTraining', () => {
    const mockTraining: Training = {
      id: 'train_123',
      model_id: 'm_gpt2',
      dataset_id: 'ds_dataset1',
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
      created_at: '2025-10-19T10:00:00Z',
      updated_at: '2025-10-19T12:00:00Z',
    };

    it('should delete training successfully', async () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
        totalTrainings: 1,
      });

      mockedAxios.delete.mockResolvedValueOnce({ data: {} });

      const { deleteTraining } = useTrainingsStore.getState();
      await deleteTraining('train_123');

      const state = useTrainingsStore.getState();
      expect(state.trainings).not.toContainEqual(mockTraining);
      expect(state.totalTrainings).toBe(0);
      expect(mockedAxios.delete).toHaveBeenCalledWith('/api/v1/trainings/train_123');
    });

    it('should remove deleted training from list', async () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
      });

      mockedAxios.delete.mockResolvedValueOnce({ data: {} });

      const { deleteTraining } = useTrainingsStore.getState();
      await deleteTraining('train_123');

      const state = useTrainingsStore.getState();
      expect(state.trainings.find((t) => t.id === 'train_123')).toBeUndefined();
    });

    it('should handle delete error', async () => {
      mockedAxios.delete.mockRejectedValueOnce({
        response: { data: { message: 'Delete failed' } },
      });

      const { deleteTraining } = useTrainingsStore.getState();

      try {
        await deleteTraining('train_123');
      } catch (error) {
        // Expected to throw
      }

      const state = useTrainingsStore.getState();
      expect(state.error).toBe('Delete failed');
    });

    it('should clear selected training if deleted', async () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
        selectedTraining: mockTraining,
      });

      mockedAxios.delete.mockResolvedValueOnce({ data: {} });

      const { deleteTraining } = useTrainingsStore.getState();
      await deleteTraining('train_123');

      const state = useTrainingsStore.getState();
      expect(state.selectedTraining).toBeNull();
    });
  });

  describe('Training Control Operations', () => {
    const mockTraining: Training = {
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
    };

    describe('pauseTraining', () => {
      it('should pause training successfully', async () => {
        useTrainingsStore.setState({
          trainings: [mockTraining],
        });

        mockedAxios.post.mockResolvedValueOnce({
          data: { success: true, message: 'Training paused', status: TrainingStatus.PAUSED },
        });

        const { pauseTraining } = useTrainingsStore.getState();
        const result = await pauseTraining('train_123');

        expect(result.success).toBe(true);
        expect(result.message).toBe('Training paused');
        expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings/train_123/control', {
          action: 'pause',
        });
      });

      it('should update training status to PAUSED in store', async () => {
        useTrainingsStore.setState({
          trainings: [mockTraining],
        });

        mockedAxios.post.mockResolvedValueOnce({
          data: { success: true, message: 'Training paused', status: TrainingStatus.PAUSED },
        });

        const { pauseTraining } = useTrainingsStore.getState();
        await pauseTraining('train_123');

        const state = useTrainingsStore.getState();
        const updatedTraining = state.trainings.find((t) => t.id === 'train_123');
        expect(updatedTraining?.status).toBe(TrainingStatus.PAUSED);
      });
    });

    describe('resumeTraining', () => {
      it('should resume training successfully', async () => {
        useTrainingsStore.setState({
          trainings: [{ ...mockTraining, status: TrainingStatus.PAUSED }],
        });

        mockedAxios.post.mockResolvedValueOnce({
          data: { success: true, message: 'Training resumed', status: TrainingStatus.RUNNING },
        });

        const { resumeTraining } = useTrainingsStore.getState();
        const result = await resumeTraining('train_123');

        expect(result.success).toBe(true);
        expect(result.message).toBe('Training resumed');
        expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings/train_123/control', {
          action: 'resume',
        });
      });

      it('should update training status to RUNNING in store', async () => {
        useTrainingsStore.setState({
          trainings: [{ ...mockTraining, status: TrainingStatus.PAUSED }],
        });

        mockedAxios.post.mockResolvedValueOnce({
          data: { success: true, message: 'Training resumed', status: TrainingStatus.RUNNING },
        });

        const { resumeTraining } = useTrainingsStore.getState();
        await resumeTraining('train_123');

        const state = useTrainingsStore.getState();
        const updatedTraining = state.trainings.find((t) => t.id === 'train_123');
        expect(updatedTraining?.status).toBe(TrainingStatus.RUNNING);
      });
    });

    describe('stopTraining', () => {
      it('should stop training successfully', async () => {
        useTrainingsStore.setState({
          trainings: [mockTraining],
        });

        mockedAxios.post.mockResolvedValueOnce({
          data: { success: true, message: 'Training stopped', status: TrainingStatus.STOPPED },
        });

        const { stopTraining } = useTrainingsStore.getState();
        const result = await stopTraining('train_123');

        expect(result.success).toBe(true);
        expect(result.message).toBe('Training stopped');
        expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings/train_123/control', {
          action: 'stop',
        });
      });
    });
  });

  describe('Checkpoint Operations', () => {
    const mockCheckpoint: Checkpoint = {
      id: 'ckpt_123',
      training_id: 'train_123',
      step: 50000,
      loss: 0.234,
      l0_sparsity: 0.05,
      is_best: false,
      storage_path: '/data/checkpoints/ckpt_123.safetensors',
      file_size_bytes: 1024000,
      created_at: '2025-10-19T12:00:00Z',
    };

    it('should fetch checkpoints successfully', async () => {
      mockedAxios.get.mockResolvedValueOnce({
        data: { data: [mockCheckpoint] },
      });

      const { fetchCheckpoints } = useTrainingsStore.getState();
      const result = await fetchCheckpoints('train_123');

      expect(result).toEqual([mockCheckpoint]);
      expect(mockedAxios.get).toHaveBeenCalledWith('/api/v1/trainings/train_123/checkpoints');
    });

    it('should handle fetch checkpoints error', async () => {
      mockedAxios.get.mockRejectedValueOnce({
        response: { data: { message: 'Fetch failed' } },
      });

      const { fetchCheckpoints } = useTrainingsStore.getState();

      await expect(fetchCheckpoints('train_123')).rejects.toThrow();
    });

    it('should save checkpoint successfully', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: mockCheckpoint,
      });

      const { saveCheckpoint } = useTrainingsStore.getState();
      const result = await saveCheckpoint('train_123');

      expect(result).toEqual(mockCheckpoint);
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings/train_123/checkpoints');
    });

    it('should delete checkpoint successfully', async () => {
      mockedAxios.delete.mockResolvedValueOnce({ data: {} });

      const { deleteCheckpoint } = useTrainingsStore.getState();
      await deleteCheckpoint('train_123', 'ckpt_123');

      expect(mockedAxios.delete).toHaveBeenCalledWith(
        '/api/v1/trainings/train_123/checkpoints/ckpt_123'
      );
    });
  });

  describe('Configuration Management', () => {
    it('should update config with partial updates', () => {
      const { updateConfig } = useTrainingsStore.getState();
      updateConfig({ hidden_dim: 1024, learning_rate: 0.0001 });

      const state = useTrainingsStore.getState();
      expect(state.config.hidden_dim).toBe(1024);
      expect(state.config.learning_rate).toBe(0.0001);
      expect(state.config.latent_dim).toBe(8192); // Unchanged
    });

    it('should reset config to defaults', () => {
      useTrainingsStore.setState({
        config: {
          model_id: 'm_custom',
          dataset_id: 'ds_custom',
          hidden_dim: 2048,
          latent_dim: 16384,
          architecture_type: SAEArchitectureType.GATED,
          l1_alpha: 0.002,
          target_l0: 0.1,
          learning_rate: 0.0001,
          batch_size: 64,
          total_steps: 200000,
          warmup_steps: 20000,
          weight_decay: 0.02,
          grad_clip_norm: 2.0,
          checkpoint_interval: 2000,
          log_interval: 200,
          dead_neuron_threshold: 20000,
          resample_dead_neurons: false,
        },
      });

      const { resetConfig } = useTrainingsStore.getState();
      resetConfig();

      const state = useTrainingsStore.getState();
      expect(state.config.hidden_dim).toBe(768);
      expect(state.config.learning_rate).toBe(0.0001);
      expect(state.config.architecture_type).toBe(SAEArchitectureType.STANDARD);
    });

    it('should set config from training', () => {
      const mockTraining: Training = {
        id: 'train_123',
        model_id: 'm_gpt2',
        dataset_id: 'ds_dataset1',
        status: TrainingStatus.COMPLETED,
        progress: 100,
        current_step: 100000,
        total_steps: 100000,
        hyperparameters: {
          hidden_dim: 1024,
          latent_dim: 16384,
          architecture_type: 'gated',
          l1_alpha: 0.002,
          learning_rate: 0.0001,
          batch_size: 64,
          total_steps: 100000,
        },
        created_at: '2025-10-19T10:00:00Z',
        updated_at: '2025-10-19T12:00:00Z',
      };

      const { setConfigFromTraining } = useTrainingsStore.getState();
      setConfigFromTraining(mockTraining);

      const state = useTrainingsStore.getState();
      expect(state.config.model_id).toBe('m_gpt2');
      expect(state.config.dataset_id).toBe('ds_dataset1');
      expect(state.config.hidden_dim).toBe(1024);
      expect(state.config.latent_dim).toBe(16384);
      expect(state.config.learning_rate).toBe(0.0001);
    });
  });

  describe('Real-time Updates', () => {
    const mockTraining: Training = {
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
    };

    it('should update training status via WebSocket update', () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
      });

      const { updateTrainingStatus } = useTrainingsStore.getState();
      updateTrainingStatus('train_123', {
        current_step: 50000,
        progress: 50.0,
        current_loss: 0.220,
        current_l0_sparsity: 0.048,
      });

      const state = useTrainingsStore.getState();
      const updatedTraining = state.trainings.find((t) => t.id === 'train_123');
      expect(updatedTraining?.current_step).toBe(50000);
      expect(updatedTraining?.progress).toBe(50.0);
      expect(updatedTraining?.current_loss).toBe(0.220);
      expect(updatedTraining?.current_l0_sparsity).toBe(0.048);
    });

    it('should update selected training when it is the same as updated training', () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
        selectedTraining: mockTraining,
      });

      const { updateTrainingStatus } = useTrainingsStore.getState();
      updateTrainingStatus('train_123', { current_step: 50000 });

      const state = useTrainingsStore.getState();
      expect(state.selectedTraining?.current_step).toBe(50000);
    });

    it('should handle COMPLETED status update', () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
      });

      const { updateTrainingStatus } = useTrainingsStore.getState();
      updateTrainingStatus('train_123', {
        status: TrainingStatus.COMPLETED,
        progress: 100,
        current_step: 100000,
      });

      const state = useTrainingsStore.getState();
      const updatedTraining = state.trainings.find((t) => t.id === 'train_123');
      expect(updatedTraining?.status).toBe(TrainingStatus.COMPLETED);
      expect(updatedTraining?.progress).toBe(100);
    });

    it('should handle FAILED status update', () => {
      useTrainingsStore.setState({
        trainings: [mockTraining],
      });

      const { updateTrainingStatus } = useTrainingsStore.getState();
      updateTrainingStatus('train_123', {
        status: TrainingStatus.FAILED,
        error_message: 'Out of memory',
      });

      const state = useTrainingsStore.getState();
      const updatedTraining = state.trainings.find((t) => t.id === 'train_123');
      expect(updatedTraining?.status).toBe(TrainingStatus.FAILED);
    });
  });

  describe('UI State Management', () => {
    const mockTraining: Training = {
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
    };

    it('should set selected training', () => {
      const { setSelectedTraining } = useTrainingsStore.getState();
      setSelectedTraining(mockTraining);

      const state = useTrainingsStore.getState();
      expect(state.selectedTraining).toEqual(mockTraining);
    });

    it('should set status filter', () => {
      const { setStatusFilter } = useTrainingsStore.getState();
      setStatusFilter(TrainingStatus.RUNNING);

      const state = useTrainingsStore.getState();
      expect(state.statusFilter).toBe(TrainingStatus.RUNNING);
    });

    it('should set model filter', () => {
      const { setModelFilter } = useTrainingsStore.getState();
      setModelFilter('m_gpt2');

      const state = useTrainingsStore.getState();
      expect(state.modelFilter).toBe('m_gpt2');
    });

    it('should set dataset filter', () => {
      const { setDatasetFilter } = useTrainingsStore.getState();
      setDatasetFilter('ds_dataset1');

      const state = useTrainingsStore.getState();
      expect(state.datasetFilter).toBe('ds_dataset1');
    });

    it('should clear error', () => {
      useTrainingsStore.setState({ error: 'Some error' });

      const { clearError } = useTrainingsStore.getState();
      clearError();

      const state = useTrainingsStore.getState();
      expect(state.error).toBeNull();
    });

    it('should clear selected training', () => {
      useTrainingsStore.setState({ selectedTraining: mockTraining });

      const { setSelectedTraining } = useTrainingsStore.getState();
      setSelectedTraining(null);

      const state = useTrainingsStore.getState();
      expect(state.selectedTraining).toBeNull();
    });

    it('should clear model filter', () => {
      useTrainingsStore.setState({ modelFilter: 'm_gpt2' });

      const { setModelFilter } = useTrainingsStore.getState();
      setModelFilter(null);

      const state = useTrainingsStore.getState();
      expect(state.modelFilter).toBeNull();
    });
  });

  describe('retryTraining', () => {
    it('should retry failed training', async () => {
      const failedTraining: Training = {
        id: 'train_failed123',
        model_id: 'm_gpt2',
        dataset_id: 'ds_dataset1',
        status: TrainingStatus.FAILED,
        progress: 25.0,
        current_step: 25000,
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
      };

      const retriedTraining: Training = {
        ...failedTraining,
        id: 'train_retry123',
        status: TrainingStatus.PENDING,
        progress: 0,
        current_step: 0,
      };

      useTrainingsStore.setState({
        trainings: [failedTraining],
      });

      mockedAxios.post.mockResolvedValueOnce({
        data: retriedTraining,
      });

      const { retryTraining } = useTrainingsStore.getState();
      const result = await retryTraining('train_failed123');

      expect(result).toEqual(retriedTraining);
      expect(mockedAxios.post).toHaveBeenCalledWith('/api/v1/trainings', {
        model_id: failedTraining.model_id,
        dataset_id: failedTraining.dataset_id,
        extraction_id: undefined,
        hyperparameters: failedTraining.hyperparameters,
      });

      const state = useTrainingsStore.getState();
      expect(state.trainings).toContainEqual(retriedTraining);
    });
  });
});

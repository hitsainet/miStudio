/**
 * Unit tests for ActivationExtractionConfig component.
 *
 * Tests cover:
 * - Modal rendering and display
 * - Dataset selection dropdown
 * - Layer selector grid (toggle, select all, deselect all)
 * - Hook type selection (residual, MLP, attention)
 * - Settings inputs (batch size, max samples, top-K)
 * - Form validation
 * - Extraction summary display
 * - Start extraction button functionality
 * - Close modal functionality
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, fireEvent, waitFor } from '@testing-library/react';
// Component tree uses useWebSocketContext() (via useModelExtractionProgress),
// so it must be wrapped in <WebSocketProvider>. renderWithProviders does that.
import { renderWithProviders as render } from '../../test/renderWithProviders';
import { ActivationExtractionConfig } from './ActivationExtractionConfig';
import { Model, ModelStatus, QuantizationFormat } from '../../types/model';
import { useDatasetsStore } from '../../stores/datasetsStore';

// Mock the datasetsStore
vi.mock('../../stores/datasetsStore', () => ({
  useDatasetsStore: vi.fn(),
}));

describe('ActivationExtractionConfig', () => {
  const mockOnClose = vi.fn();
  const mockOnExtract = vi.fn();
  const mockFetchDatasets = vi.fn();

  const testModel: Model = {
    id: 'm_test123',
    name: 'TinyLlama-1.1B',
    repo_id: 'TinyLlama/TinyLlama-1.1B',
    architecture: 'llama',
    params_count: 1100000000,
    quantization: QuantizationFormat.Q4,
    status: ModelStatus.READY,
    architecture_config: {
      num_layers: 22,
      hidden_size: 2048,
      num_attention_heads: 32,
      intermediate_size: 5632,
      vocab_size: 32000,
      max_position_embeddings: 2048,
      num_key_value_heads: 4,
      rope_theta: 10000.0,
    },
    created_at: '2025-10-12T00:00:00Z',
    updated_at: '2025-10-12T00:00:00Z',
  };

  const mockDatasets = [
    {
      id: 'ds_1',
      name: 'OpenWebText',
      status: 'ready',
      num_samples: 1000000,
      repo_id: 'openwebtext',
      created_at: '2025-10-12T00:00:00Z',
      updated_at: '2025-10-12T00:00:00Z',
    },
    {
      id: 'ds_2',
      name: 'TinyStories',
      status: 'ready',
      num_samples: 500000,
      repo_id: 'tinystories',
      created_at: '2025-10-12T00:00:00Z',
      updated_at: '2025-10-12T00:00:00Z',
    },
  ];

  // The component gates "Start Extraction" on the selected dataset having a
  // READY tokenization for this model. The first ready dataset (ds_1) is
  // auto-selected, so provide a matching ready tokenization keyed by dataset id.
  const mockTokenizations = {
    ds_1: [
      {
        id: 'tok_1',
        dataset_id: 'ds_1',
        model_id: 'm_test123',
        max_length: 512,
        tokenizer_repo_id: 'TinyLlama/TinyLlama-1.1B',
        status: 'ready',
        created_at: '2025-10-12T00:00:00Z',
        updated_at: '2025-10-12T00:00:00Z',
      },
    ],
  };

  beforeEach(() => {
    mockOnClose.mockClear();
    mockOnExtract.mockClear();
    mockFetchDatasets.mockClear();

    // Default mock implementation.
    // Component destructures: datasets, fetchDatasets, tokenizations, fetchTokenizations.
    (useDatasetsStore as any).mockReturnValue({
      datasets: mockDatasets,
      fetchDatasets: mockFetchDatasets,
      tokenizations: mockTokenizations,
      fetchTokenizations: vi.fn(),
    });
  });

  describe('Modal Display', () => {
    it('should render modal with backdrop', () => {
      const { container } = render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const backdrop = container.querySelector('.fixed.inset-0.bg-black\\/50');
      expect(backdrop).toBeInTheDocument();
    });

    it('should render modal header with model name', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.getByText('Extract Activations')).toBeInTheDocument();
      expect(screen.getByText(/Extract activations from TinyLlama-1\.1B/)).toBeInTheDocument();
    });

    it('should call fetchDatasets on mount', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(mockFetchDatasets).toHaveBeenCalledTimes(1);
    });

    it('should render close button', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const closeButton = screen.getByLabelText('Close');
      expect(closeButton).toBeInTheDocument();
    });
  });

  describe('Dataset Selection', () => {
    it('should render dataset dropdown with ready datasets', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.getByText('Select Dataset')).toBeInTheDocument();
      expect(screen.getByText(/OpenWebText/)).toBeInTheDocument();
      expect(screen.getByText(/TinyStories/)).toBeInTheDocument();
    });

    it('should show warning when no ready datasets available', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        fetchDatasets: mockFetchDatasets,
      });

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.getByText(/No ready datasets available/)).toBeInTheDocument();
    });

    it('should filter out non-ready datasets', () => {
      const mixedDatasets = [
        ...mockDatasets,
        {
          id: 'ds_3',
          name: 'Downloading Dataset',
          status: 'downloading',
          num_samples: 0,
          repo_id: 'downloading',
          created_at: '2025-10-12T00:00:00Z',
          updated_at: '2025-10-12T00:00:00Z',
        },
      ];

      (useDatasetsStore as any).mockReturnValue({
        datasets: mixedDatasets,
        fetchDatasets: mockFetchDatasets,
        tokenizations: mockTokenizations,
        fetchTokenizations: vi.fn(),
      });

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      // Should only show ready datasets
      expect(screen.getByText(/OpenWebText/)).toBeInTheDocument();
      expect(screen.getByText(/TinyStories/)).toBeInTheDocument();
      expect(screen.queryByText(/Downloading Dataset/)).not.toBeInTheDocument();
    });

    it('should allow changing selected dataset', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const select = screen.getByLabelText('Select Dataset') as HTMLSelectElement;
      fireEvent.change(select, { target: { value: 'ds_2' } });

      expect(select.value).toBe('ds_2');
    });
  });

  describe('Layer Selection', () => {
    it('should render layer selector grid', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      // Model has 22 layers (from architecture_config)
      expect(screen.getByText('L0')).toBeInTheDocument();
      expect(screen.getByText('L21')).toBeInTheDocument();
      expect(screen.queryByText('L22')).not.toBeInTheDocument();
    });

    it('should show selected layer count', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      // Default selection is [0, 5, 11]
      expect(screen.getByText(/Select Layers \(3 selected\)/)).toBeInTheDocument();
    });

    it('should toggle layer selection on click', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const layer10Button = screen.getByText('L10');

      // Selection is signalled by the emerald accent. The UNSELECTED classes
      // are dual-mode ("bg-white dark:bg-slate-800"), so asserting a bare
      // "bg-slate-800" pinned a single-theme spelling rather than the state.
      expect(layer10Button).not.toHaveClass('bg-emerald-600');

      fireEvent.click(layer10Button);
      expect(layer10Button).toHaveClass('bg-emerald-600');

      fireEvent.click(layer10Button);
      expect(layer10Button).not.toHaveClass('bg-emerald-600');
    });

    it('should select all layers when Select All clicked', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const selectAllButton = screen.getByText('Select All');
      fireEvent.click(selectAllButton);

      // Should show 22 layers selected
      expect(screen.getByText(/Select Layers \(22 selected\)/)).toBeInTheDocument();
    });

    it('should deselect all layers when Deselect All clicked', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const deselectAllButton = screen.getByText('Deselect All');
      fireEvent.click(deselectAllButton);

      // Should show 0 layers selected
      expect(screen.getByText(/Select Layers \(0 selected\)/)).toBeInTheDocument();
    });

    it('should offer no layers when the config records no depth', () => {
      // REWRITTEN 2026-08-25. This test used to assert the opposite --
      // "Should default to 12 layers" -- and so pinned the defect rather than
      // preventing it. gemma-4-12B-it has 48 layers and records its depth in a
      // nested text_config; the picker showed L0-L11 and made layers 12-47
      // unreachable, with nothing on screen saying the count was invented.
      const modelWithoutLayers = {
        ...testModel,
        architecture_config: {
          ...testModel.architecture_config,
          num_layers: undefined,
        },
      };

      render(
        <ActivationExtractionConfig
          model={modelWithoutLayers}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.queryByText('L0')).not.toBeInTheDocument();
      expect(screen.queryByText('L11')).not.toBeInTheDocument();
      expect(screen.getByText(/records no layer count/i)).toBeInTheDocument();
    });
  });

  describe('Hook Type Selection', () => {
    it('should render all hook type buttons', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.getByText('residual')).toBeInTheDocument();
      expect(screen.getByText('mlp')).toBeInTheDocument();
      expect(screen.getByText('attention')).toBeInTheDocument();
    });

    it('should have residual selected by default', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const residualButton = screen.getByText('residual');
      expect(residualButton).toHaveClass('bg-purple-600');
    });

    it('should toggle hook type selection', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const mlpButton = screen.getByText('mlp');

      // As above: assert the state marker, not a single-theme spelling of the
      // unselected background.
      expect(mlpButton).not.toHaveClass('bg-purple-600');

      fireEvent.click(mlpButton);
      expect(mlpButton).toHaveClass('bg-purple-600');

      fireEvent.click(mlpButton);
      expect(mlpButton).not.toHaveClass('bg-purple-600');
    });

    it('should allow multiple hook types selected', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const mlpButton = screen.getByText('mlp');
      const attentionButton = screen.getByText('attention');

      fireEvent.click(mlpButton);
      fireEvent.click(attentionButton);

      expect(mlpButton).toHaveClass('bg-purple-600');
      expect(attentionButton).toHaveClass('bg-purple-600');
    });
  });

  describe('Extraction Settings', () => {
    it('should render batch size input with default value', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const batchSizeInput = screen.getByLabelText('Batch Size') as HTMLInputElement;
      expect(batchSizeInput).toBeInTheDocument();
      expect(batchSizeInput.value).toBe('32');
    });

    it('should render max samples input with default value', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const maxSamplesInput = screen.getByLabelText('Max Samples') as HTMLInputElement;
      expect(maxSamplesInput).toBeInTheDocument();
      // Component default is 100000 (useState(100000)).
      expect(maxSamplesInput.value).toBe('100000');
    });

    it('should render top-K examples input with default value', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const topKInput = screen.getByLabelText('Top K Examples') as HTMLInputElement;
      expect(topKInput).toBeInTheDocument();
      expect(topKInput.value).toBe('10');
    });

    it('should allow changing batch size', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const batchSizeInput = screen.getByLabelText('Batch Size') as HTMLInputElement;
      fireEvent.change(batchSizeInput, { target: { value: '64' } });

      expect(batchSizeInput.value).toBe('64');
    });

    it('should allow changing max samples', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const maxSamplesInput = screen.getByLabelText('Max Samples') as HTMLInputElement;
      fireEvent.change(maxSamplesInput, { target: { value: '5000' } });

      expect(maxSamplesInput.value).toBe('5000');
    });

    it('should allow changing top-K examples', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const topKInput = screen.getByLabelText('Top K Examples') as HTMLInputElement;
      fireEvent.change(topKInput, { target: { value: '20' } });

      expect(topKInput.value).toBe('20');
    });
  });

  describe('Extraction Summary', () => {
    it('should display extraction summary', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.getByText('Extraction Summary:')).toBeInTheDocument();
      expect(screen.getByText(/Will extract from 3 layer\(s\)/)).toBeInTheDocument();
      expect(screen.getByText(/Using 1 hook type\(s\): residual/)).toBeInTheDocument();
      expect(screen.getByText(/Processing up to 100,000 samples/)).toBeInTheDocument();
      expect(screen.getByText(/Batch size: 32/)).toBeInTheDocument();
    });

    it('should update summary when layers change', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const selectAllButton = screen.getByText('Select All');
      fireEvent.click(selectAllButton);

      expect(screen.getByText(/Will extract from 22 layer\(s\)/)).toBeInTheDocument();
    });

    it('should update summary when hook types change', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const mlpButton = screen.getByText('mlp');
      const attentionButton = screen.getByText('attention');
      fireEvent.click(mlpButton);
      fireEvent.click(attentionButton);

      expect(screen.getByText(/Using 3 hook type\(s\): residual, mlp, attention/)).toBeInTheDocument();
    });
  });

  describe('Form Validation', () => {
    it('should disable button when no dataset selected', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        fetchDatasets: mockFetchDatasets,
      });

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      // Button should be disabled when no ready datasets
      const extractButton = screen.getByText('Start Extraction');
      expect(extractButton).toBeDisabled();
    });

    it('should disable button when no layers selected', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      // Deselect all layers
      const deselectAllButton = screen.getByText('Deselect All');
      fireEvent.click(deselectAllButton);

      // Button should be disabled
      const extractButton = screen.getByText('Start Extraction');
      expect(extractButton).toBeDisabled();
    });

    it('should disable button when no hook types selected', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      // Deselect residual (the only selected hook type by default)
      const residualButton = screen.getByText('residual');
      fireEvent.click(residualButton);

      // Button should be disabled
      const extractButton = screen.getByText('Start Extraction');
      expect(extractButton).toBeDisabled();
    });

    it('should show error for invalid batch size', async () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const batchSizeInput = screen.getByLabelText('Batch Size');
      fireEvent.change(batchSizeInput, { target: { value: '300' } });

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      await waitFor(() => {
        expect(screen.getByText('Batch size must be between 1 and 256')).toBeInTheDocument();
      });

      expect(mockOnExtract).not.toHaveBeenCalled();
    });

    it('should show error for invalid max samples', async () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const maxSamplesInput = screen.getByLabelText('Max Samples');
      // Component limit is 1,000,000; use a value above it to trigger the error.
      fireEvent.change(maxSamplesInput, { target: { value: '2000000' } });

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      await waitFor(() => {
        expect(screen.getByText('Max samples must be between 1 and 1,000,000')).toBeInTheDocument();
      });

      expect(mockOnExtract).not.toHaveBeenCalled();
    });
  });

  describe('Start Extraction', () => {
    it('should call onExtract with correct config', async () => {
      mockOnExtract.mockResolvedValue(undefined);

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      await waitFor(() => {
        expect(mockOnExtract).toHaveBeenCalledWith('m_test123', {
          dataset_id: 'ds_1',
          layer_indices: [0, 5, 11],
          hook_types: ['residual'],
          max_samples: 100000,
          batch_size: 32,
          micro_batch_size: 8,
          top_k_examples: 10,
          gpu_id: 0,
        });
      });
    });

    it('should show extracting state during submission', async () => {
      mockOnExtract.mockImplementation(() => new Promise((resolve) => setTimeout(resolve, 500)));

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      // Should show extracting state immediately
      await waitFor(() => {
        expect(screen.getByText('Starting Extraction...')).toBeInTheDocument();
      });

      // Wait for extraction to complete
      await waitFor(() => {
        expect(mockOnExtract).toHaveBeenCalled();
      }, { timeout: 2000 });

      // Modal should remain open (not call onClose)
      expect(mockOnClose).not.toHaveBeenCalled();
    });

    it('should close modal after successful extraction starts', async () => {
      mockOnExtract.mockResolvedValue(undefined);

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      // Wait for extraction to complete
      await waitFor(() => {
        expect(mockOnExtract).toHaveBeenCalled();
      }, { timeout: 2000 });

      // Component closes the modal once extraction has started successfully so
      // the user watches progress on the model card (see handleExtract).
      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });
    });

    it('should show error message on extraction failure', async () => {
      mockOnExtract.mockRejectedValue(new Error('Extraction failed'));

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      await waitFor(() => {
        expect(screen.getByText('Extraction failed')).toBeInTheDocument();
      });

      expect(mockOnClose).not.toHaveBeenCalled();
    });

    it('should disable button when no ready datasets', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        fetchDatasets: mockFetchDatasets,
      });

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const extractButton = screen.getByText('Start Extraction');
      expect(extractButton).toBeDisabled();
    });
  });

  describe('Close Functionality', () => {
    it('should call onClose when X button is clicked', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const closeButton = screen.getByLabelText('Close');
      fireEvent.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should disable close button during extraction', async () => {
      mockOnExtract.mockImplementation(() => new Promise((resolve) => setTimeout(resolve, 100)));

      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const extractButton = screen.getByText('Start Extraction');
      fireEvent.click(extractButton);

      // Close button should be disabled during extraction
      const closeButton = screen.getByLabelText('Close');
      expect(closeButton).toBeDisabled();

      // Wait for extraction to complete
      await waitFor(() => {
        expect(mockOnExtract).toHaveBeenCalled();
      });

      // On success the component closes the modal (onClose) rather than
      // re-enabling the close button.
      await waitFor(() => {
        expect(mockOnClose).toHaveBeenCalled();
      });
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA label on close button', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      const closeButton = screen.getByLabelText('Close');
      expect(closeButton).toBeInTheDocument();
      expect(closeButton).toHaveAttribute('aria-label', 'Close');
    });

    it('should have proper labels for all form inputs', () => {
      render(
        <ActivationExtractionConfig
          model={testModel}
          onClose={mockOnClose}
          onExtract={mockOnExtract}
        />
      );

      expect(screen.getByLabelText('Select Dataset')).toBeInTheDocument();
      expect(screen.getByLabelText('Batch Size')).toBeInTheDocument();
      expect(screen.getByLabelText('Max Samples')).toBeInTheDocument();
      expect(screen.getByLabelText('Top K Examples')).toBeInTheDocument();
    });
  });
});

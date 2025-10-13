/**
 * Unit tests for ModelCard component.
 *
 * Tests cover:
 * - Rendering model information correctly
 * - Status icons and badges display
 * - Progress bars for active downloads
 * - Action buttons (Extract, Delete) conditional rendering
 * - Error message display
 * - Click handlers
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModelCard } from './ModelCard';
import { Model, ModelStatus, QuantizationFormat } from '../../types/model';

describe('ModelCard', () => {
  const mockOnClick = vi.fn();
  const mockOnExtract = vi.fn();
  const mockOnDelete = vi.fn();
  const mockOnCancel = vi.fn();

  const baseModel: Model = {
    id: 'm_test123',
    name: 'TinyLlama-1.1B',
    repo_id: 'TinyLlama/TinyLlama-1.1B',
    architecture: 'llama',
    params_count: 1100000000,
    quantization: QuantizationFormat.Q4,
    status: ModelStatus.READY,
    progress: 100,
    created_at: '2025-10-12T00:00:00Z',
    updated_at: '2025-10-12T00:00:00Z',
  };

  beforeEach(() => {
    mockOnClick.mockClear();
    mockOnExtract.mockClear();
    mockOnDelete.mockClear();
    mockOnCancel.mockClear();
  });

  describe('Model Information Display', () => {
    it('should render model name', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('TinyLlama-1.1B')).toBeInTheDocument();
    });

    it('should render formatted parameters count', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/1\.1B params/)).toBeInTheDocument();
    });

    it('should render quantization format', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/Q4 quantization/)).toBeInTheDocument();
    });

    it('should render memory requirement when available', () => {
      const modelWithMemory = {
        ...baseModel,
        memory_required_bytes: 369363763, // ~352 MB
      };

      render(
        <ModelCard
          model={modelWithMemory}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/352 MB memory/)).toBeInTheDocument();
    });

    it('should render repo ID when available', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('TinyLlama/TinyLlama-1.1B')).toBeInTheDocument();
    });
  });

  describe('Status Icons and Badges', () => {
    it('should show CheckCircle icon for ready status', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const statusBadge = screen.getByText('Ready');
      expect(statusBadge).toBeInTheDocument();
      expect(statusBadge).toHaveClass('text-emerald-400');
    });

    it('should show Loader icon for downloading status', () => {
      const downloadingModel = {
        ...baseModel,
        status: ModelStatus.DOWNLOADING,
        progress: 45,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const statusBadge = screen.getByText('Downloading');
      expect(statusBadge).toBeInTheDocument();
      expect(statusBadge).toHaveClass('text-blue-400');
    });

    it('should show Activity icon for quantizing status', () => {
      const quantizingModel = {
        ...baseModel,
        status: ModelStatus.QUANTIZING,
        progress: 75,
      };

      render(
        <ModelCard
          model={quantizingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const statusBadge = screen.getByText('Quantizing');
      expect(statusBadge).toBeInTheDocument();
      expect(statusBadge).toHaveClass('text-purple-400');
    });

    it('should show AlertCircle icon for error status', () => {
      const errorModel = {
        ...baseModel,
        status: ModelStatus.ERROR,
        error_message: 'Out of memory',
      };

      render(
        <ModelCard
          model={errorModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const statusBadge = screen.getByText('Error');
      expect(statusBadge).toBeInTheDocument();
      expect(statusBadge).toHaveClass('text-red-400');
    });
  });

  describe('Progress Bar', () => {
    it('should show progress bar for downloading model', () => {
      const downloadingModel = {
        ...baseModel,
        status: ModelStatus.DOWNLOADING,
        progress: 45.7,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Download Progress')).toBeInTheDocument();
      expect(screen.getByText('45.7%')).toBeInTheDocument();
    });

    it('should show progress bar for loading model', () => {
      const loadingModel = {
        ...baseModel,
        status: ModelStatus.LOADING,
        progress: 60,
      };

      render(
        <ModelCard
          model={loadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Loading Model')).toBeInTheDocument();
      expect(screen.getByText('60.0%')).toBeInTheDocument();
    });

    it('should show progress bar for quantizing model', () => {
      const quantizingModel = {
        ...baseModel,
        status: ModelStatus.QUANTIZING,
        progress: 85,
      };

      render(
        <ModelCard
          model={quantizingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Quantizing Model')).toBeInTheDocument();
      expect(screen.getByText('85.0%')).toBeInTheDocument();
    });

    it('should not show progress bar for ready model', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.queryByText(/Progress/)).not.toBeInTheDocument();
    });
  });

  describe('Action Buttons', () => {
    it('should show Extract Activations button only for ready models', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Extract Activations')).toBeInTheDocument();
    });

    it('should not show Extract button for downloading models', () => {
      const downloadingModel = {
        ...baseModel,
        status: ModelStatus.DOWNLOADING,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.queryByText('Extract Activations')).not.toBeInTheDocument();
    });

    it('should show delete button for ready models', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      // Delete button has title attribute
      const deleteButton = screen.getByTitle('Delete model');
      expect(deleteButton).toBeInTheDocument();
    });

    it('should not show delete button for active downloads', () => {
      const downloadingModel = {
        ...baseModel,
        status: ModelStatus.DOWNLOADING,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.queryByTitle('Delete model')).not.toBeInTheDocument();
    });
  });

  describe('Click Handlers', () => {
    it('should call onClick when model info is clicked', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const modelName = screen.getByText('TinyLlama-1.1B');
      fireEvent.click(modelName);

      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it('should call onExtract when Extract button is clicked', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const extractButton = screen.getByText('Extract Activations');
      fireEvent.click(extractButton);

      expect(mockOnExtract).toHaveBeenCalledTimes(1);
      expect(mockOnClick).not.toHaveBeenCalled(); // Should not trigger model click
    });

    it('should show confirmation and call onDelete when delete button is clicked', () => {
      // Mock window.confirm
      const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const deleteButton = screen.getByTitle('Delete model');
      fireEvent.click(deleteButton);

      expect(confirmSpy).toHaveBeenCalledWith('Are you sure you want to delete TinyLlama-1.1B?');
      expect(mockOnDelete).toHaveBeenCalledWith('m_test123');

      confirmSpy.mockRestore();
    });

    it('should not call onDelete if confirmation is cancelled', () => {
      const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);

      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const deleteButton = screen.getByTitle('Delete model');
      fireEvent.click(deleteButton);

      expect(mockOnDelete).not.toHaveBeenCalled();

      confirmSpy.mockRestore();
    });
  });

  describe('Error Display', () => {
    it('should display error message for failed models', () => {
      const errorModel = {
        ...baseModel,
        status: ModelStatus.ERROR,
        error_message: 'Failed to download: Network timeout',
      };

      render(
        <ModelCard
          model={errorModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText('Failed to download: Network timeout')).toBeInTheDocument();
    });

    it('should not display error section for successful models', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      // Should not have any error styling or messages
      const card = screen.getByText('TinyLlama-1.1B').closest('div');
      expect(card?.innerHTML).not.toContain('bg-red-500');
    });
  });

  describe('Parameter Formatting', () => {
    it('should format billions correctly', () => {
      const billionModel = {
        ...baseModel,
        params_count: 7000000000, // 7B
      };

      render(
        <ModelCard
          model={billionModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/7\.0B params/)).toBeInTheDocument();
    });

    it('should format millions correctly', () => {
      const millionModel = {
        ...baseModel,
        params_count: 135000000, // 135M
      };

      render(
        <ModelCard
          model={millionModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/135M params/)).toBeInTheDocument();
    });
  });

  describe('Memory Formatting', () => {
    it('should format gigabytes correctly', () => {
      const largeModel = {
        ...baseModel,
        memory_required_bytes: 2147483648, // 2 GB
      };

      render(
        <ModelCard
          model={largeModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/2\.0 GB memory/)).toBeInTheDocument();
    });

    it('should format megabytes correctly', () => {
      const smallModel = {
        ...baseModel,
        memory_required_bytes: 524288000, // ~500 MB
      };

      render(
        <ModelCard
          model={smallModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.getByText(/500 MB memory/)).toBeInTheDocument();
    });
  });

  describe('Cancel Button', () => {
    it('should show Cancel button for downloading models', () => {
      const downloadingModel = {
        ...baseModel,
        status: ModelStatus.DOWNLOADING,
        progress: 45,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const cancelButton = screen.getByTitle('Cancel download');
      expect(cancelButton).toBeInTheDocument();
    });

    it('should show Cancel button for loading models', () => {
      const loadingModel = {
        ...baseModel,
        status: ModelStatus.LOADING,
        progress: 70,
      };

      render(
        <ModelCard
          model={loadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const cancelButton = screen.getByTitle('Cancel download');
      expect(cancelButton).toBeInTheDocument();
    });

    it('should show Cancel button for quantizing models', () => {
      const quantizingModel = {
        ...baseModel,
        status: ModelStatus.QUANTIZING,
        progress: 85,
      };

      render(
        <ModelCard
          model={quantizingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const cancelButton = screen.getByTitle('Cancel download');
      expect(cancelButton).toBeInTheDocument();
    });

    it('should not show Cancel button for ready models', () => {
      render(
        <ModelCard
          model={baseModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.queryByTitle('Cancel download')).not.toBeInTheDocument();
    });

    it('should not show Cancel button for error models', () => {
      const errorModel = {
        ...baseModel,
        status: ModelStatus.ERROR,
        error_message: 'Failed',
      };

      render(
        <ModelCard
          model={errorModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      expect(screen.queryByTitle('Cancel download')).not.toBeInTheDocument();
    });

    it('should show confirmation and call onCancel when Cancel button is clicked', () => {
      const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);

      const downloadingModel = {
        ...baseModel,
        id: 'm_cancel123',
        status: ModelStatus.DOWNLOADING,
        progress: 50,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const cancelButton = screen.getByTitle('Cancel download');
      fireEvent.click(cancelButton);

      expect(confirmSpy).toHaveBeenCalledWith(
        'Are you sure you want to cancel this download? Partial files will be deleted.'
      );
      expect(mockOnCancel).toHaveBeenCalledWith('m_cancel123');
      expect(mockOnClick).not.toHaveBeenCalled(); // Should not trigger model click

      confirmSpy.mockRestore();
    });

    it('should not call onCancel if confirmation is cancelled', () => {
      const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(false);

      const downloadingModel = {
        ...baseModel,
        status: ModelStatus.DOWNLOADING,
        progress: 50,
      };

      render(
        <ModelCard
          model={downloadingModel}
          onClick={mockOnClick}
          onExtract={mockOnExtract}
          onDelete={mockOnDelete}
          onCancel={mockOnCancel}
        />
      );

      const cancelButton = screen.getByTitle('Cancel download');
      fireEvent.click(cancelButton);

      expect(mockOnCancel).not.toHaveBeenCalled();

      confirmSpy.mockRestore();
    });
  });
});

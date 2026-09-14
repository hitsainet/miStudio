/**
 * Unit tests for ModelPreviewModal component.
 *
 * Tests modal rendering, model info fetching, quantization selection,
 * memory calculation, and download functionality.
 */

import { describe, it, expect, beforeEach, vi, Mock } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ModelPreviewModal } from './ModelPreviewModal';
import * as huggingfaceApi from '../../api/huggingface';
import { QuantizationFormat } from '../../types/model';

// Mock the huggingface API
vi.mock('../../api/huggingface', () => ({
  getModelInfo: vi.fn(),
  calculateMemoryRequirement: vi.fn(),
  formatBytes: vi.fn(),
}));

describe('ModelPreviewModal', () => {
  const mockOnClose = vi.fn();
  const mockOnDownload = vi.fn();
  const mockModelInfo = {
    id: 'TinyLlama/TinyLlama-1.1B',
    modelId: 'TinyLlama/TinyLlama-1.1B',
    private: false,
    pipeline_tag: 'text-generation',
    tags: ['pytorch', 'llama', 'text-generation'],
    downloads: 150000,
    likes: 250,
    config: {
      model_type: 'llama',
      architectures: ['LlamaForCausalLM'],
    },
    cardData: {
      license: 'apache-2.0',
      language: ['en'],
    },
  };

  beforeEach(() => {
    vi.clearAllMocks();
    (huggingfaceApi.formatBytes as Mock).mockImplementation((bytes: number) => {
      if (bytes >= 1e9) return `${(bytes / 1e9).toFixed(2)} GB`;
      if (bytes >= 1e6) return `${(bytes / 1e6).toFixed(2)} MB`;
      return `${bytes} Bytes`;
    });
    (huggingfaceApi.calculateMemoryRequirement as Mock).mockImplementation(
      (params: number, quant: string) => {
        const multipliers: Record<string, number> = {
          FP32: 4,
          FP16: 2,
          Q8: 1,
          Q4: 0.5,
          Q2: 0.25,
        };
        return params * multipliers[quant] * 1.2;
      }
    );
  });

  describe('Rendering', () => {
    it('should render modal title and repository ID', () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      expect(screen.getByText('Model Preview')).toBeInTheDocument();
      expect(screen.getByText('TinyLlama/TinyLlama-1.1B')).toBeInTheDocument();
    });

    it('should render close button', () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      const closeButton = screen.getByRole('button', { name: /close/i });
      expect(closeButton).toBeInTheDocument();
    });

    it('should show loading state initially', () => {
      (huggingfaceApi.getModelInfo as Mock).mockImplementation(
        () => new Promise(() => {}) // Never resolves
      );
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      expect(screen.getByText('Loading model information...')).toBeInTheDocument();
    });
  });

  describe('Model Info Fetching', () => {
    it('should fetch model info on mount', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(huggingfaceApi.getModelInfo).toHaveBeenCalledWith('TinyLlama/TinyLlama-1.1B');
      });
    });

    it('should display model statistics', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('150,000')).toBeInTheDocument(); // downloads
        expect(screen.getByText('250')).toBeInTheDocument(); // likes
        // text-generation appears in both pipeline and tags, so check for multiple elements
        const textGenElements = screen.getAllByText('text-generation');
        expect(textGenElements.length).toBeGreaterThan(0);
      });
    });

    it('should display model tags', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('pytorch')).toBeInTheDocument();
        // llama and text-generation appear in multiple places, just verify they exist
        const llamaElements = screen.getAllByText('llama');
        const textGenElements = screen.getAllByText('text-generation');
        expect(llamaElements.length).toBeGreaterThan(0);
        expect(textGenElements.length).toBeGreaterThan(0);
      });
    });

    it('should display architecture information', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Architecture')).toBeInTheDocument();
        expect(screen.getByText('LlamaForCausalLM')).toBeInTheDocument();
        // llama appears in multiple places
        const llamaElements = screen.getAllByText('llama');
        expect(llamaElements.length).toBeGreaterThan(0);
      });
    });

    it('should display license information', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('License')).toBeInTheDocument();
        expect(screen.getByText('apache-2.0')).toBeInTheDocument();
      });
    });

    it('should display language information', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Language')).toBeInTheDocument();
        expect(screen.getByText('en')).toBeInTheDocument();
      });
    });

    it('should handle missing optional fields', async () => {
      const minimalModelInfo = {
        ...mockModelInfo,
        cardData: undefined,
        config: undefined,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(minimalModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('150,000')).toBeInTheDocument();
      });

      // Should not crash, but these sections won't be rendered
      expect(screen.queryByText('License')).not.toBeInTheDocument();
      expect(screen.queryByText('Architecture')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should display error message when fetch fails', async () => {
      const errorMessage = 'Failed to fetch model info';
      (huggingfaceApi.getModelInfo as Mock).mockRejectedValueOnce(new Error(errorMessage));
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Failed to load model information')).toBeInTheDocument();
        expect(screen.getByText(errorMessage)).toBeInTheDocument();
      });
    });

    it('should display generic error for non-Error objects', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockRejectedValueOnce('string error');
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Failed to load model information')).toBeInTheDocument();
        expect(screen.getByText('Failed to fetch model information')).toBeInTheDocument();
      });
    });
  });

  describe('Memory Requirements Table', () => {
    it('should display all quantization options', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
        expect(screen.getByText('FP32 (Full Precision)')).toBeInTheDocument();
        expect(screen.getByText('FP16 (Half Precision)')).toBeInTheDocument();
        expect(screen.getByText('Q8 (8-bit)')).toBeInTheDocument();
        expect(screen.getByText('Q4 (4-bit) - Recommended')).toBeInTheDocument();
        expect(screen.getByText('Q2 (2-bit)')).toBeInTheDocument();
      });
    });

    it('should display memory estimates for each quantization', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      // Verify calculateMemoryRequirement was called for each quantization
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(
        expect.any(Number),
        'FP32'
      );
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(
        expect.any(Number),
        'FP16'
      );
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(
        expect.any(Number),
        'Q8'
      );
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(
        expect.any(Number),
        'Q4'
      );
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(
        expect.any(Number),
        'Q2'
      );
    });

    it('should show disclaimer about memory estimates', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(
          screen.getByText(/Memory estimates include 20% overhead for inference/i)
        ).toBeInTheDocument();
      });
    });
  });

  describe('Quantization Selection', () => {
    it('should default to Q4 quantization', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      const { container } = render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      const q4Radio = container.querySelector('input[type="radio"][value="Q4"]') as HTMLInputElement;
      expect(q4Radio).toBeChecked();
    });

    it('should allow selecting different quantization', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      const { container } = render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      const fp16Radio = container.querySelector('input[type="radio"][value="FP16"]') as HTMLInputElement;
      fireEvent.click(fp16Radio);

      expect(fp16Radio).toBeChecked();
    });

    it('should update selected quantization in download button text', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      const { container } = render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText(/Download with Q4/i)).toBeInTheDocument();
      });

      const fp16Radio = container.querySelector('input[type="radio"][value="FP16"]') as HTMLInputElement;
      fireEvent.click(fp16Radio);

      await waitFor(() => {
        expect(screen.getByText(/Download with FP16/i)).toBeInTheDocument();
      });
    });
  });

  describe('Parameter Count Estimation', () => {
    it('should extract parameter count from model ID with B suffix', async () => {
      const modelWith1B = {
        ...mockModelInfo,
        id: 'TinyLlama/TinyLlama-1.1B',
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWith1B);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      // Should call with 1.1B = 1.1e9 parameters
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(1.1e9, expect.any(String));
    });

    it('should extract parameter count from model ID with M suffix', async () => {
      const modelWith100M = {
        ...mockModelInfo,
        id: 'microsoft/DialoGPT-100M',
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWith100M);
      render(
        <ModelPreviewModal
          repoId="microsoft/DialoGPT-100M"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      // Should call with 100M = 100e6 parameters
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(100e6, expect.any(String));
    });

    it('should use fallback parameter count when not found in ID', async () => {
      const modelWithoutSize = {
        ...mockModelInfo,
        id: 'gpt2',
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithoutSize);
      render(
        <ModelPreviewModal
          repoId="gpt2"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      // Should use fallback of 1B parameters
      expect(huggingfaceApi.calculateMemoryRequirement).toHaveBeenCalledWith(1e9, expect.any(String));
    });
  });

  describe('Modal Actions', () => {
    it('should call onClose when close button is clicked', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      const closeButton = screen.getByRole('button', { name: /close/i });
      fireEvent.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when X button is clicked', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByLabelText(/close/i)).toBeInTheDocument();
      });

      const xButton = screen.getByLabelText(/close/i);
      fireEvent.click(xButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onClose when Cancel button is clicked', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /cancel/i })).toBeInTheDocument();
      });

      const cancelButton = screen.getByRole('button', { name: /cancel/i });
      fireEvent.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should call onDownload with selected quantization and trustRemoteCode false', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText(/Download with Q4/i)).toBeInTheDocument();
      });

      const downloadButton = screen.getByRole('button', { name: /Download with Q4/i });
      fireEvent.click(downloadButton);

      expect(mockOnDownload).toHaveBeenCalledWith(QuantizationFormat.Q4, false);
    });

    it('should call onDownload with different quantization when changed', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      const { container } = render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      // Change to FP16
      const fp16Radio = container.querySelector('input[type="radio"][value="FP16"]') as HTMLInputElement;
      fireEvent.click(fp16Radio);

      await waitFor(() => {
        expect(screen.getByText(/Download with FP16/i)).toBeInTheDocument();
      });

      const downloadButton = screen.getByRole('button', { name: /Download with FP16/i });
      fireEvent.click(downloadButton);

      expect(mockOnDownload).toHaveBeenCalledWith(QuantizationFormat.FP16, false);
    });

    it('should call onClose after successful download', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText(/Download with Q4/i)).toBeInTheDocument();
      });

      const downloadButton = screen.getByRole('button', { name: /Download with Q4/i });
      fireEvent.click(downloadButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it('should not render download button when onDownload is not provided', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal repoId="TinyLlama/TinyLlama-1.1B" onClose={mockOnClose} />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      expect(screen.queryByRole('button', { name: /Download with/i })).not.toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA label for close button', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByLabelText(/close/i)).toBeInTheDocument();
      });
    });

    it('should have proper radio button group for quantization', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      const { container } = render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      const radios = container.querySelectorAll('input[type="radio"][name="quantization"]');
      expect(radios.length).toBe(5); // FP32, FP16, Q8, Q4, Q2
    });

    it('should have proper table structure for memory requirements', async () => {
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(mockModelInfo);
      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      const table = screen.getByRole('table');
      expect(table).toBeInTheDocument();

      const headers = screen.getAllByRole('columnheader');
      expect(headers.length).toBe(3); // Select, Quantization, Est. Memory
    });
  });

  describe('Trust Remote Code Detection', () => {
    it('should display trust remote code warning when required', async () => {
      const modelWithTrustRemoteCode = {
        ...mockModelInfo,
        requiresTrustRemoteCode: true,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithTrustRemoteCode);

      render(
        <ModelPreviewModal
          repoId="microsoft/phi-2"
          onClose={mockOnClose}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Trust Remote Code Required')).toBeInTheDocument();
      });

      expect(screen.getByText(/This model requires executing custom code/)).toBeInTheDocument();
    });

    it('should not display trust remote code warning when not required', async () => {
      const modelWithoutTrustRemoteCode = {
        ...mockModelInfo,
        requiresTrustRemoteCode: false,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithoutTrustRemoteCode);

      render(
        <ModelPreviewModal
          repoId="TinyLlama/TinyLlama-1.1B"
          onClose={mockOnClose}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Memory Requirements')).toBeInTheDocument();
      });

      expect(screen.queryByText('Trust Remote Code Required')).not.toBeInTheDocument();
    });

    it('should display trust remote code checkbox in footer when required', async () => {
      const modelWithTrustRemoteCode = {
        ...mockModelInfo,
        requiresTrustRemoteCode: true,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithTrustRemoteCode);

      render(
        <ModelPreviewModal
          repoId="microsoft/phi-2"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByLabelText('Trust Remote Code')).toBeInTheDocument();
      });

      const checkbox = screen.getByLabelText('Trust Remote Code') as HTMLInputElement;
      expect(checkbox).not.toBeChecked();
    });

    it('should disable download button when trust remote code is required but not checked', async () => {
      const modelWithTrustRemoteCode = {
        ...mockModelInfo,
        requiresTrustRemoteCode: true,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithTrustRemoteCode);

      render(
        <ModelPreviewModal
          repoId="microsoft/phi-2"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByText(/Download with Q4/i)).toBeInTheDocument();
      });

      const downloadButton = screen.getByRole('button', { name: /Download with Q4/i });
      expect(downloadButton).toBeDisabled();
    });

    it('should enable download button when trust remote code checkbox is checked', async () => {
      const modelWithTrustRemoteCode = {
        ...mockModelInfo,
        requiresTrustRemoteCode: true,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithTrustRemoteCode);

      render(
        <ModelPreviewModal
          repoId="microsoft/phi-2"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByLabelText('Trust Remote Code')).toBeInTheDocument();
      });

      const checkbox = screen.getByLabelText('Trust Remote Code') as HTMLInputElement;
      fireEvent.click(checkbox);

      const downloadButton = screen.getByRole('button', { name: /Download with Q4/i });
      expect(downloadButton).not.toBeDisabled();
    });

    it('should call onDownload with trustRemoteCode true when checkbox is checked', async () => {
      const modelWithTrustRemoteCode = {
        ...mockModelInfo,
        requiresTrustRemoteCode: true,
      };
      (huggingfaceApi.getModelInfo as Mock).mockResolvedValueOnce(modelWithTrustRemoteCode);

      render(
        <ModelPreviewModal
          repoId="microsoft/phi-2"
          onClose={mockOnClose}
          onDownload={mockOnDownload}
        />
      );

      await waitFor(() => {
        expect(screen.getByLabelText('Trust Remote Code')).toBeInTheDocument();
      });

      const checkbox = screen.getByLabelText('Trust Remote Code') as HTMLInputElement;
      fireEvent.click(checkbox);

      const downloadButton = screen.getByRole('button', { name: /Download with Q4/i });
      fireEvent.click(downloadButton);

      expect(mockOnDownload).toHaveBeenCalledWith(QuantizationFormat.Q4, true);
    });
  });
});

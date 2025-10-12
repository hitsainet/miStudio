/**
 * Unit tests for ModelArchitectureViewer component.
 *
 * Tests cover:
 * - Modal rendering and display
 * - Architecture stats display
 * - Layer list rendering
 * - Configuration display
 * - Close functionality
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ModelArchitectureViewer } from './ModelArchitectureViewer';
import { Model, ModelStatus, QuantizationFormat } from '../../types/model';

describe('ModelArchitectureViewer', () => {
  const mockOnClose = vi.fn();

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

  beforeEach(() => {
    mockOnClose.mockClear();
  });

  describe('Modal Display', () => {
    it('should render modal with backdrop', () => {
      const { container } = render(
        <ModelArchitectureViewer model={testModel} onClose={mockOnClose} />
      );

      const backdrop = container.querySelector('.fixed.inset-0.bg-black\\/50');
      expect(backdrop).toBeInTheDocument();
    });

    it('should render model name in header', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText(/TinyLlama-1\.1B Architecture/)).toBeInTheDocument();
    });

    it('should render parameter count in header', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText(/1\.1B parameters/)).toBeInTheDocument();
    });

    it('should render quantization format in header', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText(/Q4 quantization/)).toBeInTheDocument();
    });

    it('should render architecture type when available', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      const llamaTexts = screen.getAllByText(/llama/);
      expect(llamaTexts.length).toBeGreaterThan(0);
    });
  });

  describe('Architecture Stats Grid', () => {
    it('should render Total Layers stat', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Total Layers')).toBeInTheDocument();
      // 22 layers + embedding + layernorm + output = 25 total
      expect(screen.getByText('25')).toBeInTheDocument();
    });

    it('should render Hidden Dimension stat', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Hidden Dimension')).toBeInTheDocument();
      const hiddenSizes = screen.getAllByText('2048');
      expect(hiddenSizes.length).toBeGreaterThan(0);
    });

    it('should render Attention Heads stat', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Attention Heads')).toBeInTheDocument();
      const attentionCounts = screen.getAllByText('32');
      expect(attentionCounts.length).toBeGreaterThan(0);
    });

    it('should render Parameters stat with formatting', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Parameters')).toBeInTheDocument();
      expect(screen.getByText('1.1B')).toBeInTheDocument();
    });
  });

  describe('Layer List Rendering', () => {
    it('should render "Model Layers" section header', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Model Layers')).toBeInTheDocument();
    });

    it('should render Embedding layer', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Embedding')).toBeInTheDocument();
      expect(screen.getByText(/32000 × 2048/)).toBeInTheDocument();
    });

    it('should render TransformerBlock layers', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('TransformerBlock_0')).toBeInTheDocument();
      expect(screen.getByText('TransformerBlock_21')).toBeInTheDocument();
    });

    it('should render LayerNorm layer', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('LayerNorm')).toBeInTheDocument();
    });

    it('should render Output layer', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Output')).toBeInTheDocument();
      expect(screen.getByText(/2048 × 32000/)).toBeInTheDocument();
    });

    it('should render attention and MLP details for transformer blocks', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      // Attention: 32 heads × 64 dims (2048 / 32) - appears for each transformer block
      const attentionDetails = screen.getAllByText(/32 heads × 64 dims/);
      expect(attentionDetails.length).toBeGreaterThan(0);
      // MLP: 2048 → 5632 → 2048
      const mlpDetails = screen.getAllByText(/2048 → 5632 → 2048/);
      expect(mlpDetails.length).toBeGreaterThan(0);
    });
  });

  describe('Model Configuration Section', () => {
    it('should render "Model Configuration" header', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Model Configuration')).toBeInTheDocument();
    });

    it('should render vocabulary size', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Vocabulary Size:')).toBeInTheDocument();
      const vocabSizes = screen.getAllByText('32000');
      expect(vocabSizes.length).toBeGreaterThan(0);
    });

    it('should render max position embeddings', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Max Position:')).toBeInTheDocument();
      const positions = screen.getAllByText('2048');
      expect(positions.length).toBeGreaterThan(0);
    });

    it('should render MLP ratio', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('MLP Ratio:')).toBeInTheDocument();
      // 5632 / 2048 ≈ 3
      expect(screen.getByText('3x')).toBeInTheDocument();
    });

    it('should render architecture type', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('Architecture:')).toBeInTheDocument();
      const llamaTexts = screen.getAllByText('llama');
      expect(llamaTexts.length).toBeGreaterThan(0);
    });

    it('should render KV heads when available', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('KV Heads:')).toBeInTheDocument();
      const kvHeads = screen.getAllByText('4');
      expect(kvHeads.length).toBeGreaterThan(0);
    });

    it('should render RoPE theta when available', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      expect(screen.getByText('RoPE Theta:')).toBeInTheDocument();
      expect(screen.getByText('10000')).toBeInTheDocument();
    });
  });

  describe('Close Functionality', () => {
    it('should call onClose when X button is clicked', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      const closeButton = screen.getByLabelText('Close');
      fireEvent.click(closeButton);

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });
  });

  describe('Default Configuration Handling', () => {
    it('should use defaults when architecture_config is missing', () => {
      const modelWithoutConfig: Model = {
        ...testModel,
        architecture_config: undefined,
      };

      render(<ModelArchitectureViewer model={modelWithoutConfig} onClose={mockOnClose} />);

      // Should render with default values (12 layers, 768 hidden, etc.)
      expect(screen.getByText('Model Layers')).toBeInTheDocument();
      expect(screen.getByText('Embedding')).toBeInTheDocument();
    });

    it('should handle partial architecture config', () => {
      const modelWithPartialConfig: Model = {
        ...testModel,
        architecture_config: {
          num_layers: 6,
          hidden_size: 512,
          // Missing other fields - should use defaults
        },
      };

      render(<ModelArchitectureViewer model={modelWithPartialConfig} onClose={mockOnClose} />);

      expect(screen.getByText('512')).toBeInTheDocument(); // Hidden size
      expect(screen.getByText('TransformerBlock_5')).toBeInTheDocument(); // Last layer
    });
  });

  describe('Different Model Sizes', () => {
    it('should handle small models correctly', () => {
      const smallModel: Model = {
        ...testModel,
        params_count: 135000000, // 135M
        architecture_config: {
          num_layers: 12,
          hidden_size: 768,
          num_attention_heads: 12,
          intermediate_size: 3072,
          vocab_size: 50257,
        },
      };

      render(<ModelArchitectureViewer model={smallModel} onClose={mockOnClose} />);

      expect(screen.getByText('135M')).toBeInTheDocument();
      const hiddenSizes = screen.getAllByText('768');
      expect(hiddenSizes.length).toBeGreaterThan(0); // Hidden size
      const attentionCounts = screen.getAllByText('12');
      expect(attentionCounts.length).toBeGreaterThan(0); // Attention heads
    });

    it('should handle large models correctly', () => {
      const largeModel: Model = {
        ...testModel,
        params_count: 70000000000, // 70B
        architecture_config: {
          num_layers: 80,
          hidden_size: 8192,
          num_attention_heads: 64,
          intermediate_size: 28672,
          vocab_size: 32000,
        },
      };

      render(<ModelArchitectureViewer model={largeModel} onClose={mockOnClose} />);

      expect(screen.getByText('70.0B')).toBeInTheDocument();
      const hiddenSizes = screen.getAllByText('8192');
      expect(hiddenSizes.length).toBeGreaterThan(0); // Hidden size
      const attentionCounts = screen.getAllByText('64');
      expect(attentionCounts.length).toBeGreaterThan(0); // Attention heads
    });
  });

  describe('Accessibility', () => {
    it('should have proper ARIA label on close button', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      const closeButton = screen.getByLabelText('Close');
      expect(closeButton).toBeInTheDocument();
      expect(closeButton).toHaveAttribute('aria-label', 'Close');
    });

    it('should be keyboard accessible', () => {
      render(<ModelArchitectureViewer model={testModel} onClose={mockOnClose} />);

      const closeButton = screen.getByLabelText('Close');

      // Simulate keyboard navigation
      closeButton.focus();
      expect(document.activeElement).toBe(closeButton);

      // Simulate Enter key
      fireEvent.keyPress(closeButton, { key: 'Enter', code: 'Enter', charCode: 13 });
    });
  });
});

/**
 * Unit tests for ModelDownloadForm component.
 *
 * Tests form rendering, validation, preview modal integration,
 * quantization selection, and download functionality.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ModelDownloadForm } from './ModelDownloadForm';
import { QuantizationFormat } from '../../types/model';

// Mock the ModelPreviewModal component
vi.mock('./ModelPreviewModal', () => ({
  ModelPreviewModal: ({ repoId, onClose, onDownload }: any) => (
    <div data-testid="model-preview-modal">
      <div>Model Preview Modal</div>
      <div>{repoId}</div>
      <button onClick={onClose}>Close</button>
      <button onClick={() => onDownload('Q4', false)}>Download with Q4</button>
    </div>
  ),
}));

describe('ModelDownloadForm', () => {
  const mockOnDownload = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render HuggingFace model repository input', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('type', 'text');
      expect(input).toHaveAttribute('placeholder', 'e.g., TinyLlama/TinyLlama-1.1B');
    });

    it('should render quantization format selector', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const select = screen.getByLabelText('Quantization Format');
      expect(select).toBeInTheDocument();

      // Check all quantization options are present
      expect(screen.getByText('FP32 (Full Precision)')).toBeInTheDocument();
      expect(screen.getByText('FP16 (Half Precision)')).toBeInTheDocument();
      expect(screen.getByText('Q8 (8-bit Quantization)')).toBeInTheDocument();
      expect(screen.getByText('Q4 (4-bit Quantization) - Recommended')).toBeInTheDocument();
      expect(screen.getByText('Q2 (2-bit Quantization)')).toBeInTheDocument();
    });

    it('should render access token input', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText(/Access Token/i);
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('type', 'password');
      expect(input).toHaveAttribute('placeholder', 'hf_xxxxxxxxxxxxxxxxxxxx');
    });

    it('should render trust remote code checkbox', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const checkbox = screen.getByLabelText('Trust Remote Code');
      expect(checkbox).toBeInTheDocument();
      expect(checkbox).toHaveAttribute('type', 'checkbox');
      expect(checkbox).not.toBeChecked();
    });

    it('should render Preview and Download buttons', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const previewButton = screen.getByRole('button', { name: /preview/i });
      const downloadButton = screen.getByRole('button', { name: /download/i });

      expect(previewButton).toBeInTheDocument();
      expect(downloadButton).toBeInTheDocument();
    });

    it('should render helper text for access token', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      expect(screen.getByText(/Required for gated models like Llama/i)).toBeInTheDocument();
    });

    it('should render warning about trust remote code', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      expect(screen.getByText(/Some models.*require executing custom code/i)).toBeInTheDocument();
    });
  });

  describe('Form State', () => {
    it('should update repository ID on input', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });

      expect(input.value).toBe('TinyLlama/TinyLlama-1.1B');
    });

    it('should update quantization on selection', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const select = screen.getByLabelText('Quantization Format') as HTMLSelectElement;
      fireEvent.change(select, { target: { value: QuantizationFormat.FP16 } });

      expect(select.value).toBe(QuantizationFormat.FP16);
    });

    it('should default to Q4 quantization', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const select = screen.getByLabelText('Quantization Format') as HTMLSelectElement;
      expect(select.value).toBe(QuantizationFormat.Q4);
    });

    it('should update access token on input', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText(/Access Token/i) as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'hf_token123' } });

      expect(input.value).toBe('hf_token123');
    });

    it('should toggle trust remote code checkbox', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const checkbox = screen.getByLabelText('Trust Remote Code') as HTMLInputElement;

      expect(checkbox.checked).toBe(false);

      fireEvent.click(checkbox);
      expect(checkbox.checked).toBe(true);

      fireEvent.click(checkbox);
      expect(checkbox.checked).toBe(false);
    });

    it('should disable buttons when repository ID is empty', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const previewButton = screen.getByRole('button', { name: /preview/i });
      const downloadButton = screen.getByRole('button', { name: /download/i });

      expect(previewButton).toBeDisabled();
      expect(downloadButton).toBeDisabled();
    });

    it('should enable buttons when repository ID is provided', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });

      const previewButton = screen.getByRole('button', { name: /preview/i });
      const downloadButton = screen.getByRole('button', { name: /download/i });

      expect(previewButton).not.toBeDisabled();
      expect(downloadButton).not.toBeDisabled();
    });
  });

  describe('Validation', () => {
    it('should show validation error for invalid repository ID format', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      // Enter invalid format (no slash)
      fireEvent.change(input, { target: { value: 'invalid-repo' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(/Invalid repository format/i)).toBeInTheDocument();
      });

      expect(mockOnDownload).not.toHaveBeenCalled();
    });

    it('should show validation error for empty repository ID', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');

      // Fill then clear
      fireEvent.change(input, { target: { value: 'test/model' } });
      fireEvent.change(input, { target: { value: '' } });

      const button = screen.getByRole('button', { name: /^download$/i });
      expect(button).toBeDisabled();
    });

    it('should clear validation error when input is corrected', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      // Trigger validation error
      fireEvent.change(input, { target: { value: 'invalid' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(/Invalid repository format/i)).toBeInTheDocument();
      });

      // Correct the input
      fireEvent.change(input, { target: { value: 'valid/repo' } });

      await waitFor(() => {
        expect(screen.queryByText(/Invalid repository format/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Preview Modal Integration', () => {
    it('should not show preview modal by default', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      expect(screen.queryByTestId('model-preview-modal')).not.toBeInTheDocument();
    });

    it('should show preview modal when Preview button is clicked', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const previewButton = screen.getByRole('button', { name: /preview/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(previewButton);

      await waitFor(() => {
        expect(screen.getByTestId('model-preview-modal')).toBeInTheDocument();
      });
    });

    it('should not show preview modal for invalid repository ID', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const previewButton = screen.getByRole('button', { name: /preview/i });

      fireEvent.change(input, { target: { value: 'invalid' } });
      fireEvent.click(previewButton);

      await waitFor(() => {
        expect(screen.queryByTestId('model-preview-modal')).not.toBeInTheDocument();
      });

      expect(screen.getByText(/Invalid repository format/i)).toBeInTheDocument();
    });

    it('should close preview modal when Close button is clicked', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const previewButton = screen.getByRole('button', { name: /preview/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(previewButton);

      await waitFor(() => {
        expect(screen.getByTestId('model-preview-modal')).toBeInTheDocument();
      });

      const closeButton = screen.getByText('Close');
      fireEvent.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByTestId('model-preview-modal')).not.toBeInTheDocument();
      });
    });

    it('should pass repository ID to preview modal', async () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const previewButton = screen.getByRole('button', { name: /preview/i });

      const repoId = 'TinyLlama/TinyLlama-1.1B';
      fireEvent.change(input, { target: { value: repoId } });
      fireEvent.click(previewButton);

      await waitFor(() => {
        expect(screen.getByText(repoId)).toBeInTheDocument();
      });
    });

    it('should download from preview modal and close it', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const previewButton = screen.getByRole('button', { name: /preview/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(previewButton);

      await waitFor(() => {
        expect(screen.getByTestId('model-preview-modal')).toBeInTheDocument();
      });

      const modalDownloadButton = screen.getByText('Download with Q4');
      fireEvent.click(modalDownloadButton);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(
          'TinyLlama/TinyLlama-1.1B',
          'Q4',
          undefined,
          false
        );
      });
    });
  });

  describe('Form Submission', () => {
    it('should call onDownload with repository ID and default quantization', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(
          'TinyLlama/TinyLlama-1.1B',
          QuantizationFormat.Q4,
          undefined,
          false
        );
      });
    });

    it('should call onDownload with selected quantization', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const select = screen.getByLabelText('Quantization Format');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.change(select, { target: { value: QuantizationFormat.FP16 } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(
          'TinyLlama/TinyLlama-1.1B',
          QuantizationFormat.FP16,
          undefined,
          false
        );
      });
    });

    it('should call onDownload with access token', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('HuggingFace Model Repository');
      const tokenInput = screen.getByLabelText(/Access Token/i);
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(repoInput, { target: { value: 'meta-llama/Llama-2-7b' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(
          'meta-llama/Llama-2-7b',
          QuantizationFormat.Q4,
          'hf_token123',
          false
        );
      });
    });

    it('should call onDownload with trust remote code enabled', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('HuggingFace Model Repository');
      const checkbox = screen.getByLabelText('Trust Remote Code');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(repoInput, { target: { value: 'microsoft/phi-2' } });
      fireEvent.click(checkbox);
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(
          'microsoft/phi-2',
          QuantizationFormat.Q4,
          undefined,
          true
        );
      });
    });

    it('should show loading state during submission', async () => {
      let resolveDownload: () => void;
      const downloadPromise = new Promise<void>((resolve) => {
        resolveDownload = resolve;
      });
      mockOnDownload.mockReturnValueOnce(downloadPromise);

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Downloading...')).toBeInTheDocument();
        expect(button).toBeDisabled();
      });

      // Resolve the promise
      resolveDownload!();

      await waitFor(() => {
        expect(screen.getByText(/^download$/i)).toBeInTheDocument();
      });
    });

    it('should disable all inputs during submission', async () => {
      let resolveDownload: () => void;
      const downloadPromise = new Promise<void>((resolve) => {
        resolveDownload = resolve;
      });
      mockOnDownload.mockReturnValueOnce(downloadPromise);

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('HuggingFace Model Repository');
      const select = screen.getByLabelText('Quantization Format');
      const tokenInput = screen.getByLabelText(/Access Token/i);
      const checkbox = screen.getByLabelText('Trust Remote Code');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(repoInput, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(repoInput).toBeDisabled();
        expect(select).toBeDisabled();
        expect(tokenInput).toBeDisabled();
        expect(checkbox).toBeDisabled();
        expect(button).toBeDisabled();
      });

      resolveDownload!();

      await waitFor(() => {
        expect(repoInput).not.toBeDisabled();
        expect(select).not.toBeDisabled();
        expect(tokenInput).not.toBeDisabled();
        expect(checkbox).not.toBeDisabled();
      });
    });

    it('should keep form values after successful download', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('HuggingFace Model Repository') as HTMLInputElement;
      const tokenInput = screen.getByLabelText(/Access Token/i) as HTMLInputElement;
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(repoInput, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalled();
      });

      // Form should retain values for convenience
      expect(repoInput.value).toBe('TinyLlama/TinyLlama-1.1B');
      expect(tokenInput.value).toBe('hf_token123');
    });
  });

  describe('Error Handling', () => {
    it('should display error message when onDownload throws', async () => {
      const errorMessage = 'Failed to download model: Network error';
      mockOnDownload.mockRejectedValueOnce(new Error(errorMessage));

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(errorMessage)).toBeInTheDocument();
      });
    });

    it('should display generic error message for non-Error objects', async () => {
      mockOnDownload.mockRejectedValueOnce('string error');

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Download failed')).toBeInTheDocument();
      });
    });

    it('should not reset form on error', async () => {
      mockOnDownload.mockRejectedValueOnce(new Error('Download failed'));

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('HuggingFace Model Repository') as HTMLInputElement;
      const tokenInput = screen.getByLabelText(/Access Token/i) as HTMLInputElement;
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(repoInput, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Download failed')).toBeInTheDocument();
      });

      // Form should retain values
      expect(repoInput.value).toBe('TinyLlama/TinyLlama-1.1B');
      expect(tokenInput.value).toBe('hf_token123');
    });

    it('should clear previous error on new submission', async () => {
      mockOnDownload
        .mockRejectedValueOnce(new Error('First error'))
        .mockResolvedValueOnce(undefined);

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      // First submission fails
      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('First error')).toBeInTheDocument();
      });

      // Second submission succeeds
      fireEvent.change(input, { target: { value: 'microsoft/phi-2' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.queryByText('First error')).not.toBeInTheDocument();
      });
    });

    it('should re-enable button after error', async () => {
      mockOnDownload.mockRejectedValueOnce(new Error('Download failed'));

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'TinyLlama/TinyLlama-1.1B' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Download failed')).toBeInTheDocument();
      });

      expect(button).not.toBeDisabled();
    });

    it('should detect trust_remote_code requirement and show helpful message', async () => {
      const trustRemoteCodeError = new Error(
        'Loading Salesforce/CoDA-v0-Instruct requires you to execute the configuration file in that repo on your local machine. Make sure you have read the code there to avoid malicious use, then set the option `trust_remote_code=True` to remove this error.'
      );
      mockOnDownload.mockRejectedValueOnce(trustRemoteCodeError);

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'Salesforce/CoDA-v0-Instruct' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(
          screen.getByText('This model requires executing custom code. Please enable "Trust Remote Code" below and try again.')
        ).toBeInTheDocument();
      });

      expect(mockOnDownload).toHaveBeenCalledWith(
        'Salesforce/CoDA-v0-Instruct',
        QuantizationFormat.Q4,
        undefined,
        false
      );
    });

    it('should detect unsupported architecture and show helpful message', async () => {
      const unsupportedArchError = new Error(
        'Unsupported architecture: CoDA. Supported architectures: falcon, gemma, gemma2, gemma3, gpt2, gpt_neox, lfm2, llama, mistral, mixtral, phi, phi3, phi3_v, pythia, qwen, qwen2, qwen3'
      );
      mockOnDownload.mockRejectedValueOnce(unsupportedArchError);

      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('HuggingFace Model Repository');
      const button = screen.getByRole('button', { name: /^download$/i });

      fireEvent.change(input, { target: { value: 'Salesforce/CoDA-v0-Instruct' } });
      fireEvent.click(button);

      await waitFor(() => {
        const errorText = screen.getByText((content, element) => {
          return content.includes('Unsupported architecture: CoDA') &&
                 content.includes('This model architecture is not yet supported');
        });
        expect(errorText).toBeInTheDocument();
      });

      expect(mockOnDownload).toHaveBeenCalledWith(
        'Salesforce/CoDA-v0-Instruct',
        QuantizationFormat.Q4,
        undefined,
        false
      );
    });
  });

  describe('Accessibility', () => {
    it('should have accessible labels for all inputs', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      expect(screen.getByLabelText('HuggingFace Model Repository')).toBeInTheDocument();
      expect(screen.getByLabelText('Quantization Format')).toBeInTheDocument();
      expect(screen.getByLabelText(/Access Token/i)).toBeInTheDocument();
      expect(screen.getByLabelText('Trust Remote Code')).toBeInTheDocument();
    });

    it('should have proper button with icons', () => {
      render(<ModelDownloadForm onDownload={mockOnDownload} />);

      const previewButton = screen.getByRole('button', { name: /preview/i });
      const downloadButton = screen.getByRole('button', { name: /^download$/i });

      expect(previewButton).toBeInTheDocument();
      expect(downloadButton).toBeInTheDocument();

      // Check for icons (lucide-react renders as svg)
      const svgs = document.querySelectorAll('svg');
      expect(svgs.length).toBeGreaterThan(0);
    });
  });
});

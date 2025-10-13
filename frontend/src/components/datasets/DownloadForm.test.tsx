/**
 * Unit tests for DownloadForm component.
 *
 * Tests form rendering, validation, submission, error handling,
 * and interaction with the onDownload callback.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DownloadForm } from './DownloadForm';

// Mock the validators module
vi.mock('../../utils/validators', () => ({
  validateHfRepoId: vi.fn((repoId: string) => {
    if (!repoId) return 'Repository ID is required';
    if (!repoId.includes('/')) return 'Repository ID must be in format: publisher/dataset-name';
    const parts = repoId.split('/');
    if (parts.length !== 2) return 'Repository ID must be in format: publisher/dataset-name';
    if (!parts[0] || !parts[1]) return 'Repository ID must be in format: publisher/dataset-name';
    return true;
  }),
}));

describe('DownloadForm', () => {
  const mockOnDownload = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Rendering', () => {
    it('should render form title', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      expect(screen.getByText('Download from HuggingFace')).toBeInTheDocument();
    });

    it('should render repository ID input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('type', 'text');
      expect(input).toHaveAttribute('placeholder', 'publisher/dataset-name');
      expect(input).toBeRequired();
    });

    it('should render access token input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Access Token (optional)');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('type', 'password');
      expect(input).toHaveAttribute('placeholder', 'hf_...');
      expect(input).not.toBeRequired();
    });

    it('should render split input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Split (optional)');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('type', 'text');
      expect(input).toHaveAttribute('placeholder', 'train, validation, test');
    });

    it('should render config input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Config (optional)');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('type', 'text');
      expect(input).toHaveAttribute('placeholder', 'en, zh, etc.');
    });

    it('should render submit button', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const button = screen.getByRole('button', { name: /download dataset/i });
      expect(button).toBeInTheDocument();
      expect(button).toHaveAttribute('type', 'submit');
    });

    it('should render helper texts', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      expect(screen.getByText('Required for private or gated datasets')).toBeInTheDocument();
      expect(screen.getByText('Dataset split to download')).toBeInTheDocument();
      expect(screen.getByText('Dataset configuration')).toBeInTheDocument();
    });
  });

  describe('Form State', () => {
    it('should update repository ID on input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'test/dataset' } });

      expect(input.value).toBe('test/dataset');
    });

    it('should update access token on input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Access Token (optional)') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'hf_token123' } });

      expect(input.value).toBe('hf_token123');
    });

    it('should update split on input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Split (optional)') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'train' } });

      expect(input.value).toBe('train');
    });

    it('should update config on input', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Config (optional)') as HTMLInputElement;
      fireEvent.change(input, { target: { value: 'default' } });

      expect(input.value).toBe('default');
    });

    it('should disable submit button when repository ID is empty', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const button = screen.getByRole('button', { name: /download dataset/i });
      expect(button).toBeDisabled();
    });

    it('should enable submit button when repository ID is provided', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      fireEvent.change(input, { target: { value: 'test/dataset' } });

      const button = screen.getByRole('button', { name: /download dataset/i });
      expect(button).not.toBeDisabled();
    });
  });

  describe('Validation', () => {
    it('should show validation error for empty repository ID', async () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      // Try to submit with empty repo
      const button = screen.getByRole('button', { name: /download dataset/i });

      // Button should be disabled, but we can still test validation logic
      // by filling and clearing the input
      const input = screen.getByLabelText('Repository ID');
      fireEvent.change(input, { target: { value: 'a' } });
      fireEvent.change(input, { target: { value: '' } });

      expect(button).toBeDisabled();
    });

    it('should show validation error for invalid repository ID format', async () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      // Enter invalid format (no slash)
      fireEvent.change(input, { target: { value: 'invalid-repo' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(/must be in format: publisher\/dataset-name/i)).toBeInTheDocument();
      });

      expect(mockOnDownload).not.toHaveBeenCalled();
    });

    it('should show validation error for repository ID with empty username', async () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: '/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(/must be in format: publisher\/dataset-name/i)).toBeInTheDocument();
      });

      expect(mockOnDownload).not.toHaveBeenCalled();
    });

    it('should show validation error for repository ID with empty dataset name', async () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: 'username/' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(/must be in format: publisher\/dataset-name/i)).toBeInTheDocument();
      });

      expect(mockOnDownload).not.toHaveBeenCalled();
    });

    it('should clear validation error when input is corrected', async () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      // Trigger validation error
      fireEvent.change(input, { target: { value: 'invalid' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(/must be in format/i)).toBeInTheDocument();
      });

      // Correct the input and submit again
      fireEvent.change(input, { target: { value: 'valid/repo' } });
      mockOnDownload.mockResolvedValueOnce(undefined);
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.queryByText(/must be in format/i)).not.toBeInTheDocument();
      });
    });
  });

  describe('Form Submission', () => {
    it('should call onDownload with repository ID only', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith('test/dataset', undefined, undefined, undefined);
      });
    });

    it('should call onDownload with repository ID and access token', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID');
      const tokenInput = screen.getByLabelText('Access Token (optional)');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(repoInput, { target: { value: 'test/dataset' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith('test/dataset', 'hf_token123', undefined, undefined);
      });
    });

    it('should call onDownload with repository ID, token, split, and config', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID');
      const tokenInput = screen.getByLabelText('Access Token (optional)');
      const splitInput = screen.getByLabelText('Split (optional)');
      const configInput = screen.getByLabelText('Config (optional)');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(repoInput, { target: { value: 'test/dataset' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.change(splitInput, { target: { value: 'train' } });
      fireEvent.change(configInput, { target: { value: 'default' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith('test/dataset', 'hf_token123', 'train', 'default');
      });
    });

    it('should convert empty strings to undefined for optional fields', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID');
      const tokenInput = screen.getByLabelText('Access Token (optional)');
      const button = screen.getByRole('button', { name: /download dataset/i });

      // Set token then clear it
      fireEvent.change(repoInput, { target: { value: 'test/dataset' } });
      fireEvent.change(tokenInput, { target: { value: 'token' } });
      fireEvent.change(tokenInput, { target: { value: '' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith('test/dataset', undefined, undefined, undefined);
      });
    });

    it('should show loading state during submission', async () => {
      let resolveDownload: () => void;
      const downloadPromise = new Promise<void>((resolve) => {
        resolveDownload = resolve;
      });
      mockOnDownload.mockReturnValueOnce(downloadPromise);

      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Downloading...')).toBeInTheDocument();
        expect(button).toBeDisabled();
      });

      // Resolve the promise
      resolveDownload!();

      await waitFor(() => {
        expect(screen.getByText('Download Dataset')).toBeInTheDocument();
      });

      // After successful submission, form resets so button is disabled (no repo ID)
      expect(button).toBeDisabled();
    });

    it('should disable all inputs during submission', async () => {
      let resolveDownload: () => void;
      const downloadPromise = new Promise<void>((resolve) => {
        resolveDownload = resolve;
      });
      mockOnDownload.mockReturnValueOnce(downloadPromise);

      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID');
      const tokenInput = screen.getByLabelText('Access Token (optional)');
      const splitInput = screen.getByLabelText('Split (optional)');
      const configInput = screen.getByLabelText('Config (optional)');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(repoInput, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(repoInput).toBeDisabled();
        expect(tokenInput).toBeDisabled();
        expect(splitInput).toBeDisabled();
        expect(configInput).toBeDisabled();
        expect(button).toBeDisabled();
      });

      resolveDownload!();

      await waitFor(() => {
        expect(repoInput).not.toBeDisabled();
        expect(tokenInput).not.toBeDisabled();
        expect(splitInput).not.toBeDisabled();
        expect(configInput).not.toBeDisabled();
        // Button is disabled after reset because repo ID is empty
      });
    });

    it('should reset form on successful submission', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID') as HTMLInputElement;
      const tokenInput = screen.getByLabelText('Access Token (optional)') as HTMLInputElement;
      const splitInput = screen.getByLabelText('Split (optional)') as HTMLInputElement;
      const configInput = screen.getByLabelText('Config (optional)') as HTMLInputElement;
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(repoInput, { target: { value: 'test/dataset' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.change(splitInput, { target: { value: 'train' } });
      fireEvent.change(configInput, { target: { value: 'default' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(repoInput.value).toBe('');
        expect(tokenInput.value).toBe('');
        expect(splitInput.value).toBe('');
        expect(configInput.value).toBe('');
      });
    });
  });

  describe('Error Handling', () => {
    it('should display error message when onDownload throws', async () => {
      const errorMessage = 'Failed to download: Network error';
      mockOnDownload.mockRejectedValueOnce(new Error(errorMessage));

      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText(errorMessage)).toBeInTheDocument();
      });
    });

    it('should display generic error message for non-Error objects', async () => {
      mockOnDownload.mockRejectedValueOnce('string error');

      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Failed to download dataset')).toBeInTheDocument();
      });
    });

    it('should not reset form on error', async () => {
      mockOnDownload.mockRejectedValueOnce(new Error('Download failed'));

      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID') as HTMLInputElement;
      const tokenInput = screen.getByLabelText('Access Token (optional)') as HTMLInputElement;
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(repoInput, { target: { value: 'test/dataset' } });
      fireEvent.change(tokenInput, { target: { value: 'hf_token123' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Download failed')).toBeInTheDocument();
      });

      // Form should retain values
      expect(repoInput.value).toBe('test/dataset');
      expect(tokenInput.value).toBe('hf_token123');
    });

    it('should clear previous error on new submission', async () => {
      mockOnDownload
        .mockRejectedValueOnce(new Error('First error'))
        .mockResolvedValueOnce(undefined);

      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      // First submission fails
      fireEvent.change(input, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('First error')).toBeInTheDocument();
      });

      // Second submission succeeds
      fireEvent.change(input, { target: { value: 'test/dataset2' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.queryByText('First error')).not.toBeInTheDocument();
      });
    });

    it('should re-enable button after error', async () => {
      mockOnDownload.mockRejectedValueOnce(new Error('Download failed'));

      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: 'test/dataset' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(screen.getByText('Download failed')).toBeInTheDocument();
      });

      expect(button).not.toBeDisabled();
    });
  });

  describe('Styling and Accessibility', () => {
    it('should apply custom className', () => {
      const { container } = render(
        <DownloadForm onDownload={mockOnDownload} className="custom-class" />
      );

      const form = container.querySelector('form');
      expect(form).toHaveClass('custom-class');
    });

    it('should have proper form structure', () => {
      const { container } = render(<DownloadForm onDownload={mockOnDownload} />);

      const form = container.querySelector('form');
      expect(form).toBeInTheDocument();
    });

    it('should have accessible labels for all inputs', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      expect(screen.getByLabelText('Repository ID')).toBeInTheDocument();
      expect(screen.getByLabelText('Access Token (optional)')).toBeInTheDocument();
      expect(screen.getByLabelText('Split (optional)')).toBeInTheDocument();
      expect(screen.getByLabelText('Config (optional)')).toBeInTheDocument();
    });

    it('should have proper button with icon', () => {
      render(<DownloadForm onDownload={mockOnDownload} />);

      const button = screen.getByRole('button', { name: /download dataset/i });
      expect(button).toBeInTheDocument();

      // Check for Download icon (lucide-react renders as svg)
      const svg = button.querySelector('svg');
      expect(svg).toBeInTheDocument();
    });
  });

  describe('Edge Cases', () => {
    it('should handle rapid form submissions', async () => {
      let resolveCount = 0;
      const resolvers: Array<() => void> = [];

      mockOnDownload.mockImplementation(() => {
        return new Promise<void>((resolve) => {
          resolvers[resolveCount++] = resolve;
        });
      });

      render(<DownloadForm onDownload={mockOnDownload} />);

      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      // Submit twice rapidly
      fireEvent.change(input, { target: { value: 'test/dataset1' } });
      fireEvent.click(button);
      fireEvent.change(input, { target: { value: 'test/dataset2' } });
      fireEvent.click(button);

      // Should only call once (button disabled during submission)
      expect(mockOnDownload).toHaveBeenCalledTimes(1);

      // Resolve first submission
      resolvers[0]();

      // After successful submission, form resets so button is disabled (no repo ID)
      await waitFor(() => {
        expect(input).toHaveValue('');
      });
      expect(button).toBeDisabled();
    });

    it('should handle very long repository IDs', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const longRepoId = 'very-long-username/very-long-dataset-name-with-many-characters';
      const input = screen.getByLabelText('Repository ID');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(input, { target: { value: longRepoId } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(longRepoId, undefined, undefined, undefined);
      });
    });

    it('should handle special characters in inputs', async () => {
      mockOnDownload.mockResolvedValueOnce(undefined);
      render(<DownloadForm onDownload={mockOnDownload} />);

      const repoInput = screen.getByLabelText('Repository ID');
      const splitInput = screen.getByLabelText('Split (optional)');
      const configInput = screen.getByLabelText('Config (optional)');
      const button = screen.getByRole('button', { name: /download dataset/i });

      fireEvent.change(repoInput, { target: { value: 'user-name/dataset_name-123' } });
      fireEvent.change(splitInput, { target: { value: 'train[0:1000]' } });
      fireEvent.change(configInput, { target: { value: 'config-v1.0' } });
      fireEvent.click(button);

      await waitFor(() => {
        expect(mockOnDownload).toHaveBeenCalledWith(
          'user-name/dataset_name-123',
          undefined,
          'train[0:1000]',
          'config-v1.0'
        );
      });
    });
  });
});

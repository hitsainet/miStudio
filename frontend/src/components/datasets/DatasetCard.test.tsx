/**
 * Unit tests for DatasetCard component.
 *
 * Tests rendering, status-based behavior, interactions, and conditional displays.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { DatasetCard } from './DatasetCard';
import { DatasetStatus } from '../../types/dataset';
import type { Dataset } from '../../types/dataset';

// Mock child components to isolate DatasetCard logic
vi.mock('../common/StatusBadge', () => ({
  StatusBadge: ({ status }: { status: string }) => (
    <span data-testid="status-badge">{status}</span>
  ),
}));

vi.mock('../common/ProgressBar', () => ({
  ProgressBar: ({ progress }: { progress: number }) => (
    <div data-testid="progress-bar" data-progress={progress}>
      Progress: {progress}%
    </div>
  ),
}));

// Mock formatFileSize utility
vi.mock('../../utils/formatters', () => ({
  formatFileSize: (bytes: number) => `${(bytes / 1024 / 1024).toFixed(2)} MB`,
}));

describe('DatasetCard', () => {
  // Mock window.confirm
  const originalConfirm = window.confirm;

  beforeEach(() => {
    vi.clearAllMocks();
    window.confirm = vi.fn();
  });

  afterEach(() => {
    window.confirm = originalConfirm;
    vi.restoreAllMocks();
  });

  const createMockDataset = (overrides?: Partial<Dataset>): Dataset => ({
    id: 'test-dataset-1',
    name: 'Test Dataset',
    source: 'huggingface',
    hf_repo_id: 'test/dataset',
    status: DatasetStatus.READY,
    progress: 100,
    created_at: new Date().toISOString(),
    ...overrides,
  });

  describe('Basic Rendering', () => {
    it('should render dataset name', () => {
      const dataset = createMockDataset();
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText('Test Dataset')).toBeInTheDocument();
    });

    it('should render dataset source', () => {
      const dataset = createMockDataset();
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText(/Source: huggingface/)).toBeInTheDocument();
    });

    it('should render HuggingFace repo ID', () => {
      const dataset = createMockDataset();
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText(/test\/dataset/)).toBeInTheDocument();
    });

    it('should render without HuggingFace repo ID', () => {
      const dataset = createMockDataset({ hf_repo_id: undefined });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText(/Source: huggingface/)).toBeInTheDocument();
      expect(screen.queryByText(/test\/dataset/)).not.toBeInTheDocument();
    });

    it('should render StatusBadge component', () => {
      const dataset = createMockDataset();
      render(<DatasetCard dataset={dataset} />);

      const badge = screen.getByTestId('status-badge');
      expect(badge).toBeInTheDocument();
      expect(badge).toHaveTextContent('ready');
    });

    it('should render Database icon', () => {
      const dataset = createMockDataset();
      const { container } = render(<DatasetCard dataset={dataset} />);

      // Check for the main Database icon (the large one)
      const databaseIcon = container.querySelector('svg[class*="w-8 h-8"]');
      expect(databaseIcon).toBeInTheDocument();
    });
  });

  describe('Status-Based Behavior', () => {
    it('should be clickable when status is ready', () => {
      const dataset = createMockDataset({ status: DatasetStatus.READY });
      const mockOnClick = vi.fn();
      const { container } = render(<DatasetCard dataset={dataset} onClick={mockOnClick} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('cursor-pointer');

      fireEvent.click(card);
      expect(mockOnClick).toHaveBeenCalledTimes(1);
    });

    it('should not be clickable when status is downloading', () => {
      const dataset = createMockDataset({ status: DatasetStatus.DOWNLOADING });
      const mockOnClick = vi.fn();
      const { container } = render(<DatasetCard dataset={dataset} onClick={mockOnClick} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('cursor-default');

      fireEvent.click(card);
      expect(mockOnClick).not.toHaveBeenCalled();
    });

    it('should not be clickable when status is processing', () => {
      const dataset = createMockDataset({ status: DatasetStatus.PROCESSING });
      const mockOnClick = vi.fn();
      const { container } = render(<DatasetCard dataset={dataset} onClick={mockOnClick} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('cursor-default');

      fireEvent.click(card);
      expect(mockOnClick).not.toHaveBeenCalled();
    });

    it('should not be clickable when status is error', () => {
      const dataset = createMockDataset({ status: DatasetStatus.ERROR });
      const mockOnClick = vi.fn();
      const { container } = render(<DatasetCard dataset={dataset} onClick={mockOnClick} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('cursor-default');

      fireEvent.click(card);
      expect(mockOnClick).not.toHaveBeenCalled();
    });
  });

  describe('Progress Display', () => {
    it('should show progress bar when downloading', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.DOWNLOADING,
        progress: 45,
      });
      render(<DatasetCard dataset={dataset} />);

      const progressBar = screen.getByTestId('progress-bar');
      expect(progressBar).toBeInTheDocument();
      expect(progressBar).toHaveAttribute('data-progress', '45');
    });

    it('should show progress bar when processing', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.PROCESSING,
        progress: 75,
      });
      render(<DatasetCard dataset={dataset} />);

      const progressBar = screen.getByTestId('progress-bar');
      expect(progressBar).toBeInTheDocument();
      expect(progressBar).toHaveAttribute('data-progress', '75');
    });

    it('should not show progress bar when ready', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.READY,
        progress: 100,
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByTestId('progress-bar')).not.toBeInTheDocument();
    });

    it('should not show progress bar when error', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.ERROR,
        progress: 50,
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByTestId('progress-bar')).not.toBeInTheDocument();
    });

    it('should not show progress bar when progress is undefined', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.DOWNLOADING,
        progress: undefined,
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByTestId('progress-bar')).not.toBeInTheDocument();
    });
  });

  describe('Delete Functionality', () => {
    it('should render delete button when onDelete provided', () => {
      const dataset = createMockDataset();
      const mockOnDelete = vi.fn();
      render(<DatasetCard dataset={dataset} onDelete={mockOnDelete} />);

      const deleteButton = screen.getByTitle('Delete dataset');
      expect(deleteButton).toBeInTheDocument();
    });

    it('should not render delete button when onDelete not provided', () => {
      const dataset = createMockDataset();
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByTitle('Delete dataset')).not.toBeInTheDocument();
    });

    it('should call onDelete when confirmed', () => {
      const dataset = createMockDataset();
      const mockOnDelete = vi.fn();
      (window.confirm as any).mockReturnValue(true);

      render(<DatasetCard dataset={dataset} onDelete={mockOnDelete} />);

      const deleteButton = screen.getByTitle('Delete dataset');
      fireEvent.click(deleteButton);

      expect(window.confirm).toHaveBeenCalledWith(
        'Are you sure you want to delete "Test Dataset"? This will remove all downloaded files.'
      );
      expect(mockOnDelete).toHaveBeenCalledWith('test-dataset-1');
    });

    it('should not call onDelete when cancelled', () => {
      const dataset = createMockDataset();
      const mockOnDelete = vi.fn();
      (window.confirm as any).mockReturnValue(false);

      render(<DatasetCard dataset={dataset} onDelete={mockOnDelete} />);

      const deleteButton = screen.getByTitle('Delete dataset');
      fireEvent.click(deleteButton);

      expect(window.confirm).toHaveBeenCalled();
      expect(mockOnDelete).not.toHaveBeenCalled();
    });

    it('should stop propagation to prevent card click', () => {
      const dataset = createMockDataset({ status: DatasetStatus.READY });
      const mockOnClick = vi.fn();
      const mockOnDelete = vi.fn();
      (window.confirm as any).mockReturnValue(true);

      render(<DatasetCard dataset={dataset} onClick={mockOnClick} onDelete={mockOnDelete} />);

      const deleteButton = screen.getByTitle('Delete dataset');
      fireEvent.click(deleteButton);

      expect(mockOnDelete).toHaveBeenCalled();
      expect(mockOnClick).not.toHaveBeenCalled();
    });
  });

  describe('File Size Display', () => {
    it('should display file size when size_bytes is provided', () => {
      const dataset = createMockDataset({ size_bytes: 104857600 }); // 100 MB
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText(/Size: 100\.00 MB/)).toBeInTheDocument();
    });

    it('should not display file size when size_bytes is 0', () => {
      const dataset = createMockDataset({ size_bytes: 0 });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByText(/Size:/)).not.toBeInTheDocument();
    });

    it('should not display file size when size_bytes is undefined', () => {
      const dataset = createMockDataset({ size_bytes: undefined });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByText(/Size:/)).not.toBeInTheDocument();
    });
  });

  describe('Sample Count Display', () => {
    it('should display sample count when num_samples is provided', () => {
      const dataset = createMockDataset({ num_samples: 1000000 });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText(/Samples: 1,000,000/)).toBeInTheDocument();
    });

    it('should not display sample count when num_samples is 0', () => {
      const dataset = createMockDataset({ num_samples: 0 });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByText(/Samples:/)).not.toBeInTheDocument();
    });

    it('should not display sample count when num_samples is undefined', () => {
      const dataset = createMockDataset({ num_samples: undefined });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByText(/Samples:/)).not.toBeInTheDocument();
    });

    it('should format sample count with locale string', () => {
      const dataset = createMockDataset({ num_samples: 5432 });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText(/Samples: 5,432/)).toBeInTheDocument();
    });
  });

  describe('Tokenization Indicator', () => {
    it('should show tokenized badge when tokenization metadata exists', () => {
      const dataset = createMockDataset({
        metadata: {
          tokenization: {
            model: 'gpt2',
            max_length: 512,
          },
        },
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText('Tokenized')).toBeInTheDocument();
    });

    it('should not show tokenized badge when tokenization metadata is missing', () => {
      const dataset = createMockDataset({
        metadata: {},
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByText('Tokenized')).not.toBeInTheDocument();
    });

    it('should not show tokenized badge when metadata is undefined', () => {
      const dataset = createMockDataset({
        metadata: undefined,
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.queryByText('Tokenized')).not.toBeInTheDocument();
    });
  });

  describe('Error Message Display', () => {
    it('should display error message when error_message exists', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.ERROR,
        error_message: 'Failed to download dataset',
      });
      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText('Failed to download dataset')).toBeInTheDocument();
    });

    it('should not display error message when error_message is null', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.ERROR,
        error_message: null,
      });
      render(<DatasetCard dataset={dataset} />);

      // Should not have any error message div
      const { container } = render(<DatasetCard dataset={dataset} />);
      const errorDiv = container.querySelector('[class*="bg-red-500/10"]');
      expect(errorDiv).not.toBeInTheDocument();
    });

    it('should not display error message when error_message is undefined', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.ERROR,
        error_message: undefined,
      });
      render(<DatasetCard dataset={dataset} />);

      const { container } = render(<DatasetCard dataset={dataset} />);
      const errorDiv = container.querySelector('[class*="bg-red-500/10"]');
      expect(errorDiv).not.toBeInTheDocument();
    });
  });

  describe('Status Icon Logic', () => {
    it('should use CheckCircle icon for ready status', () => {
      const dataset = createMockDataset({ status: DatasetStatus.READY });
      const { container } = render(<DatasetCard dataset={dataset} />);

      // The status icon should not have animate-spin class
      const statusIcons = container.querySelectorAll('svg[class*="w-5 h-5"]');
      const hasCheckCircle = Array.from(statusIcons).some(
        (icon) => {
          const className = icon.getAttribute('class') || '';
          return !className.includes('animate-spin');
        }
      );
      expect(hasCheckCircle).toBe(true);
    });

    it('should use animated Loader icon for downloading status', () => {
      const dataset = createMockDataset({ status: DatasetStatus.DOWNLOADING });
      const { container } = render(<DatasetCard dataset={dataset} />);

      // The status icon should have animate-spin class
      const statusIcons = container.querySelectorAll('svg[class*="w-5 h-5"]');
      const hasAnimatedLoader = Array.from(statusIcons).some((icon) => {
        const className = icon.getAttribute('class') || '';
        return className.includes('animate-spin');
      });
      expect(hasAnimatedLoader).toBe(true);
    });

    it('should use Activity icon for processing status', () => {
      const dataset = createMockDataset({ status: DatasetStatus.PROCESSING });
      const { container } = render(<DatasetCard dataset={dataset} />);

      // Should not have animate-spin
      const statusIcons = container.querySelectorAll('svg[class*="w-5 h-5"]');
      const hasAnimated = Array.from(statusIcons).some((icon) => {
        const className = icon.getAttribute('class') || '';
        return className.includes('animate-spin');
      });
      expect(hasAnimated).toBe(false);
    });

    it('should use Activity icon for error status', () => {
      const dataset = createMockDataset({ status: DatasetStatus.ERROR });
      const { container } = render(<DatasetCard dataset={dataset} />);

      // Should not have animate-spin
      const statusIcons = container.querySelectorAll('svg[class*="w-5 h-5"]');
      const hasAnimated = Array.from(statusIcons).some((icon) => {
        const className = icon.getAttribute('class') || '';
        return className.includes('animate-spin');
      });
      expect(hasAnimated).toBe(false);
    });
  });

  describe('Hover States', () => {
    it('should have hover classes when clickable', () => {
      const dataset = createMockDataset({ status: DatasetStatus.READY });
      const { container } = render(<DatasetCard dataset={dataset} onClick={vi.fn()} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('hover:bg-slate-900/70');
      expect(card.className).toContain('hover:border-slate-700');
    });

    it('should not have hover classes when not clickable', () => {
      const dataset = createMockDataset({ status: DatasetStatus.DOWNLOADING });
      const { container } = render(<DatasetCard dataset={dataset} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).not.toContain('hover:bg-slate-900/70');
      expect(card.className).not.toContain('hover:border-slate-700');
    });
  });

  describe('Edge Cases', () => {
    it('should handle very long dataset names with truncation', () => {
      const dataset = createMockDataset({
        name: 'This is a very long dataset name that should be truncated in the UI to prevent layout issues',
      });
      const { container } = render(<DatasetCard dataset={dataset} />);

      const nameElement = container.querySelector('h3');
      expect(nameElement?.className).toContain('truncate');
    });

    it('should handle very long repo IDs with truncation', () => {
      const dataset = createMockDataset({
        hf_repo_id: 'organization-with-very-long-name/dataset-with-extremely-long-name-that-should-truncate',
      });
      const { container } = render(<DatasetCard dataset={dataset} />);

      const sourceElement = container.querySelector('p.text-sm.text-slate-400');
      expect(sourceElement?.className).toContain('truncate');
    });

    it('should handle all optional props being undefined', () => {
      const dataset: Dataset = {
        id: 'minimal-dataset',
        name: 'Minimal Dataset',
        source: 'local',
        status: DatasetStatus.READY,
        progress: 100,
        created_at: new Date().toISOString(),
        hf_repo_id: undefined,
        size_bytes: undefined,
        num_samples: undefined,
        metadata: undefined,
        error_message: undefined,
      };

      expect(() => render(<DatasetCard dataset={dataset} />)).not.toThrow();
      expect(screen.getByText('Minimal Dataset')).toBeInTheDocument();
    });

    it('should handle missing onClick callback', () => {
      const dataset = createMockDataset({ status: DatasetStatus.READY });
      const { container } = render(<DatasetCard dataset={dataset} />);

      const card = container.firstChild as HTMLElement;
      expect(() => fireEvent.click(card)).not.toThrow();
    });

    it('should handle 0 progress correctly', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.DOWNLOADING,
        progress: 0,
      });
      render(<DatasetCard dataset={dataset} />);

      const progressBar = screen.getByTestId('progress-bar');
      expect(progressBar).toHaveAttribute('data-progress', '0');
    });

    it('should handle 100 progress correctly', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.DOWNLOADING,
        progress: 100,
      });
      render(<DatasetCard dataset={dataset} />);

      const progressBar = screen.getByTestId('progress-bar');
      expect(progressBar).toHaveAttribute('data-progress', '100');
    });

    it('should normalize status strings correctly', () => {
      // Test with uppercase status
      const dataset = createMockDataset({
        status: 'READY' as any,
      });
      const { container } = render(<DatasetCard dataset={dataset} />);

      const card = container.firstChild as HTMLElement;
      expect(card.className).toContain('cursor-pointer');
    });
  });

  describe('Complex Scenarios', () => {
    it('should render fully featured dataset card', () => {
      const dataset = createMockDataset({
        name: 'Complete Dataset',
        source: 'huggingface',
        hf_repo_id: 'openai/gsm8k',
        status: DatasetStatus.READY,
        progress: 100,
        size_bytes: 52428800, // 50 MB
        num_samples: 7473,
        metadata: {
          tokenization: {
            model: 'gpt2',
            max_length: 512,
          },
        },
      });

      render(<DatasetCard dataset={dataset} onDelete={vi.fn()} onClick={vi.fn()} />);

      expect(screen.getByText('Complete Dataset')).toBeInTheDocument();
      expect(screen.getByText(/Source: huggingface/)).toBeInTheDocument();
      expect(screen.getByText(/openai\/gsm8k/)).toBeInTheDocument();
      expect(screen.getByText(/Size: 50\.00 MB/)).toBeInTheDocument();
      expect(screen.getByText(/Samples: 7,473/)).toBeInTheDocument();
      expect(screen.getByText('Tokenized')).toBeInTheDocument();
      expect(screen.getByTitle('Delete dataset')).toBeInTheDocument();
    });

    it('should render downloading dataset with progress', () => {
      const dataset = createMockDataset({
        name: 'Downloading Dataset',
        status: DatasetStatus.DOWNLOADING,
        progress: 67,
        size_bytes: 104857600,
      });

      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText('Downloading Dataset')).toBeInTheDocument();
      expect(screen.getByTestId('progress-bar')).toBeInTheDocument();
      expect(screen.queryByText('Tokenized')).not.toBeInTheDocument();
    });

    it('should render error dataset with message', () => {
      const dataset = createMockDataset({
        name: 'Failed Dataset',
        status: DatasetStatus.ERROR,
        error_message: 'Network timeout during download',
        progress: 45,
      });

      render(<DatasetCard dataset={dataset} />);

      expect(screen.getByText('Failed Dataset')).toBeInTheDocument();
      expect(screen.getByText('Network timeout during download')).toBeInTheDocument();
      expect(screen.queryByTestId('progress-bar')).not.toBeInTheDocument();
    });
  });
});

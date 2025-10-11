/**
 * Unit tests for DatasetDetailModal component.
 *
 * Tests modal rendering, tab switching, close functionality, and header display.
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, within, waitFor } from '@testing-library/react';
import { DatasetDetailModal } from './DatasetDetailModal';
import { DatasetStatus } from '../../types/dataset';
import type { Dataset } from '../../types/dataset';

// Mock child components
vi.mock('../common/StatusBadge', () => ({
  StatusBadge: ({ status }: { status: string }) => (
    <span data-testid="status-badge">{status}</span>
  ),
}));

vi.mock('../common/ProgressBar', () => ({
  ProgressBar: ({ progress }: { progress: number }) => (
    <div data-testid="progress-bar">{progress}%</div>
  ),
}));

// Mock formatters
vi.mock('../../utils/formatters', () => ({
  formatFileSize: (bytes: number) => `${(bytes / 1024 / 1024).toFixed(2)} MB`,
  formatDateTime: (date: string) => new Date(date).toLocaleDateString(),
}));

// Mock WebSocket hook
vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
    emit: vi.fn(),
    isConnected: false,
  }),
}));

// Mock API base URL
vi.mock('../../config/api', () => ({
  API_BASE_URL: 'http://localhost:8000',
}));

// Mock fetch globally
global.fetch = vi.fn();

describe('DatasetDetailModal', () => {
  const mockOnClose = vi.fn();
  const mockOnDatasetUpdate = vi.fn();

  const createMockDataset = (overrides?: Partial<Dataset>): Dataset => ({
    id: 'test-dataset-1',
    name: 'Test Dataset',
    source: 'huggingface',
    hf_repo_id: 'test/dataset',
    status: DatasetStatus.READY,
    progress: 100,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    num_samples: 1000,
    size_bytes: 10485760, // 10 MB
    ...overrides,
  });

  beforeEach(() => {
    vi.clearAllMocks();
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => ({ data: [], pagination: null }),
    });
  });

  describe('Modal Rendering', () => {
    it('should render modal backdrop', () => {
      const dataset = createMockDataset();
      const { container } = render(
        <DatasetDetailModal dataset={dataset} onClose={mockOnClose} />
      );

      const backdrop = container.querySelector('.fixed.inset-0');
      expect(backdrop).toBeInTheDocument();
      expect(backdrop?.className).toContain('bg-black/50');
    });

    it('should render modal container', () => {
      const dataset = createMockDataset();
      const { container } = render(
        <DatasetDetailModal dataset={dataset} onClose={mockOnClose} />
      );

      const modal = container.querySelector('.bg-slate-900');
      expect(modal).toBeInTheDocument();
      expect(modal?.className).toContain('max-w-6xl');
    });

    it('should render dataset name in header', () => {
      const dataset = createMockDataset({ name: 'My Special Dataset' });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByText('My Special Dataset')).toBeInTheDocument();
    });

    it('should render close button', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const closeButton = screen.getByRole('button', { name: '' }); // X icon button
      expect(closeButton).toBeInTheDocument();
    });
  });

  describe('Header Information', () => {
    it('should display status badge', () => {
      const dataset = createMockDataset({ status: DatasetStatus.READY });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const badges = screen.getAllByTestId('status-badge');
      expect(badges.length).toBeGreaterThan(0);
      expect(badges[0]).toHaveTextContent('ready');
    });

    it('should display source', () => {
      const dataset = createMockDataset({ source: 'huggingface' });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByText(/Source: huggingface/)).toBeInTheDocument();
    });

    it('should display number of samples', () => {
      const dataset = createMockDataset({ num_samples: 5000 });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByText(/5,000 samples/)).toBeInTheDocument();
    });

    it('should display file size', () => {
      const dataset = createMockDataset({ size_bytes: 104857600 }); // 100 MB
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const fileSizeElements = screen.getAllByText(/100\.00 MB/);
      expect(fileSizeElements.length).toBeGreaterThan(0);
    });

    it('should not display samples when undefined', () => {
      const dataset = createMockDataset({ num_samples: undefined });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.queryByText(/samples/)).not.toBeInTheDocument();
    });

    it('should not display file size when undefined', () => {
      const dataset = createMockDataset({ size_bytes: undefined });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      // Should not find any MB text in header (might be in tabs)
      const header = screen.getByText('Test Dataset').parentElement?.parentElement;
      expect(header?.textContent).not.toMatch(/MB/);
    });
  });

  describe('Tab Navigation', () => {
    it('should render all four tabs', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByRole('button', { name: /Overview/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Samples/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Tokenization/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Statistics/i })).toBeInTheDocument();
    });

    it('should have Overview tab active by default', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const overviewTab = screen.getByRole('button', { name: /Overview/i });
      expect(overviewTab.className).toContain('border-emerald-500');
      expect(overviewTab.className).toContain('text-emerald-400');
    });

    it('should switch to Samples tab when clicked', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const samplesTab = screen.getByRole('button', { name: /Samples/i });
      fireEvent.click(samplesTab);

      expect(samplesTab.className).toContain('border-emerald-500');
      expect(samplesTab.className).toContain('text-emerald-400');
    });

    it('should switch to Tokenization tab when clicked', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const tokenizationTab = screen.getByRole('button', { name: /Tokenization/i });
      fireEvent.click(tokenizationTab);

      expect(tokenizationTab.className).toContain('border-emerald-500');
      expect(tokenizationTab.className).toContain('text-emerald-400');
    });

    it('should switch to Statistics tab when clicked', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const statisticsTab = screen.getByRole('button', { name: /Statistics/i });
      fireEvent.click(statisticsTab);

      expect(statisticsTab.className).toContain('border-emerald-500');
      expect(statisticsTab.className).toContain('text-emerald-400');
    });

    it('should deactivate previous tab when switching', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const overviewTab = screen.getByRole('button', { name: /Overview/i });
      const samplesTab = screen.getByRole('button', { name: /Samples/i });

      // Overview should be active initially
      expect(overviewTab.className).toContain('border-emerald-500');

      // Click Samples tab
      fireEvent.click(samplesTab);

      // Overview should no longer be active
      expect(overviewTab.className).toContain('border-transparent');
      expect(overviewTab.className).not.toContain('border-emerald-500');
    });

    it('should switch between multiple tabs', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const overviewTab = screen.getByRole('button', { name: /Overview/i });
      const samplesTab = screen.getByRole('button', { name: /Samples/i });
      const statisticsTab = screen.getByRole('button', { name: /Statistics/i });

      // Start with Overview active
      expect(overviewTab.className).toContain('border-emerald-500');

      // Switch to Samples
      fireEvent.click(samplesTab);
      expect(samplesTab.className).toContain('border-emerald-500');
      expect(overviewTab.className).toContain('border-transparent');

      // Switch to Statistics
      fireEvent.click(statisticsTab);
      expect(statisticsTab.className).toContain('border-emerald-500');
      expect(samplesTab.className).toContain('border-transparent');

      // Switch back to Overview
      fireEvent.click(overviewTab);
      expect(overviewTab.className).toContain('border-emerald-500');
      expect(statisticsTab.className).toContain('border-transparent');
    });
  });

  describe('Tab Content Rendering', () => {
    it('should render Overview tab content by default', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      // Overview tab shows dataset ID
      expect(screen.getByText('Dataset ID')).toBeInTheDocument();
      expect(screen.getByText(dataset.id)).toBeInTheDocument();
    });

    it('should render Samples tab content when selected', async () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const samplesTab = screen.getByRole('button', { name: /Samples/i });
      fireEvent.click(samplesTab);

      // Should show loading state
      await waitFor(() => {
        expect(screen.getByText(/Loading samples/i)).toBeInTheDocument();
      });
    });

    it('should render Statistics tab content when selected', async () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const statisticsTab = screen.getByRole('button', { name: /Statistics/i });
      fireEvent.click(statisticsTab);

      // Should show "no statistics" message since dataset has no tokenization metadata
      await waitFor(() => {
        expect(screen.getByText(/No tokenization statistics available/i)).toBeInTheDocument();
      });
    });

    it('should render Tokenization tab content when selected', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const tokenizationTab = screen.getByRole('button', { name: /Tokenization/i });
      fireEvent.click(tokenizationTab);

      // Should show tokenization form
      expect(screen.getByText(/Tokenize Dataset/i)).toBeInTheDocument();
      expect(screen.getByText(/Tokenizer Model/i)).toBeInTheDocument();
    });
  });

  describe('Close Functionality', () => {
    it('should call onClose when close button clicked', () => {
      const dataset = createMockDataset();
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      // Find close button (X icon button in header)
      const closeButtons = screen.getAllByRole('button');
      const closeButton = closeButtons.find(
        (btn) => btn.querySelector('.lucide-x') || btn.className.includes('hover:bg-slate-800')
      );

      if (closeButton) {
        fireEvent.click(closeButton);
        expect(mockOnClose).toHaveBeenCalledTimes(1);
      }
    });

    it('should not close when clicking inside modal', () => {
      const dataset = createMockDataset();
      const { container } = render(
        <DatasetDetailModal dataset={dataset} onClose={mockOnClose} />
      );

      const modal = container.querySelector('.bg-slate-900');
      if (modal) {
        fireEvent.click(modal);
        expect(mockOnClose).not.toHaveBeenCalled();
      }
    });
  });

  describe('Tab Icons', () => {
    it('should render icons for all tabs', () => {
      const dataset = createMockDataset();
      const { container } = render(
        <DatasetDetailModal dataset={dataset} onClose={mockOnClose} />
      );

      // Each tab button should have an icon (lucide icon)
      const tabButtons = screen.getAllByRole('button').filter(
        (btn) => btn.textContent?.match(/Overview|Samples|Tokenization|Statistics/)
      );

      tabButtons.forEach((button) => {
        const icon = button.querySelector('svg');
        expect(icon).toBeInTheDocument();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle dataset with minimal data', () => {
      const dataset: Dataset = {
        id: 'minimal',
        name: 'Minimal Dataset',
        source: 'local',
        status: DatasetStatus.READY,
        progress: 100,
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      };

      expect(() =>
        render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />)
      ).not.toThrow();

      expect(screen.getByText('Minimal Dataset')).toBeInTheDocument();
    });

    it('should handle dataset with error status', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.ERROR,
        error_message: 'Download failed',
      });

      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const badges = screen.getAllByTestId('status-badge');
      expect(badges[0]).toHaveTextContent('error');
    });

    it('should handle dataset with tokenization metadata', () => {
      const dataset = createMockDataset({
        metadata: {
          tokenization: {
            tokenizer_name: 'gpt2',
            text_column_used: 'text',
            max_length: 512,
            stride: 0,
            num_tokens: 100000,
            avg_seq_length: 250.5,
            min_seq_length: 10,
            max_seq_length: 512,
          },
        },
      });

      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      // Switch to Tokenization tab
      const tokenizationTab = screen.getByRole('button', { name: /Tokenization/i });
      fireEvent.click(tokenizationTab);

      // Should show "already tokenized" message
      expect(screen.getByText(/Dataset Already Tokenized/i)).toBeInTheDocument();
    });

    it('should handle very long dataset names with truncation', () => {
      const dataset = createMockDataset({
        name: 'This is a very long dataset name that should be truncated to prevent layout issues in the modal header',
      });

      const { container } = render(
        <DatasetDetailModal dataset={dataset} onClose={mockOnClose} />
      );

      const nameElement = container.querySelector('h2');
      expect(nameElement?.className).toContain('truncate');
    });

    it('should handle missing onDatasetUpdate callback', () => {
      const dataset = createMockDataset();

      expect(() =>
        render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />)
      ).not.toThrow();
    });
  });

  describe('Samples Tab States', () => {
    it('should show "not ready" message for non-ready datasets', async () => {
      const dataset = createMockDataset({ status: DatasetStatus.DOWNLOADING });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const samplesTab = screen.getByRole('button', { name: /Samples/i });
      fireEvent.click(samplesTab);

      await waitFor(() => {
        expect(screen.getByText(/Dataset not ready/i)).toBeInTheDocument();
        expect(screen.getByText(/Samples can be viewed once the dataset is in "ready" status/i)).toBeInTheDocument();
      });
    });
  });

  describe('Statistics Tab States', () => {
    it('should show "no statistics" message when no tokenization metadata', () => {
      const dataset = createMockDataset({ metadata: undefined });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const statisticsTab = screen.getByRole('button', { name: /Statistics/i });
      fireEvent.click(statisticsTab);

      expect(screen.getByText(/No tokenization statistics available/i)).toBeInTheDocument();
    });

    it('should show statistics when tokenization metadata exists', () => {
      const dataset = createMockDataset({
        metadata: {
          tokenization: {
            tokenizer_name: 'gpt2',
            text_column_used: 'text',
            max_length: 512,
            stride: 0,
            num_tokens: 100000,
            avg_seq_length: 250.5,
            min_seq_length: 10,
            max_seq_length: 512,
          },
        },
      });

      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      const statisticsTab = screen.getByRole('button', { name: /Statistics/i });
      fireEvent.click(statisticsTab);

      expect(screen.getByText(/Tokenization Configuration/i)).toBeInTheDocument();
      expect(screen.getByText(/Token Statistics/i)).toBeInTheDocument();
    });
  });

  describe('Overview Tab Content', () => {
    it('should display dataset ID in overview', () => {
      const dataset = createMockDataset({ id: 'test-id-123' });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByText('Dataset ID')).toBeInTheDocument();
      expect(screen.getByText('test-id-123')).toBeInTheDocument();
    });

    it('should display HuggingFace repo ID when present', () => {
      const dataset = createMockDataset({ hf_repo_id: 'openai/gsm8k' });
      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByText('HuggingFace Repository')).toBeInTheDocument();
      expect(screen.getByText('openai/gsm8k')).toBeInTheDocument();
    });

    it('should display error message when present', () => {
      const dataset = createMockDataset({
        status: DatasetStatus.ERROR,
        error_message: 'Failed to download: Network timeout',
      });

      render(<DatasetDetailModal dataset={dataset} onClose={mockOnClose} />);

      expect(screen.getByText('Error')).toBeInTheDocument();
      expect(screen.getByText('Failed to download: Network timeout')).toBeInTheDocument();
    });
  });
});

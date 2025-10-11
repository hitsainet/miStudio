/**
 * Unit tests for DatasetsPanel component.
 *
 * Tests rendering, state management integration, loading/error states,
 * user interactions, and child component integration.
 */

import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { DatasetsPanel } from './DatasetsPanel';
import { useDatasetsStore } from '../../stores/datasetsStore';
import { DatasetStatus } from '../../types/dataset';

// Mock the Zustand store
vi.mock('../../stores/datasetsStore');

// Mock child components to isolate DatasetsPanel logic
vi.mock('../datasets/DownloadForm', () => ({
  DownloadForm: ({ onDownload }: { onDownload: Function }) => (
    <div data-testid="download-form">
      <button
        onClick={() =>
          onDownload('test/repo', 'token123', 'train', 'default')
        }
      >
        Download
      </button>
    </div>
  ),
}));

vi.mock('../datasets/DatasetCard', () => ({
  DatasetCard: ({
    dataset,
    onClick,
    onDelete,
  }: {
    dataset: any;
    onClick: Function;
    onDelete: Function;
  }) => (
    <div data-testid={`dataset-card-${dataset.id}`}>
      <span>{dataset.name}</span>
      <button onClick={() => onClick(dataset)}>View</button>
      <button onClick={() => onDelete(dataset.id)}>Delete</button>
    </div>
  ),
}));

vi.mock('../datasets/DatasetDetailModal', () => ({
  DatasetDetailModal: ({
    dataset,
    onClose,
  }: {
    dataset: any;
    onClose: Function;
  }) => (
    <div data-testid="detail-modal">
      <span>{dataset?.name}</span>
      <button onClick={() => onClose()}>Close</button>
    </div>
  ),
}));

describe('DatasetsPanel', () => {
  const mockFetchDatasets = vi.fn();
  const mockDownloadDataset = vi.fn();
  const mockDeleteDataset = vi.fn();

  const mockDatasets = [
    {
      id: 'dataset-1',
      name: 'Test Dataset 1',
      status: DatasetStatus.READY,
      progress: 100,
      created_at: new Date().toISOString(),
      source: 'huggingface',
      hf_repo_id: 'test/dataset-1',
    },
    {
      id: 'dataset-2',
      name: 'Test Dataset 2',
      status: DatasetStatus.DOWNLOADING,
      progress: 50,
      created_at: new Date().toISOString(),
      source: 'huggingface',
      hf_repo_id: 'test/dataset-2',
    },
  ];

  beforeEach(() => {
    // Reset mocks
    vi.clearAllMocks();

    // Default mock implementation
    (useDatasetsStore as any).mockReturnValue({
      datasets: [],
      loading: false,
      error: null,
      fetchDatasets: mockFetchDatasets,
      downloadDataset: mockDownloadDataset,
      deleteDataset: mockDeleteDataset,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Rendering', () => {
    it('should render title and description', () => {
      render(<DatasetsPanel />);

      expect(screen.getByText('Datasets')).toBeInTheDocument();
      expect(
        screen.getByText(
          'Manage training datasets from HuggingFace or local sources'
        )
      ).toBeInTheDocument();
    });

    it('should render DownloadForm component', () => {
      render(<DatasetsPanel />);

      expect(screen.getByTestId('download-form')).toBeInTheDocument();
    });

    it('should call fetchDatasets on mount', () => {
      render(<DatasetsPanel />);

      expect(mockFetchDatasets).toHaveBeenCalledTimes(1);
    });
  });

  describe('Loading State', () => {
    it('should show loading spinner when loading with no datasets', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: true,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(screen.getByText(/loading datasets/i)).toBeInTheDocument();
    });

    it('should not show loading spinner when datasets exist', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: true,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(screen.queryByText(/loading datasets/i)).not.toBeInTheDocument();
    });
  });

  describe('Empty State', () => {
    it('should show empty state when not loading and no datasets', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(
        screen.getByText(/no datasets yet/i)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/download a dataset from huggingface/i)
      ).toBeInTheDocument();
    });

    it('should not show empty state when datasets exist', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(screen.queryByText(/no datasets yet/i)).not.toBeInTheDocument();
    });

    it('should not show empty state when loading', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: true,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(screen.queryByText(/no datasets yet/i)).not.toBeInTheDocument();
    });
  });

  describe('Error State', () => {
    it('should display error message when error exists', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: false,
        error: 'Failed to fetch datasets',
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(screen.getByText('Failed to fetch datasets')).toBeInTheDocument();
    });

    it('should not display error when error is null', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(
        screen.queryByText(/failed to fetch datasets/i)
      ).not.toBeInTheDocument();
    });
  });

  describe('Datasets Display', () => {
    it('should render DatasetCard for each dataset', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      expect(screen.getByTestId('dataset-card-dataset-1')).toBeInTheDocument();
      expect(screen.getByTestId('dataset-card-dataset-2')).toBeInTheDocument();
      expect(screen.getByText('Test Dataset 1')).toBeInTheDocument();
      expect(screen.getByText('Test Dataset 2')).toBeInTheDocument();
    });

    it('should render datasets in a grid layout', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      const { container } = render(<DatasetsPanel />);

      // Check for grid container
      const gridElement = container.querySelector('.grid');
      expect(gridElement).toBeInTheDocument();
    });

  });

  describe('Download Interaction', () => {
    it('should call downloadDataset when download form is submitted', async () => {
      mockDownloadDataset.mockResolvedValueOnce(undefined);

      render(<DatasetsPanel />);

      const downloadButton = screen.getByText('Download');
      fireEvent.click(downloadButton);

      await waitFor(() => {
        expect(mockDownloadDataset).toHaveBeenCalledWith(
          'test/repo',
          'token123',
          'train',
          'default'
        );
      });
    });

    it('should handle download errors gracefully', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      mockDownloadDataset.mockRejectedValueOnce(
        new Error('Download failed')
      );

      render(<DatasetsPanel />);

      const downloadButton = screen.getByText('Download');
      fireEvent.click(downloadButton);

      await waitFor(() => {
        expect(mockDownloadDataset).toHaveBeenCalled();
      });

      // Should not crash the component
      expect(screen.getByTestId('download-form')).toBeInTheDocument();

      consoleErrorSpy.mockRestore();
    });
  });

  describe('Dataset Card Interaction', () => {
    beforeEach(() => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });
    });

    it('should open modal when dataset card is clicked', async () => {
      render(<DatasetsPanel />);

      // Modal should not be present initially
      expect(screen.queryByTestId('detail-modal')).not.toBeInTheDocument();

      const viewButton = screen.getAllByText('View')[0];
      fireEvent.click(viewButton);

      await waitFor(() => {
        const modal = screen.getByTestId('detail-modal');
        expect(modal).toBeInTheDocument();
        // Check the modal contains the dataset name
        expect(modal).toHaveTextContent('Test Dataset 1');
      });
    });

    it('should close modal when close button is clicked', async () => {
      render(<DatasetsPanel />);

      // Open modal
      const viewButton = screen.getAllByText('View')[0];
      fireEvent.click(viewButton);

      await waitFor(() => {
        expect(screen.getByTestId('detail-modal')).toBeInTheDocument();
      });

      // Close modal
      const closeButton = screen.getByText('Close');
      fireEvent.click(closeButton);

      await waitFor(() => {
        expect(screen.queryByTestId('detail-modal')).not.toBeInTheDocument();
      });
    });

    it('should open modal with correct dataset data', async () => {
      render(<DatasetsPanel />);

      // Click second dataset
      const viewButtons = screen.getAllByText('View');
      fireEvent.click(viewButtons[1]);

      await waitFor(() => {
        const modal = screen.getByTestId('detail-modal');
        expect(modal).toBeInTheDocument();
        // Check the modal contains the second dataset name
        expect(modal).toHaveTextContent('Test Dataset 2');
      });
    });

    it('should call deleteDataset when delete button is clicked', async () => {
      mockDeleteDataset.mockResolvedValueOnce(undefined);

      render(<DatasetsPanel />);

      const deleteButtons = screen.getAllByText('Delete');
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(mockDeleteDataset).toHaveBeenCalledWith('dataset-1');
      });
    });

    it('should handle delete errors gracefully', async () => {
      const consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      mockDeleteDataset.mockRejectedValueOnce(new Error('Delete failed'));

      render(<DatasetsPanel />);

      const deleteButtons = screen.getAllByText('Delete');
      fireEvent.click(deleteButtons[0]);

      await waitFor(() => {
        expect(mockDeleteDataset).toHaveBeenCalled();
        expect(consoleErrorSpy).toHaveBeenCalledWith(
          'Failed to delete dataset:',
          expect.any(Error)
        );
      });

      // Should not crash the component
      expect(screen.getByTestId('dataset-card-dataset-1')).toBeInTheDocument();

      consoleErrorSpy.mockRestore();
    });
  });

  describe('Store Integration', () => {
    it('should use all required store functions', () => {
      render(<DatasetsPanel />);

      // Verify store hook was called
      expect(useDatasetsStore).toHaveBeenCalled();
    });

    it('should react to store updates', () => {
      const { rerender } = render(<DatasetsPanel />);

      // Initial render with no datasets
      expect(screen.queryByTestId('dataset-card-dataset-1')).not.toBeInTheDocument();

      // Update store to return datasets
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      rerender(<DatasetsPanel />);

      // Should now show datasets
      expect(screen.getByTestId('dataset-card-dataset-1')).toBeInTheDocument();
    });

    it('should handle loading state transitions', () => {
      // Start with loading
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: true,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      const { rerender } = render(<DatasetsPanel />);
      expect(screen.getByText(/loading datasets/i)).toBeInTheDocument();

      // Transition to loaded with data
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      rerender(<DatasetsPanel />);
      expect(screen.queryByText(/loading datasets/i)).not.toBeInTheDocument();
      expect(screen.getByTestId('dataset-card-dataset-1')).toBeInTheDocument();
    });
  });

  describe('Component Props', () => {
    beforeEach(() => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: mockDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });
    });

    it('should pass correct props to DownloadForm', () => {
      render(<DatasetsPanel />);

      const downloadForm = screen.getByTestId('download-form');
      expect(downloadForm).toBeInTheDocument();
    });

    it('should pass correct props to DatasetCard', () => {
      render(<DatasetsPanel />);

      const card = screen.getByTestId('dataset-card-dataset-1');
      expect(card).toBeInTheDocument();
      expect(screen.getByText('Test Dataset 1')).toBeInTheDocument();
    });

    it('should pass correct props to DatasetDetailModal', async () => {
      render(<DatasetsPanel />);

      // Open modal
      const viewButton = screen.getAllByText('View')[0];
      fireEvent.click(viewButton);

      await waitFor(() => {
        const modal = screen.getByTestId('detail-modal');
        expect(modal).toBeInTheDocument();
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle simultaneous loading and error states', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: true,
        error: 'Some error',
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      render(<DatasetsPanel />);

      // Should show error even when loading
      expect(screen.getByText('Some error')).toBeInTheDocument();
    });

    it('should not crash with very long dataset arrays', () => {
      const manyDatasets = Array.from({ length: 100 }, (_, i) => ({
        id: `dataset-${i}`,
        name: `Dataset ${i}`,
        status: DatasetStatus.READY,
        progress: 100,
        created_at: new Date().toISOString(),
        source: 'huggingface',
        hf_repo_id: `test/dataset-${i}`,
      }));

      (useDatasetsStore as any).mockReturnValue({
        datasets: manyDatasets,
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      expect(() => render(<DatasetsPanel />)).not.toThrow();
    });

    it('should handle empty datasets array without crashing', () => {
      (useDatasetsStore as any).mockReturnValue({
        datasets: [],
        loading: false,
        error: null,
        fetchDatasets: mockFetchDatasets,
        downloadDataset: mockDownloadDataset,
        deleteDataset: mockDeleteDataset,
      });

      expect(() => render(<DatasetsPanel />)).not.toThrow();
      expect(screen.getByText(/no datasets yet/i)).toBeInTheDocument();
    });
  });
});

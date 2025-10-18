/**
 * ExtractionListModal - Display a list of extractions for selection.
 *
 * Features:
 * - Compact list view of all extractions
 * - Click to view detailed modal
 * - Shows basic info: date, layers, samples, size
 */

import { useState, useEffect } from 'react';
import { Activity, Calendar, Layers, Database, AlertCircle, Trash2 } from 'lucide-react';
import { Model } from '../../types/model';
import { getModelExtractions, deleteExtractions } from '../../api/models';
import { DeleteExtractionsModal } from './DeleteExtractionsModal';

interface ExtractionStatistics {
  shape: number[];
  mean_magnitude: number;
  max_activation: number;
  min_activation: number;
  std_activation: number;
  sparsity_percent: number;
  size_mb: number;
}

export interface Extraction {
  extraction_id: string;
  model_id: string;
  status: string;
  architecture?: string;
  quantization?: string;
  dataset_path?: string;
  layer_indices: number[];
  hook_types: string[];
  max_samples: number;
  batch_size?: number;
  samples_processed?: number;
  num_samples_processed?: number;
  created_at: string;
  completed_at?: string;
  saved_files?: string[];
  statistics?: Record<string, ExtractionStatistics>;
  error_message?: string;
  progress?: number;
}

interface ExtractionListModalProps {
  model: Model;
  onClose: () => void;
  onSelectExtraction: (extraction: Extraction) => void;
}

export function ExtractionListModal({
  model,
  onClose,
  onSelectExtraction
}: ExtractionListModalProps) {
  const [extractions, setExtractions] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  useEffect(() => {
    fetchExtractions();
  }, [model.id]);

  const fetchExtractions = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getModelExtractions(model.id);
      setExtractions(data.extractions || []);
    } catch (err) {
      console.error('[ExtractionListModal] Failed to fetch:', err);
      setError(err instanceof Error ? err.message : 'Failed to load extractions');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const getTotalSize = (extraction: Extraction) => {
    if (!extraction.statistics || Object.keys(extraction.statistics).length === 0) {
      return 'N/A';
    }
    const totalMB = Object.values(extraction.statistics).reduce(
      (sum, stat) => sum + stat.size_mb, 0
    );
    return totalMB >= 1024
      ? `${(totalMB / 1024).toFixed(2)} GB`
      : `${totalMB.toFixed(2)} MB`;
  };

  const getSamplesProcessed = (extraction: Extraction) => {
    return extraction.samples_processed ?? extraction.num_samples_processed ?? 0;
  };

  const getStatusBadge = (status: string) => {
    const baseClasses = 'px-2 py-1 rounded text-xs font-medium';

    switch (status.toLowerCase()) {
      case 'completed':
        return <span className={`${baseClasses} bg-emerald-900/30 text-emerald-400`}>Completed</span>;
      case 'failed':
        return <span className={`${baseClasses} bg-red-900/30 text-red-400`}>Failed</span>;
      case 'cancelled':
        return <span className={`${baseClasses} bg-yellow-900/30 text-yellow-400`}>Cancelled</span>;
      default:
        return <span className={`${baseClasses} bg-blue-900/30 text-blue-400`}>{status}</span>;
    }
  };

  const toggleSelection = (extractionId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent opening detail modal
    const newSelected = new Set(selectedIds);
    if (newSelected.has(extractionId)) {
      newSelected.delete(extractionId);
    } else {
      newSelected.add(extractionId);
    }
    setSelectedIds(newSelected);
  };

  const toggleSelectAll = () => {
    if (selectedIds.size === extractions.length) {
      setSelectedIds(new Set());
    } else {
      const allIds = new Set(extractions.map(e => e.extraction_id));
      setSelectedIds(allIds);
    }
  };

  const handleDelete = async (extractionIds: string[]) => {
    try {
      await deleteExtractions(model.id, extractionIds);
      // Refresh the list after deletion
      await fetchExtractions();
      setSelectedIds(new Set());
    } catch (error) {
      console.error('[ExtractionListModal] Failed to delete:', error);
      throw error;
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-2xl font-semibold text-emerald-400">Extraction History</h2>
            <p className="text-sm text-slate-400 mt-1">
              {model.name} - {extractions.length} extraction{extractions.length !== 1 ? 's' : ''}
              {selectedIds.size > 0 && ` (${selectedIds.size} selected)`}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {extractions.length > 0 && selectedIds.size > 0 && (
              <button
                type="button"
                onClick={() => setShowDeleteModal(true)}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors text-white flex items-center gap-2"
              >
                <Trash2 className="w-4 h-4" />
                Delete Selected ({selectedIds.size})
              </button>
            )}
            <button
              type="button"
              onClick={onClose}
              className="text-slate-400 hover:text-slate-300 transition-colors"
              aria-label="Close"
            >
              <span className="text-2xl">×</span>
            </button>
          </div>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500"></div>
              <p className="text-slate-400 mt-4">Loading extractions...</p>
            </div>
          )}

          {error && (
            <div className="text-center py-12">
              <AlertCircle className="w-16 h-16 text-red-400 mx-auto mb-4" />
              <p className="text-red-400 text-lg mb-2">Failed to load extractions</p>
              <p className="text-slate-500">{error}</p>
              <button
                onClick={fetchExtractions}
                className="mt-4 px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300"
              >
                Retry
              </button>
            </div>
          )}

          {!loading && !error && extractions.length === 0 && (
            <div className="text-center py-12">
              <Database className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400 text-lg">No extractions yet</p>
              <p className="text-slate-500 mt-2">
                Extract activations from this model to see results here
              </p>
            </div>
          )}

          {!loading && !error && extractions.length > 0 && (
            <div className="space-y-3">
              {/* Select All */}
              <div className="flex items-center gap-3 bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <input
                  type="checkbox"
                  checked={selectedIds.size === extractions.length && extractions.length > 0}
                  onChange={toggleSelectAll}
                  className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900 cursor-pointer"
                />
                <span className="text-slate-300 font-medium">
                  Select All ({extractions.length})
                </span>
              </div>

              {extractions.map((extraction) => (
                <div
                  key={extraction.extraction_id}
                  className={`bg-slate-800/50 border rounded-lg p-4 transition-colors ${
                    selectedIds.has(extraction.extraction_id)
                      ? 'border-emerald-500/50 bg-emerald-900/10'
                      : 'border-slate-700 hover:border-slate-600'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    {/* Checkbox */}
                    <input
                      type="checkbox"
                      checked={selectedIds.has(extraction.extraction_id)}
                      onChange={(e) => toggleSelection(extraction.extraction_id, e)}
                      onClick={(e) => e.stopPropagation()}
                      className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900 cursor-pointer mt-0.5 flex-shrink-0"
                    />

                    <div className="flex-1 min-w-0 cursor-pointer" onClick={() => onSelectExtraction(extraction)}>
                      <div className="flex items-center gap-2 mb-2">
                        <Activity className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                        <h3 className="text-sm font-mono text-slate-100 truncate">
                          {extraction.extraction_id}
                        </h3>
                        {getStatusBadge(extraction.status)}
                      </div>
                      <div className="flex items-center gap-4 text-xs text-slate-400 flex-wrap">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          <span>{formatDate(extraction.created_at)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Layers className="w-3 h-3" />
                          <span>{extraction.layer_indices.length} layer{extraction.layer_indices.length !== 1 ? 's' : ''}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Database className="w-3 h-3" />
                          <span>{getSamplesProcessed(extraction).toLocaleString()} samples</span>
                        </div>
                      </div>
                      {extraction.error_message && (
                        <div className="mt-2 text-xs text-red-400 truncate">
                          Error: {extraction.error_message}
                        </div>
                      )}
                    </div>
                    <div className="text-right flex-shrink-0 ml-4">
                      <div className="text-sm font-mono text-emerald-400">
                        {getTotalSize(extraction)}
                      </div>
                      {extraction.saved_files && extraction.saved_files.length > 0 && (
                        <div className="text-xs text-slate-500 mt-1">
                          {extraction.saved_files.length} file{extraction.saved_files.length !== 1 ? 's' : ''}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 p-4 bg-slate-900/50">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <div>
              {selectedIds.size > 0 ? (
                <>
                  <span className="font-medium text-emerald-400">{selectedIds.size}</span> extraction{selectedIds.size !== 1 ? 's' : ''} selected
                </>
              ) : (
                'Click an extraction to view detailed statistics'
              )}
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300"
            >
              Close
            </button>
          </div>
        </div>
      </div>

      {/* Delete Modal */}
      {showDeleteModal && (
        <DeleteExtractionsModal
          model={model}
          onClose={() => setShowDeleteModal(false)}
          onDelete={handleDelete}
        />
      )}
    </div>
  );
}

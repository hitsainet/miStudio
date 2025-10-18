/**
 * DeleteExtractionsModal - Modal for selecting and deleting multiple extractions.
 *
 * Features:
 * - List all extractions with checkboxes
 * - All selected by default
 * - Shows size, date, and status
 * - Confirmation before deletion
 */

import { useState, useEffect } from 'react';
import { Trash2, Calendar, Database, AlertCircle, Loader2 } from 'lucide-react';
import { Model } from '../../types/model';
import { getModelExtractions } from '../../api/models';
import { Extraction } from './ExtractionListModal';

interface DeleteExtractionsModalProps {
  model: Model;
  onClose: () => void;
  onDelete: (extractionIds: string[]) => Promise<void>;
}

export function DeleteExtractionsModal({
  model,
  onClose,
  onDelete
}: DeleteExtractionsModalProps) {
  const [extractions, setExtractions] = useState<Extraction[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchExtractions();
  }, [model.id]);

  const fetchExtractions = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getModelExtractions(model.id);
      const fetchedExtractions = data.extractions || [];
      setExtractions(fetchedExtractions);

      // Select all by default
      const allIds = new Set(fetchedExtractions.map((e: Extraction) => e.extraction_id));
      setSelectedIds(allIds);
    } catch (err) {
      console.error('[DeleteExtractionsModal] Failed to fetch:', err);
      setError(err instanceof Error ? err.message : 'Failed to load extractions');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString();
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

  const toggleSelection = (extractionId: string) => {
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
      // Deselect all
      setSelectedIds(new Set());
    } else {
      // Select all
      const allIds = new Set(extractions.map(e => e.extraction_id));
      setSelectedIds(allIds);
    }
  };

  const handleDelete = async () => {
    if (selectedIds.size === 0) {
      return;
    }

    const confirmMessage = selectedIds.size === 1
      ? 'Are you sure you want to delete this extraction? This action cannot be undone.'
      : `Are you sure you want to delete ${selectedIds.size} extractions? This action cannot be undone.`;

    if (!confirm(confirmMessage)) {
      return;
    }

    try {
      setDeleting(true);
      setError(null);
      await onDelete(Array.from(selectedIds));
      onClose();
    } catch (err) {
      console.error('[DeleteExtractionsModal] Failed to delete:', err);
      setError(err instanceof Error ? err.message : 'Failed to delete extractions');
    } finally {
      setDeleting(false);
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'completed':
        return 'text-emerald-400';
      case 'failed':
        return 'text-red-400';
      case 'cancelled':
        return 'text-yellow-400';
      default:
        return 'text-blue-400';
    }
  };

  const calculateTotalSize = () => {
    let totalMB = 0;
    extractions.forEach(extraction => {
      if (selectedIds.has(extraction.extraction_id) && extraction.statistics) {
        totalMB += Object.values(extraction.statistics).reduce(
          (sum, stat) => sum + stat.size_mb, 0
        );
      }
    });
    return totalMB >= 1024
      ? `${(totalMB / 1024).toFixed(2)} GB`
      : `${totalMB.toFixed(2)} MB`;
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <Trash2 className="w-6 h-6 text-red-400" />
              <h2 className="text-2xl font-semibold text-slate-100">Delete Extractions</h2>
            </div>
            <p className="text-sm text-slate-400">
              {model.name} - Select extractions to delete
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-300 transition-colors"
            aria-label="Close"
            disabled={deleting}
          >
            <span className="text-2xl">×</span>
          </button>
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
              <p className="text-red-400 text-lg mb-2">Error</p>
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
              <p className="text-slate-400 text-lg">No extractions to delete</p>
            </div>
          )}

          {!loading && !error && extractions.length > 0 && (
            <div className="space-y-4">
              {/* Select All */}
              <div className="flex items-center justify-between bg-slate-800/50 border border-slate-700 rounded-lg p-4">
                <label className="flex items-center gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={selectedIds.size === extractions.length}
                    onChange={toggleSelectAll}
                    className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900 cursor-pointer"
                  />
                  <span className="text-slate-100 font-medium">
                    Select All ({extractions.length} extractions)
                  </span>
                </label>
                <div className="text-slate-400 text-sm">
                  {selectedIds.size} selected
                </div>
              </div>

              {/* Extraction List */}
              <div className="space-y-3">
                {extractions.map((extraction) => (
                  <label
                    key={extraction.extraction_id}
                    className={`flex items-start gap-4 bg-slate-800/50 border rounded-lg p-4 cursor-pointer transition-colors ${
                      selectedIds.has(extraction.extraction_id)
                        ? 'border-red-500/50 bg-red-900/10'
                        : 'border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    <input
                      type="checkbox"
                      checked={selectedIds.has(extraction.extraction_id)}
                      onChange={() => toggleSelection(extraction.extraction_id)}
                      className="w-5 h-5 rounded border-slate-600 bg-slate-800 text-emerald-600 focus:ring-emerald-500 focus:ring-offset-slate-900 cursor-pointer mt-0.5 flex-shrink-0"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-2">
                        <h3 className="text-sm font-mono text-slate-100 truncate">
                          {extraction.extraction_id}
                        </h3>
                        <span className={`text-xs px-2 py-1 rounded ${getStatusColor(extraction.status)}`}>
                          {extraction.status}
                        </span>
                      </div>
                      <div className="flex items-center gap-4 text-xs text-slate-400 flex-wrap">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-3 h-3" />
                          <span>{formatDate(extraction.created_at)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Database className="w-3 h-3" />
                          <span>{getSamplesProcessed(extraction).toLocaleString()} samples</span>
                        </div>
                        <div>
                          <span>Size: {getTotalSize(extraction)}</span>
                        </div>
                      </div>
                    </div>
                  </label>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 p-4 bg-slate-900/50">
          <div className="flex items-center justify-between">
            <div className="text-sm text-slate-400">
              {selectedIds.size > 0 ? (
                <>
                  Will delete {selectedIds.size} extraction{selectedIds.size !== 1 ? 's' : ''} ({calculateTotalSize()})
                </>
              ) : (
                'No extractions selected'
              )}
            </div>
            <div className="flex gap-2">
              <button
                onClick={onClose}
                disabled={deleting}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={selectedIds.size === 0 || deleting}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded transition-colors text-white disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
              >
                {deleting ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    Delete Selected
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

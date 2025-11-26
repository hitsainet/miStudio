/**
 * SAEsPanel - Main SAE management panel.
 *
 * This component provides the primary interface for managing Sparse Autoencoders,
 * including downloading from HuggingFace, importing from training, and managing
 * local SAE files.
 *
 * Features:
 * - Download SAEs from HuggingFace repositories (SAELens format)
 * - Import SAEs from completed training jobs
 * - Real-time download progress tracking
 * - Filter by source, status, and model
 * - Navigate to Steering panel with selected SAE
 */

import { useEffect, useState } from 'react';
import { useSAEsStore } from '../../stores/saesStore';
import { useSteeringStore } from '../../stores/steeringStore';
import { SAE, SAESource, SAEStatus } from '../../types/sae';
import { SAECard } from '../saes/SAECard';
import { DownloadFromHF } from '../saes/DownloadFromHF';
import { Search, Filter, Brain, Cloud, Layers, HardDrive } from 'lucide-react';
import { COMPONENTS } from '../../config/brand';

interface SAEsPanelProps {
  onNavigateToSteering?: () => void;
}

export function SAEsPanel({ onNavigateToSteering }: SAEsPanelProps = {}) {
  const { selectSAE } = useSteeringStore();
  const {
    saes,
    loading,
    error,
    filters,
    pagination,
    fetchSAEs,
    deleteSAE,
    setFilters,
    setPage,
    clearError,
  } = useSAEsStore();

  const [searchInput, setSearchInput] = useState(filters.search);
  const [showFilters, setShowFilters] = useState(false);

  // Fetch SAEs on mount
  useEffect(() => {
    fetchSAEs();
  }, [fetchSAEs]);

  // Refetch when filters or pagination change
  useEffect(() => {
    fetchSAEs();
  }, [filters, pagination.skip, fetchSAEs]);

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchInput !== filters.search) {
        setFilters({ search: searchInput });
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput, filters.search, setFilters]);

  const handleUseSteering = (sae: SAE) => {
    // Select the SAE in the steering store, then navigate if callback provided
    selectSAE(sae);
    if (onNavigateToSteering) {
      onNavigateToSteering();
    }
  };

  const handleDelete = async (id: string) => {
    try {
      await deleteSAE(id);
    } catch (error) {
      console.error('[SAEsPanel] Failed to delete SAE:', error);
    }
  };

  const handleSourceFilter = (source: SAESource | null) => {
    setFilters({ source });
  };

  const handleStatusFilter = (status: SAEStatus | null) => {
    setFilters({ status });
  };

  const sortedSAEs = [...saes].sort((a, b) => {
    // Active downloads first
    const aActive = [SAEStatus.DOWNLOADING, SAEStatus.CONVERTING, SAEStatus.PENDING].includes(a.status);
    const bActive = [SAEStatus.DOWNLOADING, SAEStatus.CONVERTING, SAEStatus.PENDING].includes(b.status);

    if (aActive && !bActive) return -1;
    if (!aActive && bActive) return 1;

    // Then by creation time (newest first)
    return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
  });

  return (
    <div className="">
      <div className="max-w-[90%] mx-auto px-4 py-6">
        {/* Header */}
        <div className="mb-6">
          <h1 className="text-xl font-semibold text-slate-900 dark:text-slate-100 mb-2">
            Sparse Autoencoders (SAEs)
          </h1>
          <p className="text-slate-600 dark:text-slate-400">
            Download and manage SAEs for feature steering. Supports SAELens format from HuggingFace.
          </p>
        </div>

        {/* Error message */}
        {error && (
          <div className="mb-6 p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 flex items-center justify-between">
            <span>{error}</span>
            <button onClick={clearError} className="text-red-400 hover:text-red-300">
              Dismiss
            </button>
          </div>
        )}

        {/* Download Form */}
        <div className="mb-6">
          <DownloadFromHF onDownloadComplete={() => fetchSAEs()} />
        </div>

        {/* Search and Filters */}
        <div className="mb-6 space-y-3">
          <div className="flex items-center gap-3">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
              <input
                type="text"
                placeholder="Search SAEs by name, model, or repository..."
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 transition-colors"
              />
            </div>

            {/* Filter Toggle */}
            <button
              onClick={() => setShowFilters(!showFilters)}
              className={`px-4 py-2 rounded-lg flex items-center gap-2 transition-colors ${
                showFilters || filters.source || filters.status
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                  : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
              }`}
            >
              <Filter className="w-4 h-4" />
              Filters
              {(filters.source || filters.status) && (
                <span className="w-2 h-2 bg-emerald-400 rounded-full" />
              )}
            </button>
          </div>

          {/* Filter Options */}
          {showFilters && (
            <div className={`${COMPONENTS.card.base} p-4 space-y-4`}>
              {/* Source Filter */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Source</label>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => handleSourceFilter(null)}
                    className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-colors ${
                      !filters.source
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    All Sources
                  </button>
                  <button
                    onClick={() => handleSourceFilter(SAESource.HUGGINGFACE)}
                    className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-colors ${
                      filters.source === SAESource.HUGGINGFACE
                        ? 'bg-yellow-500/20 text-yellow-400 border border-yellow-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    <Cloud className="w-3.5 h-3.5" />
                    HuggingFace
                  </button>
                  <button
                    onClick={() => handleSourceFilter(SAESource.TRAINED)}
                    className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-colors ${
                      filters.source === SAESource.TRAINED
                        ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    <Layers className="w-3.5 h-3.5" />
                    Trained
                  </button>
                  <button
                    onClick={() => handleSourceFilter(SAESource.LOCAL)}
                    className={`px-3 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition-colors ${
                      filters.source === SAESource.LOCAL
                        ? 'bg-slate-500/20 text-slate-300 border border-slate-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    <HardDrive className="w-3.5 h-3.5" />
                    Local File
                  </button>
                </div>
              </div>

              {/* Status Filter */}
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">Status</label>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => handleStatusFilter(null)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      !filters.status
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    All Statuses
                  </button>
                  <button
                    onClick={() => handleStatusFilter(SAEStatus.READY)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      filters.status === SAEStatus.READY
                        ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    Ready
                  </button>
                  <button
                    onClick={() => handleStatusFilter(SAEStatus.DOWNLOADING)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      filters.status === SAEStatus.DOWNLOADING
                        ? 'bg-blue-500/20 text-blue-400 border border-blue-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    Downloading
                  </button>
                  <button
                    onClick={() => handleStatusFilter(SAEStatus.ERROR)}
                    className={`px-3 py-1.5 rounded-lg text-sm transition-colors ${
                      filters.status === SAEStatus.ERROR
                        ? 'bg-red-500/20 text-red-400 border border-red-500/50'
                        : 'bg-slate-800 text-slate-300 border border-slate-700 hover:border-slate-600'
                    }`}
                  >
                    Error
                  </button>
                </div>
              </div>

              {/* Clear Filters */}
              {(filters.source || filters.status) && (
                <button
                  onClick={() => {
                    handleSourceFilter(null);
                    handleStatusFilter(null);
                  }}
                  className="text-sm text-slate-400 hover:text-slate-300"
                >
                  Clear all filters
                </button>
              )}
            </div>
          )}
        </div>

        {/* Loading state */}
        {loading && saes.length === 0 && (
          <div className="text-center py-12">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500"></div>
            <p className="text-slate-400 mt-4">Loading SAEs...</p>
          </div>
        )}

        {/* Empty state */}
        {!loading && saes.length === 0 && (
          <div className="text-center py-12">
            <Brain className="w-12 h-12 text-slate-600 mx-auto mb-4" />
            <p className="text-slate-400 text-lg">No SAEs yet</p>
            <p className="text-slate-500 mt-2">
              Download an SAE from HuggingFace or import from a completed training job
            </p>
          </div>
        )}

        {/* SAEs grid */}
        {saes.length > 0 && (
          <div>
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-base font-semibold text-slate-100">
                Your SAEs ({pagination.total})
              </h2>
              {pagination.total > pagination.limit && (
                <div className="flex items-center gap-2 text-sm text-slate-400">
                  <span>
                    Showing {pagination.skip + 1}-{Math.min(pagination.skip + pagination.limit, pagination.total)}
                  </span>
                </div>
              )}
            </div>

            <div className="grid gap-3">
              {sortedSAEs.map((sae) => (
                <SAECard
                  key={sae.id}
                  sae={sae}
                  onUseSteering={() => handleUseSteering(sae)}
                  onDelete={handleDelete}
                />
              ))}
            </div>

            {/* Pagination */}
            {pagination.total > pagination.limit && (
              <div className="flex items-center justify-center gap-2 mt-6">
                <button
                  onClick={() => setPage(Math.max(0, pagination.skip - pagination.limit))}
                  disabled={pagination.skip === 0}
                  className={`px-4 py-2 rounded-lg text-sm ${COMPONENTS.button.secondary} disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  Previous
                </button>
                <span className="text-sm text-slate-400">
                  Page {Math.floor(pagination.skip / pagination.limit) + 1} of{' '}
                  {Math.ceil(pagination.total / pagination.limit)}
                </span>
                <button
                  onClick={() => setPage(pagination.skip + pagination.limit)}
                  disabled={!pagination.hasMore}
                  className={`px-4 py-2 rounded-lg text-sm ${COMPONENTS.button.secondary} disabled:opacity-50 disabled:cursor-not-allowed`}
                >
                  Next
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

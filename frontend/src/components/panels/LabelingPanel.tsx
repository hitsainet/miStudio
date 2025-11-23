/**
 * LabelingPanel Component
 *
 * Panel for viewing and managing all semantic labeling jobs.
 * Shows labeling jobs in a grid with status, progress, and actions.
 */

import React, { useEffect, useState, useMemo } from 'react';
import { Tag, Filter } from 'lucide-react';
import { useLabelingStore } from '../../stores/labelingStore';
import { useLabelingWebSocket } from '../../hooks/useLabelingWebSocket';
import { LabelingJobCard } from '../labeling/LabelingJobCard';
import { LabelingStatus } from '../../types/labeling';
import { COMPONENTS } from '../../config/brand';

export const LabelingPanel: React.FC = () => {
  const {
    labelingJobs,
    isLoading,
    error,
    fetchLabelingJobs,
    cancelLabeling,
    deleteLabeling,
  } = useLabelingStore();

  const [statusFilter, setStatusFilter] = useState<string[]>([]);

  // Get IDs of active labeling jobs for WebSocket subscription
  const activeJobIds = useMemo(
    () =>
      (labelingJobs || [])
        .filter(
          (job) =>
            job.status === LabelingStatus.QUEUED || job.status === LabelingStatus.LABELING
        )
        .map((job) => job.id),
    [labelingJobs]
  );

  // Subscribe to WebSocket updates for active jobs
  useLabelingWebSocket(activeJobIds);

  // Load labeling jobs on mount
  useEffect(() => {
    fetchLabelingJobs();
  }, []);

  // Filter jobs based on status filter (client-side filtering)
  const filteredJobs = useMemo(() => {
    if (statusFilter.length === 0) {
      return labelingJobs || [];
    }
    return (labelingJobs || []).filter((job) =>
      statusFilter.includes(job.status.toLowerCase())
    );
  }, [labelingJobs, statusFilter]);

  // Toggle status filter
  const toggleStatusFilter = (status: string) => {
    setStatusFilter((prev) => {
      if (prev.includes(status)) {
        return prev.filter((s) => s !== status);
      } else {
        return [...prev, status];
      }
    });
  };

  return (
    <div className="max-w-[80%] mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className={`text-2xl font-semibold ${COMPONENTS.text.primary} mb-2`}>Semantic Labeling</h1>
        <p className={COMPONENTS.text.secondary}>
          View and manage all feature labeling jobs
        </p>
      </div>

      {/* Filters */}
      <div className={`mb-6 ${COMPONENTS.card.base} p-4`}>
        <div className="flex items-center gap-3">
          <Filter className={`w-5 h-5 ${COMPONENTS.text.secondary}`} />
          <span className={`text-sm font-medium ${COMPONENTS.text.primary}`}>Filter by status:</span>
          <div className="flex gap-2">
            {['queued', 'labeling', 'completed', 'failed', 'cancelled'].map((status) => (
              <button
                key={status}
                onClick={() => toggleStatusFilter(status)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  statusFilter.includes(status)
                    ? 'bg-emerald-600 text-white'
                    : `${COMPONENTS.button.secondary}`
                }`}
              >
                {status.charAt(0).toUpperCase() + status.slice(1)}
              </button>
            ))}
          </div>
          {statusFilter.length > 0 && (
            <button
              onClick={() => setStatusFilter([])}
              className={`ml-auto text-sm ${COMPONENTS.text.secondary} hover:text-slate-300`}
            >
              Clear filters
            </button>
          )}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="text-center py-12">
          <div className={COMPONENTS.spinner}></div>
          <p className={`mt-4 ${COMPONENTS.text.secondary}`}>Loading labeling jobs...</p>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-6 text-center">
          <p className="text-red-400">{error}</p>
          <button
            onClick={() => fetchLabelingJobs()}
            className={`mt-4 ${COMPONENTS.button.secondary}`}
          >
            Retry
          </button>
        </div>
      )}

      {/* Labeling Jobs Grid */}
      {!isLoading && !error && (
        <>
          {filteredJobs.length === 0 ? (
            <div className="text-center py-12">
              <Tag className={`w-16 h-16 ${COMPONENTS.text.muted} mx-auto mb-4`} />
              <p className={`${COMPONENTS.text.secondary} text-lg mb-2`}>No labeling jobs found</p>
              <p className={`${COMPONENTS.text.muted} text-sm`}>
                {statusFilter.length > 0
                  ? 'Try removing some filters'
                  : 'Start a labeling job from a completed extraction'}
              </p>
            </div>
          ) : (
            <>
              {/* Stats */}
              <div className={`mb-4 text-sm ${COMPONENTS.text.secondary}`}>
                Showing {filteredJobs.length} labeling job{filteredJobs.length !== 1 ? 's' : ''}
              </div>

              {/* Grid */}
              <div className="grid grid-cols-1 gap-4">
                {filteredJobs.map((job) => (
                  <LabelingJobCard
                    key={job.id}
                    job={job}
                    onCancel={async () => {
                      await cancelLabeling(job.id);
                      // Refresh list after cancellation
                      fetchLabelingJobs();
                    }}
                    onDelete={async () => {
                      await deleteLabeling(job.id);
                      // Refresh list after deletion
                      fetchLabelingJobs();
                    }}
                  />
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  );
};

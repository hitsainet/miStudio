/**
 * TokenizationProgressDisplay - Display detailed tokenization progress.
 *
 * Shows comprehensive progress information including:
 * - Current stage (loading, tokenizing, filtering, saving)
 * - Progress bar with percentage
 * - Samples processed / total
 * - Elapsed time and estimated time remaining
 * - Processing rate (samples/second)
 * - Filter statistics (when filtering is enabled)
 */

import { useEffect, useState } from 'react';
import { Clock, Zap, Filter } from 'lucide-react';
import { DatasetTokenizationProgress } from '../../types/dataset';

interface TokenizationProgressDisplayProps {
  progress: DatasetTokenizationProgress;
}

export function TokenizationProgressDisplay({ progress }: TokenizationProgressDisplayProps) {
  // Calculate elapsed time locally to ensure it updates even when backend doesn't send updates
  // (This is needed because multiprocessing tokenization doesn't emit progress during processing)
  const [localElapsed, setLocalElapsed] = useState<number>(0);

  useEffect(() => {
    if (!progress.started_at) return;

    const startTime = new Date(progress.started_at).getTime();

    const updateElapsed = () => {
      const now = Date.now();
      const elapsed = (now - startTime) / 1000; // Convert to seconds
      setLocalElapsed(elapsed);
    };

    // Update immediately
    updateElapsed();

    // Update every second
    const interval = setInterval(updateElapsed, 1000);

    return () => clearInterval(interval);
  }, [progress.started_at]);

  // Use local elapsed time when started_at is available, otherwise fall back to backend value
  const displayElapsed = progress.started_at ? localElapsed : progress.elapsed_seconds;

  // Format time in seconds to human-readable string
  const formatTime = (seconds: number | undefined): string => {
    if (seconds === undefined || seconds === null) return 'N/A';

    if (seconds < 60) {
      return `${Math.round(seconds)}s`;
    } else if (seconds < 3600) {
      const mins = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      return `${mins}m ${secs}s`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const mins = Math.floor((seconds % 3600) / 60);
      return `${hours}h ${mins}m`;
    }
  };

  // Get stage display name
  const getStageDisplay = (stage: string): string => {
    const stageMap: Record<string, string> = {
      loading: 'Loading Dataset',
      tokenizing: 'Tokenizing',
      filtering: 'Filtering Samples',
      saving: 'Saving Results',
      complete: 'Complete',
    };
    return stageMap[stage] || stage;
  };

  // Get stage color
  const getStageColor = (stage: string): string => {
    const colorMap: Record<string, string> = {
      loading: 'text-blue-400',
      tokenizing: 'text-emerald-400',
      filtering: 'text-yellow-400',
      saving: 'text-purple-400',
      complete: 'text-green-400',
    };
    return colorMap[stage] || 'text-slate-400';
  };

  return (
    <div className="space-y-2">
      {/* Stage and Progress Bar */}
      <div>
        <div className="flex items-center justify-between text-xs text-slate-400 mb-1">
          <span className={getStageColor(progress.stage)}>
            {getStageDisplay(progress.stage)}
          </span>
          <span className="font-medium text-emerald-400">
            {progress.progress.toFixed(1)}%
          </span>
        </div>
        <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-emerald-500 transition-all duration-300"
            style={{ width: `${progress.progress}%` }}
          />
        </div>
      </div>

      {/* Samples Progress */}
      <div className="flex items-center justify-between text-xs">
        <span className="text-slate-500">Samples:</span>
        <span className="text-slate-300 font-medium">
          {progress.samples_processed.toLocaleString()} / {progress.total_samples.toLocaleString()}
        </span>
      </div>

      {/* Time Information */}
      {(displayElapsed !== undefined || progress.estimated_seconds_remaining !== undefined) && (
        <div className="grid grid-cols-2 gap-2 text-xs">
          {displayElapsed !== undefined && (
            <div className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-slate-500" />
              <span className="text-slate-500">Elapsed:</span>
              <span className="text-slate-300 font-medium">
                {formatTime(displayElapsed)}
              </span>
            </div>
          )}
          {progress.estimated_seconds_remaining !== undefined && (
            <div className="flex items-center gap-1.5">
              <Clock className="w-3 h-3 text-slate-500" />
              <span className="text-slate-500">Remaining:</span>
              <span className="text-slate-300 font-medium">
                {formatTime(progress.estimated_seconds_remaining)}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Processing Rate */}
      {progress.samples_per_second !== undefined && (
        <div className="flex items-center gap-1.5 text-xs">
          <Zap className="w-3 h-3 text-yellow-400" />
          <span className="text-slate-500">Rate:</span>
          <span className="text-slate-300 font-medium">
            {progress.samples_per_second.toFixed(1)} samples/sec
          </span>
        </div>
      )}

      {/* Filter Statistics */}
      {progress.filter_stats && (
        <div className="pt-2 border-t border-slate-700/50">
          <div className="flex items-center gap-1.5 mb-1.5">
            <Filter className="w-3 h-3 text-blue-400" />
            <span className="text-xs font-medium text-slate-300">Filter Statistics</span>
          </div>
          <div className="grid grid-cols-2 gap-1.5 text-xs">
            <div className="flex justify-between">
              <span className="text-slate-500">Filtered:</span>
              <span className="text-slate-300">
                {progress.filter_stats.samples_filtered.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Rate:</span>
              <span className="text-slate-300">
                {progress.filter_stats.filter_rate.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between col-span-2">
              <span className="text-slate-500">Junk Tokens:</span>
              <span className="text-slate-300">
                {progress.filter_stats.junk_tokens.toLocaleString()} / {progress.filter_stats.total_tokens.toLocaleString()}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

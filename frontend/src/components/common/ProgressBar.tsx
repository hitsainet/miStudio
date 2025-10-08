/**
 * ProgressBar component for displaying download/processing progress.
 *
 * This component renders a progress bar with percentage display.
 */

import React from 'react';

interface ProgressBarProps {
  progress: number; // 0-100
  className?: string;
  showPercentage?: boolean;
}

export function ProgressBar({
  progress,
  className = '',
  showPercentage = true,
}: ProgressBarProps) {
  const clampedProgress = Math.max(0, Math.min(100, progress));

  return (
    <div className={`space-y-1 ${className}`}>
      <div className="w-full h-2 bg-slate-800 rounded-full overflow-hidden">
        <div
          className="h-full bg-emerald-500 rounded-full transition-all duration-300 ease-out"
          style={{ width: `${clampedProgress}%` }}
        />
      </div>
      {showPercentage && (
        <div className="text-right text-xs text-slate-400">
          {clampedProgress.toFixed(1)}%
        </div>
      )}
    </div>
  );
}

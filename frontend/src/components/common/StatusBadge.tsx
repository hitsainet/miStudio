/**
 * StatusBadge component for displaying dataset/model status.
 *
 * This component renders a colored badge based on the status value.
 */

import { DatasetStatus } from '../../types/dataset';

interface StatusBadgeProps {
  status: DatasetStatus | string;
  className?: string;
}

const STATUS_COLORS: Record<string, string> = {
  downloading: 'bg-blue-500/20 text-blue-400 border-blue-500/30',
  processing: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  ready: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  error: 'bg-red-500/20 text-red-400 border-red-500/30',
};

export function StatusBadge({ status, className = '' }: StatusBadgeProps) {
  const normalizedStatus = String(status).toLowerCase();
  const colorClass = STATUS_COLORS[normalizedStatus] || STATUS_COLORS.ready;

  return (
    <span
      className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-medium border ${colorClass} ${className}`}
    >
      {normalizedStatus.charAt(0).toUpperCase() + normalizedStatus.slice(1)}
    </span>
  );
}

/**
 * MetricWarning Component
 *
 * Displays warning badge for critical metric thresholds
 */

import { AlertTriangle } from 'lucide-react';

interface MetricWarningProps {
  type: 'temperature' | 'memory' | 'utilization';
  value: number;
  label?: string;
}

export function MetricWarning({ type, value, label }: MetricWarningProps) {
  const shouldWarn = () => {
    switch (type) {
      case 'temperature':
        return value > 85;
      case 'memory':
        return value > 95;
      case 'utilization':
        return value > 95;
      default:
        return false;
    }
  };

  const getMessage = () => {
    switch (type) {
      case 'temperature':
        if (value > 90) return 'Critical Temperature!';
        if (value > 85) return 'High Temperature';
        break;
      case 'memory':
        if (value > 98) return 'Memory Critical!';
        if (value > 95) return 'Memory High';
        break;
      case 'utilization':
        if (value > 98) return 'Max Utilization';
        if (value > 95) return 'High Utilization';
        break;
    }
    return label || 'Warning';
  };

  const getStyles = () => {
    if (type === 'temperature' && value > 90) {
      return 'bg-red-900/80 border-red-600 text-red-100';
    }
    if ((type === 'memory' || type === 'utilization') && value > 98) {
      return 'bg-red-900/80 border-red-600 text-red-100';
    }
    return 'bg-orange-900/80 border-orange-600 text-orange-100';
  };

  if (!shouldWarn()) {
    return null;
  }

  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded-md border ${getStyles()} animate-pulse`}>
      <AlertTriangle className="w-4 h-4" />
      <span className="text-sm font-medium">{getMessage()}</span>
    </div>
  );
}

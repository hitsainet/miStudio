/**
 * MetricValue Component
 *
 * Displays a metric value with graceful handling of missing/null/undefined values
 */

interface MetricValueProps {
  value: number | null | undefined;
  format?: 'number' | 'percent' | 'temperature' | 'memory' | 'power';
  decimals?: number;
  suffix?: string;
  fallback?: string;
  className?: string;
}

export function MetricValue({
  value,
  format = 'number',
  decimals = 1,
  suffix,
  fallback = 'N/A',
  className = ''
}: MetricValueProps) {
  // Handle missing values
  if (value === null || value === undefined || isNaN(value)) {
    return (
      <span className={`text-slate-500 ${className}`}>
        {fallback}
      </span>
    );
  }

  // Format the value based on type
  let formattedValue: string;
  let unit: string = suffix || '';

  switch (format) {
    case 'percent':
      formattedValue = value.toFixed(decimals);
      unit = suffix || '%';
      break;
    case 'temperature':
      formattedValue = value.toFixed(0);
      unit = suffix || '°C';
      break;
    case 'memory':
      formattedValue = value.toFixed(decimals);
      unit = suffix || 'GB';
      break;
    case 'power':
      formattedValue = value.toFixed(0);
      unit = suffix || 'W';
      break;
    case 'number':
    default:
      formattedValue = value.toFixed(decimals);
      break;
  }

  return (
    <span className={className}>
      {formattedValue}
      {unit && <span className="ml-0.5">{unit}</span>}
    </span>
  );
}

/**
 * Metric Helper Utilities
 *
 * Safe accessors and validators for metric values
 */

/**
 * Safely get a nested property value
 */
export function safeGet<T>(
  obj: any,
  path: string,
  defaultValue: T
): T {
  const keys = path.split('.');
  let result = obj;

  for (const key of keys) {
    if (result === null || result === undefined) {
      return defaultValue;
    }
    result = result[key];
  }

  return result !== null && result !== undefined ? result : defaultValue;
}

/**
 * Check if a metric value is valid
 */
export function isValidMetric(value: any): boolean {
  return value !== null && value !== undefined && !isNaN(Number(value));
}

/**
 * Clamp a value between min and max
 */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

/**
 * Get temperature color class based on value
 */
export function getTemperatureColor(temp: number | null | undefined): string {
  if (!isValidMetric(temp)) return 'text-slate-400 bg-slate-500';
  const temperature = temp as number;

  if (temperature > 85) return 'text-red-400 bg-red-500';
  if (temperature > 80) return 'text-orange-400 bg-orange-500';
  if (temperature > 70) return 'text-yellow-400 bg-yellow-500';
  return 'text-emerald-400 bg-emerald-500';
}

/**
 * Get utilization color class based on value
 */
export function getUtilizationColor(util: number | null | undefined): string {
  if (!isValidMetric(util)) return 'bg-slate-500';
  const utilization = util as number;

  if (utilization > 95) return 'bg-red-500';
  if (utilization > 85) return 'bg-orange-500';
  if (utilization > 70) return 'bg-yellow-500';
  return 'bg-emerald-500';
}

/**
 * Format bytes to human-readable string
 */
export function formatBytes(bytes: number | null | undefined): string {
  if (!isValidMetric(bytes)) return 'N/A';

  const value = bytes as number;
  if (value === 0) return '0 B';

  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(value) / Math.log(k));

  return `${(value / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
}

/**
 * Check if metrics are stale (older than threshold)
 */
export function areMetricsStale(
  lastUpdate: number | null | undefined,
  thresholdMs: number = 10000
): boolean {
  if (!lastUpdate) return true;
  return Date.now() - lastUpdate > thresholdMs;
}

/**
 * Format file size in bytes to human-readable format using 1024 divisor
 * @param bytes File size in bytes
 * @returns Formatted string (e.g., "2.3 GB", "450 MB")
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Format ISO date string to readable date format
 * @param isoString ISO 8601 date string
 * @returns Formatted date (e.g., "Oct 5, 2025")
 */
export function formatDate(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

/**
 * Format ISO date string to readable date and time format
 * @param isoString ISO 8601 date string
 * @returns Formatted date and time (e.g., "Oct 5, 2025, 2:34 PM")
 */
export function formatDateTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: 'numeric',
    minute: '2-digit',
  });
}

/**
 * Format number with thousands separators
 * @param num Number to format
 * @returns Formatted string (e.g., "10,000")
 */
export function formatNumber(num: number): string {
  return num.toLocaleString('en-US');
}

/**
 * Format progress percentage
 * @param progress Progress value (0-100)
 * @returns Formatted string (e.g., "45.2%")
 */
export function formatProgress(progress: number): string {
  return `${progress.toFixed(1)}%`;
}

/**
 * Format duration in seconds to human-readable format
 * @param seconds Duration in seconds
 * @returns Formatted string (e.g., "2h 30m", "45m 20s", "30s")
 */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.floor(seconds)}s`;
  }

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }

  if (minutes > 0) {
    return `${minutes}m ${secs}s`;
  }

  return `${secs}s`;
}

/**
 * Format activation value with intelligent precision
 * Uses scientific notation for very small/large numbers, fixed decimals for normal range
 * @param value Activation value
 * @param decimals Number of decimal places for normal numbers (default: 3)
 * @returns Formatted string (e.g., "12.345", "5.20e-41", "0.000")
 */
export function formatActivation(value: number, decimals: number = 3): string {
  // Handle edge cases
  if (value === 0) {
    return '0.000';
  }
  if (!isFinite(value)) {
    return 'N/A';
  }

  const absValue = Math.abs(value);

  // Use scientific notation for very small numbers (< 0.001) or very large numbers (> 10000)
  if (absValue < 0.001 || absValue > 10000) {
    return value.toExponential(2);
  }

  // Use fixed decimal notation for normal range
  return value.toFixed(decimals);
}

/**
 * Format L0 sparsity in dual format: absolute count and percentage
 *
 * Aligns with Neuronpedia convention where L0 is shown as absolute count
 * (e.g., "~71 features/token") while also showing percentage for context.
 *
 * @param l0Fraction L0 sparsity as fraction (0-1), e.g., 0.0043
 * @param totalFeatures Total number of features in the SAE (latent_dim)
 * @param showBoth Whether to show both formats or just absolute
 * @returns Formatted string (e.g., "~11 features (4.3%)" or "~11 features")
 *
 * @example
 * formatL0Sparsity(0.0043, 16384) // "~71 features (0.4%)"
 * formatL0Sparsity(0.043, 256) // "~11 features (4.3%)"
 * formatL0Sparsity(0.043, 256, false) // "~11"
 */
export function formatL0Sparsity(
  l0Fraction: number,
  totalFeatures: number,
  showBoth: boolean = true
): string {
  if (l0Fraction === 0 || totalFeatures === 0) {
    return showBoth ? '0 features (0%)' : '0';
  }

  // Calculate absolute L0 (average features active per token)
  const absoluteL0 = Math.round(l0Fraction * totalFeatures);
  const percentage = (l0Fraction * 100).toFixed(1);

  if (showBoth) {
    return `~${absoluteL0} features (${percentage}%)`;
  }
  return `~${absoluteL0}`;
}

/**
 * Format L0 sparsity as just the absolute count (Neuronpedia style)
 * @param l0Fraction L0 sparsity as fraction (0-1)
 * @param totalFeatures Total number of features in the SAE
 * @returns Formatted string (e.g., "~71")
 */
export function formatL0Absolute(l0Fraction: number, totalFeatures: number): string {
  if (l0Fraction === 0 || totalFeatures === 0) {
    return '0';
  }
  const absoluteL0 = Math.round(l0Fraction * totalFeatures);
  return `~${absoluteL0}`;
}

/**
 * Format L0 sparsity as percentage (original miStudio style)
 * @param l0Fraction L0 sparsity as fraction (0-1)
 * @returns Formatted string (e.g., "4.3%")
 */
export function formatL0Percent(l0Fraction: number): string {
  return `${(l0Fraction * 100).toFixed(1)}%`;
}

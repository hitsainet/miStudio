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

/**
 * Feature Quality Color Grades
 *
 * Color coding system for feature activation metrics in the feature browser.
 * Helps users quickly identify useful vs useless features for steering.
 */

export type QualityGrade = 'excellent' | 'good' | 'moderate' | 'poor' | 'bad';

export interface QualityColor {
  grade: QualityGrade;
  textClass: string;
  bgClass: string;
  label: string;
}

/**
 * Get quality grade and color for activation frequency.
 *
 * Activation frequency indicates how often a feature fires across samples:
 * - Too low (< 0.1%): Feature is essentially dead, rarely activates
 * - Sweet spot (0.5% - 10%): Selective features, good for steering
 * - Too high (> 30%): Feature fires too often, not selective
 *
 * @param frequency Activation frequency as decimal (0-1)
 * @returns Quality color information
 */
export function getActivationFrequencyColor(frequency: number): QualityColor {
  const percentage = frequency * 100;

  if (percentage < 0.1) {
    // Dead features - almost never activate
    return {
      grade: 'bad',
      textClass: 'text-red-400',
      bgClass: 'bg-red-900/30',
      label: 'Dead',
    };
  } else if (percentage < 0.5) {
    // Very sparse - might be too specialized
    return {
      grade: 'poor',
      textClass: 'text-orange-400',
      bgClass: 'bg-orange-900/30',
      label: 'Very sparse',
    };
  } else if (percentage < 1) {
    // Sparse but potentially useful
    return {
      grade: 'moderate',
      textClass: 'text-yellow-400',
      bgClass: 'bg-yellow-900/30',
      label: 'Sparse',
    };
  } else if (percentage <= 10) {
    // Sweet spot - selective and interpretable
    return {
      grade: 'excellent',
      textClass: 'text-emerald-400',
      bgClass: 'bg-emerald-900/30',
      label: 'Good',
    };
  } else if (percentage <= 30) {
    // Getting frequent
    return {
      grade: 'moderate',
      textClass: 'text-yellow-400',
      bgClass: 'bg-yellow-900/30',
      label: 'Frequent',
    };
  } else if (percentage <= 50) {
    // Too frequent
    return {
      grade: 'poor',
      textClass: 'text-orange-400',
      bgClass: 'bg-orange-900/30',
      label: 'Too frequent',
    };
  } else {
    // Way too frequent - not useful
    return {
      grade: 'bad',
      textClass: 'text-red-400',
      bgClass: 'bg-red-900/30',
      label: 'Overactive',
    };
  }
}

/**
 * Get quality grade and color for max activation value.
 *
 * Max activation indicates the peak signal strength of a feature:
 * - Low values may indicate weak/noisy features
 * - Higher values indicate clear, strong activations (good for steering)
 *
 * @param maxActivation Max activation value
 * @returns Quality color information
 */
export function getMaxActivationColor(maxActivation: number): QualityColor {
  if (maxActivation < 1.0) {
    // Very weak signal
    return {
      grade: 'bad',
      textClass: 'text-red-400',
      bgClass: 'bg-red-900/30',
      label: 'Weak',
    };
  } else if (maxActivation < 3.0) {
    // Moderate signal
    return {
      grade: 'moderate',
      textClass: 'text-yellow-400',
      bgClass: 'bg-yellow-900/30',
      label: 'Moderate',
    };
  } else if (maxActivation < 10.0) {
    // Good strong signal
    return {
      grade: 'excellent',
      textClass: 'text-emerald-400',
      bgClass: 'bg-emerald-900/30',
      label: 'Strong',
    };
  } else if (maxActivation < 30.0) {
    // Very strong signal
    return {
      grade: 'good',
      textClass: 'text-cyan-400',
      bgClass: 'bg-cyan-900/30',
      label: 'Very strong',
    };
  } else {
    // Extremely strong signal - use smaller steering coefficients
    return {
      grade: 'moderate',
      textClass: 'text-purple-400',
      bgClass: 'bg-purple-900/30',
      label: 'Extreme - use caution',
    };
  }
}

/**
 * Get combined quality assessment based on both metrics.
 * Features are most useful for steering when they have:
 * - Moderate activation frequency (selective)
 * - High max activation (strong signal)
 *
 * @param frequency Activation frequency (0-1)
 * @param maxActivation Max activation value
 * @returns Combined quality grade
 */
export function getFeatureQualityScore(frequency: number, maxActivation: number): QualityGrade {
  const freqColor = getActivationFrequencyColor(frequency);
  const maxActColor = getMaxActivationColor(maxActivation);

  // Both metrics need to be at least moderate for a good overall score
  const gradeOrder: Record<QualityGrade, number> = {
    excellent: 4,
    good: 3,
    moderate: 2,
    poor: 1,
    bad: 0,
  };

  const avgScore = (gradeOrder[freqColor.grade] + gradeOrder[maxActColor.grade]) / 2;

  if (avgScore >= 3.5) return 'excellent';
  if (avgScore >= 2.5) return 'good';
  if (avgScore >= 1.5) return 'moderate';
  if (avgScore >= 0.5) return 'poor';
  return 'bad';
}

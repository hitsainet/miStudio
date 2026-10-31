/**
 * TypeScript types for Neuronpedia Export functionality.
 */

// Export format options
export type ExportFormat = 'neuronpedia_json' | 'saelens' | 'both';

// Feature selection options
// - 'all': Export all features in the SAE
// - 'extracted': Export only features with extracted activation data
// - 'custom': Export specific feature indices
export type FeatureSelection = 'all' | 'extracted' | 'custom';

// Compression options
export type CompressionType = 'none' | 'gzip' | 'zip';

// Export job status
export type ExportStatus = 'pending' | 'computing' | 'packaging' | 'completed' | 'failed' | 'cancelled';

/**
 * Configuration for Neuronpedia export.
 */
export interface NeuronpediaExportConfig {
  // Feature selection
  featureSelection: FeatureSelection;
  featureIndices?: number[];
  minActivationFrequency?: number;
  maxActivationFrequency?: number;

  // Dashboard data options
  includeLogitLens: boolean;
  logitLensK: number;
  includeHistograms: boolean;
  histogramBins: number;
  includeTopTokens: boolean;
  topTokensK: number;

  // SAELens format options
  includeSaelensFormat: boolean;

  // Include explanations/labels
  includeExplanations: boolean;
}

/**
 * Request to start a Neuronpedia export.
 */
export interface NeuronpediaExportRequest {
  saeId: string;
  config: NeuronpediaExportConfig;
}

/**
 * Response from starting an export.
 */
export interface NeuronpediaExportJobResponse {
  jobId: string;
  status: ExportStatus;
  message: string;
}

/**
 * Full export job status.
 */
export interface NeuronpediaExportJob {
  id: string;
  saeId: string;
  status: ExportStatus;
  progress: number;
  currentStage?: string;
  featureCount?: number;
  createdAt: string;
  startedAt?: string;
  completedAt?: string;
  outputPath?: string;
  fileSizeBytes?: number;
  errorMessage?: string;
  downloadUrl?: string;
  config?: NeuronpediaExportConfig;
}

/**
 * List response for export jobs.
 */
export interface NeuronpediaExportJobListResponse {
  jobs: NeuronpediaExportJob[];
  total: number;
}

/**
 * Logit lens token data.
 */
export interface LogitLensToken {
  token: string;
  value: number;
}

/**
 * Logit lens data for a feature.
 */
export interface LogitLensData {
  topPositive: LogitLensToken[];
  topNegative: LogitLensToken[];
}

/**
 * Histogram data for feature activations.
 */
export interface HistogramData {
  binEdges: number[];
  counts: number[];
  totalCount: number;
  nonzeroCount: number;
  mean?: number;
  std?: number;
  min?: number;
  max?: number;
}

/**
 * Aggregated token data.
 */
export interface TokenAggregation {
  token: string;
  totalActivation: number;
  count: number;
  meanActivation: number;
  maxActivation: number;
}

/**
 * Complete dashboard data for a feature.
 */
export interface FeatureDashboardData {
  featureId: string;
  logitLensData?: LogitLensData;
  histogramData?: HistogramData;
  topTokens?: TokenAggregation[];
  computedAt?: string;
}

/**
 * Request to compute dashboard data.
 */
export interface ComputeDashboardDataRequest {
  saeId: string;
  featureIndices?: number[];
  includeLogitLens: boolean;
  includeHistograms: boolean;
  includeTopTokens: boolean;
  forceRecompute: boolean;
}

/**
 * Response from computing dashboard data.
 */
export interface ComputeDashboardDataResponse {
  featuresComputed: number;
  status: 'completed' | 'failed';
  message: string;
}

/**
 * Validation error for export.
 */
export interface ExportValidationError {
  code: string;
  message: string;
  field?: string;
  featureIndex?: number;
}

/**
 * Validation warning for export.
 */
export interface ExportValidationWarning {
  code: string;
  message: string;
  count?: number;
}

/**
 * Missing data summary.
 */
export interface MissingDataSummary {
  featuresWithoutLogitLens: number;
  featuresWithoutHistograms: number;
  featuresWithoutTopTokens: number;
  featuresWithoutLabels: number;
}

/**
 * Validation report for export.
 */
export interface ExportValidationReport {
  isValid: boolean;
  errors: ExportValidationError[];
  warnings: ExportValidationWarning[];
  missingData: MissingDataSummary;
  autoFixAvailable: boolean;
}

/**
 * WebSocket progress event for export.
 */
export interface ExportProgressEvent {
  jobId: string;
  progress: number;
  stage: string;
  status: ExportStatus;
  message?: string;
  featureCount?: number;
  outputPath?: string;
}

/**
 * Default export configuration.
 */
export const DEFAULT_EXPORT_CONFIG: NeuronpediaExportConfig = {
  featureSelection: 'all',
  includeLogitLens: true,
  logitLensK: 20,
  includeHistograms: true,
  histogramBins: 50,
  includeTopTokens: true,
  topTokensK: 50,
  includeSaelensFormat: true,
  includeExplanations: true,
};

/**
 * HuggingFace API client for external API calls.
 *
 * This module provides functions to query HuggingFace Hub and Dataset Viewer APIs
 * to get metadata about models and datasets before downloading them.
 *
 * API Documentation:
 * - Hub API: https://huggingface.co/docs/hub/en/api
 * - Dataset Viewer: https://huggingface.co/docs/datasets-server
 */

// HuggingFace API base URLs
const HF_HUB_API = 'https://huggingface.co/api';
const HF_DATASETS_SERVER_API = 'https://datasets-server.huggingface.co';

// Supported model architectures (based on backend error message)
const SUPPORTED_ARCHITECTURES = [
  'falcon',
  'gpt2',
  'gpt_neox',
  'llama',
  'mistral',
  'mixtral',
  'phi',
  'phi3',
  'phi3_v',
  'pythia',
  'qwen',
  'qwen3',
];

/**
 * Dataset split information from the /info endpoint
 */
export interface DatasetSplit {
  name: string;
  num_bytes: number;
  num_examples: number;
}

/**
 * Dataset feature information (column types)
 */
export interface DatasetFeature {
  [key: string]: any; // Feature structure can vary
}

/**
 * Dataset information response from /info endpoint
 */
export interface DatasetInfo {
  dataset_name: string;
  config_name: string;
  description?: string;
  citation?: string;
  homepage?: string;
  license?: string;
  features: DatasetFeature;
  splits: {
    [splitName: string]: {
      num_bytes: number;
      num_examples: number;
    };
  };
  download_size: number;
  dataset_size: number;
}

/**
 * Dataset splits response from /splits endpoint
 */
export interface DatasetSplitsResponse {
  splits: Array<{
    dataset: string;
    config: string;
    split: string;
  }>;
}

/**
 * Model information from Hub API
 */
export interface ModelInfo {
  id: string;
  modelId: string;
  author?: string;
  sha?: string;
  lastModified?: string;
  private: boolean;
  disabled?: boolean;
  gated?: boolean | 'auto' | 'manual';
  pipeline_tag?: string;
  tags: string[];
  downloads: number;
  library_name?: string;
  likes: number;
  config?: {
    model_type?: string;
    architectures?: string[];
    auto_map?: {
      [key: string]: string;
    };
    [key: string]: any;
  };
  cardData?: {
    language?: string[];
    license?: string;
    datasets?: string[];
    metrics?: string[];
    [key: string]: any;
  };
  siblings?: Array<{
    rfilename: string;
  }>;
  requiresTrustRemoteCode?: boolean;
  unsupportedArchitecture?: string | null;
}

/**
 * Fetch dataset information from HuggingFace Dataset Viewer API.
 *
 * @param repoId - Dataset repository ID (e.g., "openwebtext", "squad")
 * @param config - Optional configuration/subset name
 * @returns Dataset metadata including splits, features, sizes
 * @throws Error if the request fails or dataset is not found
 *
 * @example
 * ```typescript
 * const info = await getDatasetInfo('squad', 'plain_text');
 * console.log(info.splits); // { train: { num_examples: 87599, ... }, ... }
 * ```
 */
export async function getDatasetInfo(
  repoId: string,
  config?: string
): Promise<DatasetInfo> {
  const params = new URLSearchParams({ dataset: repoId });
  if (config) {
    params.append('config', config);
  }

  const url = `${HF_DATASETS_SERVER_API}/info?${params.toString()}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch dataset info: ${response.statusText}`);
  }

  const data = await response.json();

  // The API returns dataset_info with configs as keys (e.g., "default", "plain_text")
  // If no config specified, use the first available config
  const datasetInfo = data.dataset_info;
  const configName = config || Object.keys(datasetInfo)[0];
  const info = datasetInfo[configName];

  if (!info) {
    throw new Error(`Configuration "${configName}" not found for dataset`);
  }

  return info;
}

/**
 * Fetch available splits for a dataset.
 *
 * @param repoId - Dataset repository ID
 * @returns List of available splits with their configs
 * @throws Error if the request fails
 *
 * @example
 * ```typescript
 * const splits = await getDatasetSplits('squad');
 * // Result: { splits: [{ dataset: 'squad', config: 'plain_text', split: 'train' }, ...] }
 * ```
 */
export async function getDatasetSplits(
  repoId: string
): Promise<DatasetSplitsResponse> {
  const params = new URLSearchParams({ dataset: repoId });
  const url = `${HF_DATASETS_SERVER_API}/splits?${params.toString()}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch dataset splits: ${response.statusText}`);
  }

  return response.json();
}

/**
 * Fetch model information from HuggingFace Hub API.
 *
 * @param repoId - Model repository ID (e.g., "gpt2", "meta-llama/Llama-2-7b-hf")
 * @returns Model metadata including architecture, tags, config
 * @throws Error if the request fails or model is not found
 *
 * @example
 * ```typescript
 * const info = await getModelInfo('gpt2');
 * console.log(info.pipeline_tag); // "text-generation"
 * console.log(info.tags); // ["transformers", "pytorch", ...]
 * ```
 */
export async function getModelInfo(repoId: string): Promise<ModelInfo> {
  const url = `${HF_HUB_API}/models/${repoId}`;

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.statusText}`);
  }

  const modelInfo: ModelInfo = await response.json();

  // Detect if model requires trust_remote_code
  // This is indicated by:
  // 1. Having custom code files (modeling_*.py, configuration_*.py, etc.) in the repo
  // 2. Having auto_map in the config (which points to custom model classes)
  let requiresTrustRemoteCode = false;

  // Check for auto_map in config (most reliable indicator)
  if (modelInfo.config?.auto_map) {
    requiresTrustRemoteCode = true;
  }

  // Check for custom modeling files in siblings
  if (modelInfo.siblings && !requiresTrustRemoteCode) {
    const hasCustomCode = modelInfo.siblings.some((file) => {
      const filename = file.rfilename.toLowerCase();
      return (
        filename.startsWith('modeling_') ||
        filename.startsWith('configuration_') ||
        filename.startsWith('tokenization_')
      );
    });
    requiresTrustRemoteCode = hasCustomCode;
  }

  // Detect if model architecture is unsupported
  let unsupportedArchitecture: string | null = null;
  if (modelInfo.config?.model_type) {
    const modelType = modelInfo.config.model_type.toLowerCase();
    if (!SUPPORTED_ARCHITECTURES.includes(modelType)) {
      unsupportedArchitecture = modelInfo.config.model_type;
    }
  }

  return {
    ...modelInfo,
    requiresTrustRemoteCode,
    unsupportedArchitecture,
  };
}

/**
 * Calculate memory requirement for a model based on parameter count and quantization.
 *
 * @param paramsCount - Number of parameters (e.g., 1.1e9 for 1.1B params)
 * @param quantization - Quantization format (FP32, FP16, Q8, Q4, Q2)
 * @returns Memory requirement in bytes
 *
 * @example
 * ```typescript
 * const memory = calculateMemoryRequirement(1.1e9, 'Q4');
 * const memoryGB = memory / (1024 ** 3);
 * console.log(`${memoryGB.toFixed(2)} GB`); // "0.55 GB"
 * ```
 */
export function calculateMemoryRequirement(
  paramsCount: number,
  quantization: 'FP32' | 'FP16' | 'Q8' | 'Q4' | 'Q2'
): number {
  const bytesPerParam: Record<string, number> = {
    FP32: 4,
    FP16: 2,
    Q8: 1,
    Q4: 0.5,
    Q2: 0.25,
  };

  const baseMemory = paramsCount * bytesPerParam[quantization];
  // Add 20% overhead for inference
  return baseMemory * 1.2;
}

/**
 * Format bytes to human-readable size string.
 *
 * @param bytes - Number of bytes
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted string (e.g., "1.25 GB", "500 MB")
 *
 * @example
 * ```typescript
 * formatBytes(1234567890); // "1.15 GB"
 * formatBytes(500000); // "488.28 KB"
 * ```
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

/**
 * Training Template types and interfaces.
 *
 * These types match the backend Pydantic schemas for training template management.
 */

import { HyperparametersConfig, SAEArchitectureType } from './training';

/**
 * Training template for saving and reusing SAE training configurations.
 * Matches backend TrainingTemplate model.
 */
export interface TrainingTemplate {
  /** UUID identifier */
  id: string;
  /** Template name */
  name: string;
  /** Optional description */
  description?: string;
  /** Optional reference to specific model */
  model_id?: string | null;
  /** Dataset IDs for training (supports multiple) */
  dataset_ids: string[];
  /** Primary dataset ID (backward compat) */
  dataset_id?: string | null;
  /** SAE architecture type (standard/skip/transcoder) */
  encoder_type: string;
  /** Complete training hyperparameters */
  hyperparameters: HyperparametersConfig;
  /** Whether this template is marked as favorite */
  is_favorite: boolean;
  /** Additional JSON metadata */
  extra_metadata?: Record<string, any>;
  /** Creation timestamp */
  created_at: string;
  /** Last update timestamp */
  updated_at: string;
}

/**
 * Data required to create a new training template.
 * Matches backend TrainingTemplateCreate schema.
 */
export interface TrainingTemplateCreate {
  /** Template name (required) */
  name: string;
  /** Optional description */
  description?: string;
  /** Optional reference to specific model */
  model_id?: string | null;
  /** Dataset IDs for training (supports multiple) */
  dataset_ids: string[];
  /** SAE architecture type */
  encoder_type: SAEArchitectureType;
  /** Complete training hyperparameters */
  hyperparameters: HyperparametersConfig;
  /** Whether this template is marked as favorite (default: false) */
  is_favorite?: boolean;
  /** Additional JSON metadata */
  extra_metadata?: Record<string, any>;
}

/**
 * Data for updating an existing training template.
 * All fields are optional. Matches backend TrainingTemplateUpdate schema.
 */
export interface TrainingTemplateUpdate {
  /** Template name */
  name?: string;
  /** Description */
  description?: string;
  /** Optional reference to specific model */
  model_id?: string | null;
  /** Dataset IDs for training (supports multiple) */
  dataset_ids?: string[];
  /** SAE architecture type */
  encoder_type?: SAEArchitectureType;
  /** Training hyperparameters */
  hyperparameters?: HyperparametersConfig;
  /** Whether this template is marked as favorite */
  is_favorite?: boolean;
  /** Additional JSON metadata */
  extra_metadata?: Record<string, any>;
}

/**
 * Pagination metadata for list responses.
 */
export interface PaginationMetadata {
  /** Current page number (1-indexed) */
  page: number;
  /** Items per page */
  limit: number;
  /** Total number of items */
  total: number;
  /** Total number of pages */
  total_pages: number;
  /** Whether there is a next page */
  has_next: boolean;
  /** Whether there is a previous page */
  has_prev: boolean;
}

/**
 * API response for a single training template.
 */
export interface TrainingTemplateResponse {
  /** The training template */
  data?: TrainingTemplate;
}

/**
 * API response for list of training templates.
 */
export interface TrainingTemplateListResponse {
  /** Array of training templates */
  data: TrainingTemplate[];
  /** Pagination information */
  pagination?: PaginationMetadata;
}

/**
 * Export data structure for training templates.
 */
export interface TrainingTemplateExport {
  /** Export format version */
  version: string;
  /** ISO timestamp of export */
  exported_at: string;
  /** Array of templates */
  templates: TrainingTemplate[];
}

/**
 * Import result from importing training templates.
 */
export interface TrainingTemplateImportResult {
  /** Number of templates created */
  created: number;
  /** Number of templates updated */
  updated: number;
  /** Number of templates skipped (duplicates) */
  skipped: number;
  /** Total number of templates processed */
  total: number;
}

/**
 * Query parameters for listing training templates.
 */
export interface TrainingTemplateListParams {
  /** Page number (1-indexed) */
  page?: number;
  /** Items per page (1-100) */
  limit?: number;
  /** Search query for name or description */
  search?: string;
  /** Filter by favorite status */
  is_favorite?: boolean;
  /** Filter by encoder architecture type */
  encoder_type?: string;
  /** Sort by field (default: 'created_at') */
  sort_by?: 'name' | 'created_at' | 'updated_at' | 'encoder_type';
  /** Sort order (default: 'desc') */
  order?: 'asc' | 'desc';
}

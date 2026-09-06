/**
 * Prompt Template types and interfaces.
 *
 * These types match the backend Pydantic schemas for prompt template management.
 */

/**
 * Prompt template for saving and reusing prompt series for steering experiments.
 * Matches backend PromptTemplate model.
 */
export interface PromptTemplate {
  /** UUID identifier */
  id: string;
  /** Template name */
  name: string;
  /** Optional description */
  description?: string;
  /** Array of prompt strings */
  prompts: string[];
  /** Whether this template is marked as favorite */
  is_favorite: boolean;
  /** Tags for organization */
  tags: string[];
  /** Creation timestamp */
  created_at: string;
  /** Last update timestamp */
  updated_at: string;
}

/**
 * Data required to create a new prompt template.
 * Matches backend PromptTemplateCreate schema.
 */
export interface PromptTemplateCreate {
  /** Template name (required) */
  name: string;
  /** Optional description */
  description?: string;
  /** Array of prompt strings (at least one required) */
  prompts: string[];
  /** Whether this template is marked as favorite (default: false) */
  is_favorite?: boolean;
  /** Tags for organization */
  tags?: string[];
}

/**
 * Data for updating an existing prompt template.
 * All fields are optional. Matches backend PromptTemplateUpdate schema.
 */
export interface PromptTemplateUpdate {
  /** Template name */
  name?: string;
  /** Description */
  description?: string;
  /** Array of prompt strings */
  prompts?: string[];
  /** Whether this template is marked as favorite */
  is_favorite?: boolean;
  /** Tags for organization */
  tags?: string[];
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
 * API response for list of prompt templates.
 */
export interface PromptTemplateListResponse {
  /** Array of prompt templates */
  data: PromptTemplate[];
  /** Pagination information */
  pagination: PaginationMetadata;
}

/**
 * Export data structure for prompt templates.
 */
export interface PromptTemplateExport {
  /** Export format version */
  version: string;
  /** ISO timestamp of export */
  exported_at: string;
  /** Array of templates */
  templates: PromptTemplate[];
}

/**
 * Import result from importing prompt templates.
 */
export interface PromptTemplateImportResult {
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
 * Query parameters for listing prompt templates.
 */
export interface PromptTemplateListParams {
  /** Page number (1-indexed) */
  page?: number;
  /** Items per page (1-100) */
  limit?: number;
  /** Search query for name or description */
  search?: string;
  /** Filter by favorite status */
  is_favorite?: boolean;
  /** Filter by tag */
  tag?: string;
  /** Sort by field (default: 'created_at') */
  sort_by?: 'name' | 'created_at' | 'updated_at';
  /** Sort order (default: 'desc') */
  order?: 'asc' | 'desc';
}

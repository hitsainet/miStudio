/**
 * Extraction Template types and interfaces.
 *
 * These types match the backend Pydantic schemas for extraction template management.
 */

/**
 * Hook type enum for forward hook locations in transformer models.
 */
export enum HookType {
  RESIDUAL = 'residual',
  MLP = 'mlp',
  ATTENTION = 'attention',
}

/**
 * Extraction template for capturing activation patterns.
 * Matches backend ExtractionTemplate model.
 */
export interface ExtractionTemplate {
  /** UUID identifier */
  id: string;
  /** Template name */
  name: string;
  /** Optional description */
  description?: string;
  /** Array of layer indices to extract from (e.g., [0, 5, 11, 23]) */
  layer_indices: number[];
  /** Types of hooks to attach (residual, mlp, attention) */
  hook_types: HookType[] | string[];
  /** Maximum number of samples to process (default: 1000) */
  max_samples: number;
  /** Batch size for processing (default: 32) */
  batch_size: number;
  /** GPU micro-batch size for memory efficiency (defaults to batch_size if not specified) */
  micro_batch_size?: number;
  /** Number of top examples to save per feature (default: 10) */
  top_k_examples: number;
  /** Whether this template is marked as favorite */
  is_favorite: boolean;
  /** Number of tokens before max activation (context window prefix, default: 5) */
  context_prefix_tokens?: number;
  /** Number of tokens after max activation (context window suffix, default: 3) */
  context_suffix_tokens?: number;
  /** Enable token filtering during extraction (default: false) */
  extraction_filter_enabled?: boolean;
  /** Filter mode: minimal, conservative, standard, or aggressive (default: 'standard') */
  extraction_filter_mode?: 'minimal' | 'conservative' | 'standard' | 'aggressive';
  /** Additional JSON metadata */
  extra_metadata?: Record<string, any>;
  /** Creation timestamp */
  created_at: string;
  /** Last update timestamp */
  updated_at: string;
}

/**
 * Data required to create a new extraction template.
 * Matches backend ExtractionTemplateCreate schema.
 */
export interface ExtractionTemplateCreate {
  /** Template name (required) */
  name: string;
  /** Optional description */
  description?: string;
  /** Array of layer indices to extract from */
  layer_indices: number[];
  /** Types of hooks to attach */
  hook_types: (HookType | string)[];
  /** Maximum number of samples to process (default: 1000) */
  max_samples?: number;
  /** Batch size for processing (default: 32) */
  batch_size?: number;
  /** GPU micro-batch size for memory efficiency (defaults to batch_size if not specified) */
  micro_batch_size?: number;
  /** Number of top examples to save per feature (default: 10) */
  top_k_examples?: number;
  /** Whether this template is marked as favorite (default: false) */
  is_favorite?: boolean;
  /** Number of tokens before max activation (context window prefix, default: 5) */
  context_prefix_tokens?: number;
  /** Number of tokens after max activation (context window suffix, default: 3) */
  context_suffix_tokens?: number;
  /** Enable token filtering during extraction (default: false) */
  extraction_filter_enabled?: boolean;
  /** Filter mode: minimal, conservative, standard, or aggressive (default: 'standard') */
  extraction_filter_mode?: 'minimal' | 'conservative' | 'standard' | 'aggressive';
  /** Additional JSON metadata */
  extra_metadata?: Record<string, any>;
}

/**
 * Data for updating an existing extraction template.
 * All fields are optional. Matches backend ExtractionTemplateUpdate schema.
 */
export interface ExtractionTemplateUpdate {
  /** Template name */
  name?: string;
  /** Description */
  description?: string;
  /** Array of layer indices to extract from */
  layer_indices?: number[];
  /** Types of hooks to attach */
  hook_types?: (HookType | string)[];
  /** Maximum number of samples to process */
  max_samples?: number;
  /** Batch size for processing */
  batch_size?: number;
  /** GPU micro-batch size for memory efficiency */
  micro_batch_size?: number;
  /** Number of top examples to save per feature */
  top_k_examples?: number;
  /** Whether this template is marked as favorite */
  is_favorite?: boolean;
  /** Number of tokens before max activation (context window prefix) */
  context_prefix_tokens?: number;
  /** Number of tokens after max activation (context window suffix) */
  context_suffix_tokens?: number;
  /** Enable token filtering during extraction */
  extraction_filter_enabled?: boolean;
  /** Filter mode: minimal, conservative, standard, or aggressive */
  extraction_filter_mode?: 'minimal' | 'conservative' | 'standard' | 'aggressive';
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
 * API response for a single extraction template.
 */
export interface ExtractionTemplateResponse {
  /** The extraction template */
  data?: ExtractionTemplate;
}

/**
 * API response for list of extraction templates.
 */
export interface ExtractionTemplateListResponse {
  /** Array of extraction templates */
  data: ExtractionTemplate[];
  /** Pagination information */
  pagination?: PaginationMetadata;
}

/**
 * Export data structure for extraction templates.
 */
export interface ExtractionTemplateExport {
  /** Export format version */
  version: string;
  /** ISO timestamp of export */
  export_date: string;
  /** Number of templates in export */
  count: number;
  /** Array of template data (without IDs and timestamps) */
  templates: Omit<ExtractionTemplate, 'id' | 'created_at' | 'updated_at'>[];
}

/**
 * Import result from importing extraction templates.
 */
export interface ExtractionTemplateImportResult {
  /** Number of templates created */
  created: number;
  /** Number of templates updated */
  updated: number;
  /** Number of templates skipped (duplicates) */
  skipped: number;
  /** Total number of templates processed */
  total_processed: number;
  /** Array of errors encountered during import */
  errors: Array<{
    template_name: string;
    error: string;
  }>;
}

/**
 * Query parameters for listing extraction templates.
 */
export interface ExtractionTemplateListParams {
  /** Page number (1-indexed) */
  page?: number;
  /** Items per page (1-100) */
  limit?: number;
  /** Search query for name or description */
  search?: string;
  /** Filter by favorite status */
  is_favorite?: boolean;
  /** Sort by field (default: 'created_at') */
  sort_by?: 'name' | 'created_at' | 'updated_at';
  /** Sort order (default: 'desc') */
  order?: 'asc' | 'desc';
}

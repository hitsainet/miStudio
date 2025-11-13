/**
 * Labeling Prompt Template types and interfaces.
 *
 * These types match the backend Pydantic schemas for labeling prompt template management.
 */

/**
 * Labeling prompt template for semantic feature labeling.
 * Matches backend LabelingPromptTemplate model.
 */
export interface LabelingPromptTemplate {
  /** Template identifier (format: lpt_{uuid}) */
  id: string;
  /** Template name */
  name: string;
  /** Optional description */
  description?: string;
  /** System message for the LLM */
  system_message: string;
  /** User prompt template with {tokens_table} placeholder */
  user_prompt_template: string;
  /** Temperature for LLM generation (0.0-2.0) */
  temperature: number;
  /** Maximum tokens for LLM response (10-500) */
  max_tokens: number;
  /** Top-p sampling parameter (0.0-1.0) */
  top_p: number;
  /** Whether this template is the default */
  is_default: boolean;
  /** Whether this is a system template (cannot be deleted/modified) */
  is_system: boolean;
  /** User ID who created this template (optional) */
  created_by?: string | null;
  /** Creation timestamp */
  created_at: string;
  /** Last update timestamp */
  updated_at: string;
  /** Number of labeling jobs using this template (optional, from API) */
  usage_count?: number | null;
}

/**
 * Data required to create a new labeling prompt template.
 * Matches backend LabelingPromptTemplateCreate schema.
 */
export interface LabelingPromptTemplateCreate {
  /** Template name (required) */
  name: string;
  /** Optional description */
  description?: string;
  /** System message for the LLM */
  system_message: string;
  /** User prompt template with {tokens_table} placeholder (required) */
  user_prompt_template: string;
  /** Temperature for LLM generation (0.0-2.0, default: 0.3) */
  temperature?: number;
  /** Maximum tokens for LLM response (10-500, default: 50) */
  max_tokens?: number;
  /** Top-p sampling parameter (0.0-1.0, default: 0.9) */
  top_p?: number;
  /** Whether this template should be the default (default: false) */
  is_default?: boolean;
}

/**
 * Data for updating an existing labeling prompt template.
 * All fields are optional. Matches backend LabelingPromptTemplateUpdate schema.
 */
export interface LabelingPromptTemplateUpdate {
  /** Template name */
  name?: string;
  /** Description */
  description?: string;
  /** System message */
  system_message?: string;
  /** User prompt template */
  user_prompt_template?: string;
  /** Temperature */
  temperature?: number;
  /** Max tokens */
  max_tokens?: number;
  /** Top-p */
  top_p?: number;
  /** Whether this template should be the default */
  is_default?: boolean;
}

/**
 * API response for a single labeling prompt template.
 */
export interface LabelingPromptTemplateResponse {
  id: string;
  name: string;
  description?: string;
  system_message: string;
  user_prompt_template: string;
  temperature: number;
  max_tokens: number;
  top_p: number;
  is_default: boolean;
  is_system: boolean;
  created_by?: string | null;
  created_at: string;
  updated_at: string;
  usage_count?: number | null;
}

/**
 * API response for list of labeling prompt templates with pagination metadata.
 */
export interface LabelingPromptTemplateListResponse {
  /** Array of labeling prompt templates */
  data: LabelingPromptTemplate[];
  /** Pagination and query metadata */
  meta: {
    /** Current page number (1-indexed) */
    page: number;
    /** Items per page */
    limit: number;
    /** Total number of templates matching query */
    total: number;
    /** Total number of pages */
    total_pages: number;
    /** Whether there is a next page */
    has_next: boolean;
    /** Whether there is a previous page */
    has_prev: boolean;
  };
}

/**
 * API response for template deletion.
 */
export interface LabelingPromptTemplateDeleteResponse {
  /** Template ID that was deleted */
  id: string;
  /** Success message */
  message: string;
  /** Success indicator */
  success: boolean;
}

/**
 * API response for setting default template.
 */
export interface LabelingPromptTemplateSetDefaultResponse {
  /** Template ID that was set as default */
  id: string;
  /** Template name */
  name: string;
  /** Success message */
  message: string;
  /** Success indicator */
  success: boolean;
}

/**
 * Parameters for listing labeling prompt templates.
 */
export interface LabelingPromptTemplateListParams {
  /** Page number (1-indexed, default: 1) */
  page?: number;
  /** Items per page (default: 50, max: 100) */
  limit?: number;
  /** Search query for name or description */
  search?: string;
  /** Include system templates in results (default: true) */
  include_system?: boolean;
  /** Sort by field (default: created_at) */
  sort_by?: string;
  /** Sort order: asc or desc (default: desc) */
  order?: 'asc' | 'desc';
}

/**
 * Validation constraints for labeling prompt template fields.
 */
export const LabelingPromptTemplateConstraints = {
  name: {
    minLength: 1,
    maxLength: 255,
  },
  temperature: {
    min: 0.0,
    max: 2.0,
  },
  max_tokens: {
    min: 10,
    max: 500,
  },
  top_p: {
    min: 0.0,
    max: 1.0,
  },
  // Required placeholder in user_prompt_template
  requiredPlaceholder: '{tokens_table}',
} as const;

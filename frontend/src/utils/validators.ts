/**
 * Input validation utilities.
 *
 * This module provides validation functions for user inputs.
 */

/**
 * Validate HuggingFace repository ID format
 * @param repoId - Repository ID to validate (e.g., "username/dataset-name")
 * @returns True if valid, error message if invalid
 */
export function validateHfRepoId(repoId: string): true | string {
  if (!repoId || repoId.trim().length === 0) {
    return 'Repository ID is required';
  }

  if (!repoId.includes('/')) {
    return 'Repository ID must be in format "username/dataset-name"';
  }

  const [username, datasetName] = repoId.split('/');

  if (!username || username.trim().length === 0) {
    return 'Username is required';
  }

  if (!datasetName || datasetName.trim().length === 0) {
    return 'Dataset name is required';
  }

  // Check for valid characters (alphanumeric, hyphens, underscores)
  const validPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/;
  if (!validPattern.test(repoId)) {
    return 'Repository ID contains invalid characters';
  }

  return true;
}

/**
 * Validate tokenization settings
 * @param settings - Tokenization settings to validate
 * @returns True if valid, error message if invalid
 */
export function validateTokenizationSettings(settings: {
  max_length?: number;
  truncation?: boolean;
  padding?: boolean;
  add_special_tokens?: boolean;
}): true | string {
  if (settings.max_length !== undefined) {
    if (settings.max_length <= 0) {
      return 'Max length must be greater than 0';
    }

    if (settings.max_length > 100000) {
      return 'Max length cannot exceed 100,000 tokens';
    }

    if (!Number.isInteger(settings.max_length)) {
      return 'Max length must be an integer';
    }
  }

  return true;
}

/**
 * Validate dataset name
 * @param name - Dataset name to validate
 * @returns True if valid, error message if invalid
 */
export function validateDatasetName(name: string): true | string {
  if (!name || name.trim().length === 0) {
    return 'Dataset name is required';
  }

  if (name.length < 3) {
    return 'Dataset name must be at least 3 characters';
  }

  if (name.length > 255) {
    return 'Dataset name cannot exceed 255 characters';
  }

  return true;
}

/**
 * Validate file path
 * @param path - File path to validate
 * @returns True if valid, error message if invalid
 */
export function validateFilePath(path: string): true | string {
  if (!path || path.trim().length === 0) {
    return 'File path is required';
  }

  // Check for absolute path
  if (!path.startsWith('/') && !path.match(/^[a-zA-Z]:\\/)) {
    return 'Path must be an absolute path';
  }

  // Check for invalid characters
  const invalidChars = ['<', '>', '"', '|', '\0'];
  for (const char of invalidChars) {
    if (path.includes(char)) {
      return `Path contains invalid character: ${char}`;
    }
  }

  return true;
}

/**
 * Feature Example Formatter Utility
 *
 * Formats feature activation examples for clipboard export in various formats.
 * Supports plain text, markdown, and JSON output formats.
 */

import { FeatureActivationExample } from '../types/features';

export type ExportFormat = 'text' | 'markdown' | 'json';

/**
 * Clean token by removing BPE markers from various tokenizers.
 * Same logic as TokenHighlight component.
 */
const cleanToken = (token: string): string => {
  let cleaned = token.replace(/^["']|["']$/g, '');
  cleaned = cleaned
    .replace(/^Ġ/g, '')
    .replace(/^▁/g, '')
    .replace(/^\u2581/g, '')
    .replace(/^##/g, '')
    .replace(/^_/g, '');
  return cleaned;
};

/**
 * Get the display text for an example's tokens.
 * Handles both legacy (flat tokens) and enhanced (prefix/prime/suffix) formats.
 */
function getTokensDisplay(
  example: FeatureActivationExample,
  highlightPrime: boolean = true
): string {
  // Enhanced format with context window
  if (example.prefix_tokens && example.prime_token !== undefined && example.suffix_tokens) {
    const prefix = example.prefix_tokens.map(cleanToken).join(' ');
    const prime = cleanToken(example.prime_token);
    const suffix = example.suffix_tokens.map(cleanToken).join(' ');

    if (highlightPrime) {
      return `${prefix} [${prime}] ${suffix}`.trim();
    }
    return `${prefix} ${prime} ${suffix}`.trim();
  }

  // Legacy format - flat token list
  if (example.tokens && example.tokens.length > 0) {
    // Find the max activation index to highlight
    if (highlightPrime && example.activations && example.activations.length > 0) {
      const maxIdx = example.activations.indexOf(Math.max(...example.activations));
      return example.tokens
        .map((t, i) => {
          const cleaned = cleanToken(t);
          return i === maxIdx ? `[${cleaned}]` : cleaned;
        })
        .join(' ');
    }
    return example.tokens.map(cleanToken).join(' ');
  }

  return '';
}

/**
 * Format a single example as plain text.
 */
function formatExampleAsText(
  example: FeatureActivationExample,
  _index: number
): string {
  const lines: string[] = [];

  lines.push(`Sample #${example.sample_index} (Max: ${example.max_activation.toFixed(3)})`);
  lines.push(getTokensDisplay(example, true));
  lines.push('');

  return lines.join('\n');
}

/**
 * Format a single example as markdown (single line with pipe separators).
 */
function formatExampleAsMarkdown(
  example: FeatureActivationExample,
  index: number
): string {
  // Format tokens with bold for prime token
  let tokensStr = '';
  if (example.prefix_tokens && example.prime_token !== undefined && example.suffix_tokens) {
    const prefix = example.prefix_tokens.map(cleanToken).join(' ');
    const prime = cleanToken(example.prime_token);
    const suffix = example.suffix_tokens.map(cleanToken).join(' ');
    tokensStr = `${prefix} **[${prime}]** ${suffix}`.trim();
  } else if (example.tokens) {
    const maxIdx = example.activations?.indexOf(Math.max(...(example.activations || []))) ?? -1;
    tokensStr = example.tokens
      .map((t, i) => {
        const cleaned = cleanToken(t);
        return i === maxIdx ? `**[${cleaned}]**` : cleaned;
      })
      .join(' ');
  }

  // Single line format with pipe separators
  return `### Example ${index + 1} - Sample #${example.sample_index}|**Max Activation:** \`${example.max_activation.toFixed(3)}\`|> ${tokensStr}`;
}

/**
 * Format examples for clipboard export.
 *
 * @param examples - Array of feature activation examples
 * @param count - Number of examples to include (or 'all' for all examples)
 * @param format - Output format (text, markdown, or json)
 * @param featureInfo - Optional feature metadata for header
 * @returns Formatted string ready for clipboard
 */
export function formatExamplesForClipboard(
  examples: FeatureActivationExample[],
  count: number | 'all',
  format: ExportFormat,
  featureInfo?: {
    featureIndex?: number;
    featureName?: string;
    maxActivation?: number;
  }
): string {
  const numExamples = count === 'all' ? examples.length : Math.min(count, examples.length);
  const selectedExamples = examples.slice(0, numExamples);

  if (selectedExamples.length === 0) {
    return 'No examples available.';
  }

  switch (format) {
    case 'json':
      return formatAsJson(selectedExamples, featureInfo);
    case 'markdown':
      return formatAsMarkdown(selectedExamples, featureInfo);
    case 'text':
    default:
      return formatAsText(selectedExamples, featureInfo);
  }
}

/**
 * Format all examples as plain text.
 */
function formatAsText(
  examples: FeatureActivationExample[],
  featureInfo?: {
    featureIndex?: number;
    featureName?: string;
    maxActivation?: number;
  }
): string {
  const lines: string[] = [];

  // Header
  if (featureInfo) {
    if (featureInfo.featureIndex !== undefined) {
      lines.push(`Feature #${featureInfo.featureIndex}`);
    }
    if (featureInfo.featureName) {
      lines.push(`Name: ${featureInfo.featureName}`);
    }
    if (featureInfo.maxActivation !== undefined) {
      lines.push(`Max Activation: ${featureInfo.maxActivation.toFixed(3)}`);
    }
    lines.push('');
  }

  lines.push(`Max-Activating Examples (${examples.length})`);
  lines.push('='.repeat(40));
  lines.push('');

  // Examples
  examples.forEach((example, index) => {
    lines.push(formatExampleAsText(example, index));
  });

  return lines.join('\n');
}

/**
 * Format all examples as markdown (single-line format with pipe separators).
 */
function formatAsMarkdown(
  examples: FeatureActivationExample[],
  featureInfo?: {
    featureIndex?: number;
    featureName?: string;
    maxActivation?: number;
  }
): string {
  const lines: string[] = [];

  // Header - single line with pipe separators
  const headerParts: string[] = [];
  if (featureInfo?.featureIndex !== undefined) {
    headerParts.push(`# Feature #${featureInfo.featureIndex}`);
  } else {
    headerParts.push('# Feature Activation Examples');
  }
  if (featureInfo?.featureName) {
    headerParts.push(`**Name:** ${featureInfo.featureName}`);
  }
  if (featureInfo?.maxActivation !== undefined) {
    headerParts.push(`**Max Activation:** \`${featureInfo.maxActivation.toFixed(3)}\``);
  }
  headerParts.push(`## Max-Activating Examples (${examples.length})`);
  lines.push(headerParts.join('|'));

  // Examples - each on single line
  examples.forEach((example, index) => {
    lines.push(formatExampleAsMarkdown(example, index));
  });

  return lines.join('\n');
}

/**
 * Format all examples as JSON.
 */
function formatAsJson(
  examples: FeatureActivationExample[],
  featureInfo?: {
    featureIndex?: number;
    featureName?: string;
    maxActivation?: number;
  }
): string {
  const output = {
    feature: featureInfo || {},
    examples: examples.map((ex, idx) => ({
      index: idx + 1,
      sample_index: ex.sample_index,
      max_activation: ex.max_activation,
      // Include context format if available
      ...(ex.prefix_tokens && ex.prime_token !== undefined && ex.suffix_tokens
        ? {
            prefix_tokens: ex.prefix_tokens.map(cleanToken),
            prime_token: cleanToken(ex.prime_token),
            suffix_tokens: ex.suffix_tokens.map(cleanToken),
          }
        : {
            tokens: ex.tokens.map(cleanToken),
          }),
      activations: ex.activations,
    })),
    exported_at: new Date().toISOString(),
    count: examples.length,
  };

  return JSON.stringify(output, null, 2);
}

/**
 * Format a single example for clipboard export (plain text).
 *
 * @param example - The feature activation example
 * @param index - The example index (0-based)
 * @returns Formatted string ready for clipboard
 */
export function formatSingleExampleForClipboard(
  example: FeatureActivationExample,
  index: number
): string {
  const lines: string[] = [];

  lines.push(`Example ${index + 1} - Sample #${example.sample_index}`);
  lines.push(`Max Activation: ${example.max_activation.toFixed(3)}`);
  lines.push('');
  lines.push(getTokensDisplay(example, true));

  return lines.join('\n');
}

/**
 * Copy text to clipboard with fallback for older browsers.
 */
export async function copyToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }

    // Fallback for older browsers
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.left = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    const success = document.execCommand('copy');
    document.body.removeChild(textarea);
    return success;
  } catch (error) {
    console.error('Failed to copy to clipboard:', error);
    return false;
  }
}

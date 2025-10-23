/**
 * TokenHighlight Component
 *
 * Displays tokens with activation-based highlighting.
 * Implements the exact styling from Mock UI specification:
 * - Background intensity: rgba(16, 185, 129, intensity * 0.4)
 * - Text color: white if intensity > 0.6, slate-300 otherwise
 * - Border: 1px emerald if intensity > 0.7
 * - Tooltip on hover showing activation value
 */

import React from 'react';

interface TokenHighlightProps {
  tokens: string[];
  activations: number[];
  maxActivation: number;
  className?: string;
}

/**
 * Calculate activation intensity (0-1) for a token.
 */
const calculateIntensity = (activation: number, maxActivation: number): number => {
  if (maxActivation === 0) return 0;
  return activation / maxActivation;
};

/**
 * Get background color based on activation intensity.
 * Uses emerald color with alpha based on intensity.
 */
const getBackgroundColor = (intensity: number): string => {
  // rgba(16, 185, 129, intensity * 0.4)
  const alpha = intensity * 0.4;
  return `rgba(16, 185, 129, ${alpha})`;
};

/**
 * Get text color based on activation intensity.
 */
const getTextColor = (intensity: number): string => {
  return intensity > 0.6 ? 'text-white' : 'text-slate-300';
};

/**
 * Get border class based on activation intensity.
 */
const getBorderClass = (intensity: number): string => {
  return intensity > 0.7 ? 'border border-emerald-500' : '';
};

/**
 * TokenHighlight component.
 * Renders a sequence of tokens with activation-based highlighting.
 */
export const TokenHighlight: React.FC<TokenHighlightProps> = ({
  tokens,
  activations,
  maxActivation,
  className = '',
}) => {
  if (tokens.length !== activations.length) {
    console.warn('TokenHighlight: tokens and activations length mismatch');
    return null;
  }

  return (
    <div className={`flex flex-wrap gap-1 font-mono text-xs ${className}`}>
      {tokens.map((token, index) => {
        const activation = activations[index];
        const intensity = calculateIntensity(activation, maxActivation);
        const backgroundColor = getBackgroundColor(intensity);
        const textColor = getTextColor(intensity);
        const borderClass = getBorderClass(intensity);

        return (
          <span
            key={index}
            className={`relative group px-1 py-0.5 rounded cursor-help ${textColor} ${borderClass}`}
            style={{ backgroundColor }}
            title={`Activation: ${activation.toFixed(3)}`}
          >
            {token}
            {/* Tooltip */}
            <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
              Activation: {activation.toFixed(3)}
            </span>
          </span>
        );
      })}
    </div>
  );
};

/**
 * Compact version of TokenHighlight for use in lists.
 * Shows max 20 tokens with ellipsis.
 */
export const TokenHighlightCompact: React.FC<TokenHighlightProps & { maxTokens?: number }> = ({
  tokens,
  activations,
  maxActivation,
  maxTokens = 20,
  className = '',
}) => {
  const displayTokens = tokens.slice(0, maxTokens);
  const displayActivations = activations.slice(0, maxTokens);
  const hasMore = tokens.length > maxTokens;

  return (
    <div className={`flex flex-wrap gap-1 font-mono text-xs max-w-md ${className}`}>
      {displayTokens.map((token, index) => {
        const activation = displayActivations[index];
        const intensity = calculateIntensity(activation, maxActivation);
        const backgroundColor = getBackgroundColor(intensity);
        const textColor = getTextColor(intensity);
        const borderClass = getBorderClass(intensity);

        return (
          <span
            key={index}
            className={`relative group px-1 py-0.5 rounded cursor-help ${textColor} ${borderClass}`}
            style={{ backgroundColor }}
            title={`Activation: ${activation.toFixed(3)}`}
          >
            {token}
            {/* Tooltip */}
            <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10">
              Activation: {activation.toFixed(3)}
            </span>
          </span>
        );
      })}
      {hasMore && (
        <span className="text-slate-500 px-1">
          ... (+{tokens.length - maxTokens} more)
        </span>
      )}
    </div>
  );
};

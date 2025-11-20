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
 * Clean token by removing BPE markers from GPT-2/BERT tokenizers.
 * - "Ġ" = GPT-2 space marker (indicates token starts a new word)
 * - "▁" (U+2581) = SentencePiece/T5 space marker
 * - "##" = BERT word-piece continuation marker
 * - "\u2581" = Escaped Unicode representation of SentencePiece marker
 */
const cleanToken = (token: string): string => {
  // First remove surrounding quotes if present
  let cleaned = token.replace(/^["']|["']$/g, '');

  // Then remove BPE markers
  cleaned = cleaned
    .replace(/^Ġ/g, '')  // GPT-2 space marker → remove
    .replace(/^▁/g, '')  // SentencePiece space marker (U+2581) → remove
    .replace(/^\u2581/g, '')  // Escaped SentencePiece marker → remove
    .replace(/^##/g, '')  // BERT continuation marker → remove
    .replace(/^_/g, '');  // Underscore fallback → remove

  return cleaned;
};

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
        const cleanedToken = cleanToken(token);

        return (
          <span
            key={index}
            className={`relative group px-1 py-0.5 rounded cursor-help ${textColor} ${borderClass}`}
            style={{ backgroundColor }}
            title={`Activation: ${activation.toFixed(3)}`}
          >
            {cleanedToken}
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
        const cleanedToken = cleanToken(token);

        return (
          <span
            key={index}
            className={`relative group px-1 py-0.5 rounded cursor-help ${textColor} ${borderClass}`}
            style={{ backgroundColor }}
            title={`Activation: ${activation.toFixed(3)}`}
          >
            {cleanedToken}
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

/**
 * Enhanced TokenHighlight with context window support.
 * Shows prefix, prime, and suffix tokens with visual hierarchy.
 */
interface TokenHighlightContextProps {
  // Enhanced format (preferred)
  prefixTokens?: string[];
  primeToken?: string;
  suffixTokens?: string[];
  activations: number[];
  maxActivation: number;
  primeActivationIndex?: number;
  // Legacy format (fallback)
  tokens?: string[];
  className?: string;
  showGradient?: boolean;
}

export const TokenHighlightContext: React.FC<TokenHighlightContextProps> = ({
  prefixTokens,
  primeToken,
  suffixTokens,
  activations,
  maxActivation,
  primeActivationIndex,
  tokens,
  className = '',
  showGradient = true,
}) => {
  // Use enhanced format if available, otherwise fall back to legacy
  const hasContextFormat = prefixTokens !== undefined && primeToken !== undefined && suffixTokens !== undefined;

  if (!hasContextFormat && !tokens) {
    console.warn('TokenHighlightContext: No tokens provided');
    return null;
  }

  // Legacy format - use original component
  if (!hasContextFormat && tokens) {
    return <TokenHighlight tokens={tokens} activations={activations} maxActivation={maxActivation} className={className} />;
  }

  // Enhanced format - render with context window structure
  const allTokens = [...(prefixTokens || []), primeToken || '', ...(suffixTokens || [])];
  const primeIndex = primeActivationIndex ?? (prefixTokens?.length ?? 0);

  if (allTokens.length !== activations.length) {
    console.warn('TokenHighlightContext: tokens and activations length mismatch');
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {/* Token display with visual hierarchy */}
      <div className="flex flex-wrap gap-1 font-mono text-xs">
        {allTokens.map((token, index) => {
          const activation = activations[index] || 0;
          const intensity = calculateIntensity(activation, maxActivation);
          const backgroundColor = getBackgroundColor(intensity);
          const textColor = getTextColor(intensity);
          const borderClass = getBorderClass(intensity);
          const cleanedToken = cleanToken(token);

          // Determine token role
          const isPrime = index === primeIndex;
          const isPrefix = index < primeIndex;

          // Enhanced styling for prime token
          const roleClass = isPrime
            ? 'ring-2 ring-emerald-400 ring-offset-1 ring-offset-slate-900 font-bold scale-110'
            : 'opacity-80';

          return (
            <span
              key={index}
              className={`relative group px-1.5 py-1 rounded cursor-help transition-all ${textColor} ${borderClass} ${roleClass}`}
              style={{ backgroundColor }}
              title={`${isPrime ? '★ PRIME TOKEN\n' : ''}Activation: ${activation.toFixed(3)}\nPosition: ${isPrefix ? 'prefix' : isPrime ? 'prime' : 'suffix'}`}
            >
              {cleanedToken}
              {/* Enhanced tooltip */}
              <span className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-slate-900 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap pointer-events-none z-10 border border-slate-700">
                {isPrime && <div className="font-bold text-emerald-400 mb-1">★ PRIME TOKEN</div>}
                <div>Activation: {activation.toFixed(3)}</div>
                <div className="text-slate-400 text-[10px]">
                  {isPrefix ? 'Context (prefix)' : isPrime ? 'Maximum activation' : 'Context (suffix)'}
                </div>
              </span>
            </span>
          );
        })}
      </div>

      {/* Activation gradient visualization */}
      {showGradient && (
        <div className="space-y-1">
          <div className="flex items-center gap-2 text-[10px] text-slate-500">
            <span>Activation Strength</span>
            <div className="flex-1 h-px bg-slate-700"></div>
          </div>
          <div className="flex gap-0.5 h-2">
            {activations.map((activation, index) => {
              const intensity = calculateIntensity(activation, maxActivation);
              const isPrime = index === primeIndex;

              return (
                <div
                  key={index}
                  className={`flex-1 rounded-sm transition-all ${isPrime ? 'ring-1 ring-emerald-400' : ''}`}
                  style={{
                    backgroundColor: `rgba(16, 185, 129, ${intensity})`,
                    minWidth: '2px'
                  }}
                  title={`Token ${index + 1}: ${activation.toFixed(3)}`}
                />
              );
            })}
          </div>
          {/* Context window labels */}
          <div className="flex text-[9px] text-slate-600 font-mono">
            <div className="flex-1 text-left">← prefix ({prefixTokens?.length || 0})</div>
            <div className="text-center text-emerald-400">prime</div>
            <div className="flex-1 text-right">suffix ({suffixTokens?.length || 0}) →</div>
          </div>
        </div>
      )}
    </div>
  );
};

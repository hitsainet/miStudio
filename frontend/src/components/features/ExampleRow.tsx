/**
 * ExampleRow Component
 *
 * Displays a single feature activation example in Neuronpedia-style layout:
 * - Left sub-stack: Prime token on top, activation value below (fixed-width)
 * - Center: Context tokens (prefix + prime + suffix) with wrapping
 * - Right: Copy button
 */

import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { FeatureActivationExample } from '../../types/features';
import { formatActivation } from '../../utils/formatters';
import { formatSingleExampleForClipboard, copyToClipboard } from '../../utils/featureExampleFormatter';
import { cleanToken, getPrimeWord, hasMarkers, isWordStart } from '../../utils/tokenUtils';

interface ExampleRowProps {
  example: FeatureActivationExample;
  index: number;
  maxActivation?: number; // Overall max activation for the feature (for intensity normalization)
}

/**
 * Calculate activation intensity (0-1) for a token.
 */
const calculateIntensity = (activation: number, maxActivation: number): number => {
  if (maxActivation === 0) return 0;
  return Math.min(activation / maxActivation, 1);
};

/**
 * Get background color based on activation intensity.
 */
const getBackgroundColor = (intensity: number): string => {
  const alpha = intensity * 0.7;
  return `rgba(16, 185, 129, ${alpha})`;
};

export const ExampleRow: React.FC<ExampleRowProps> = ({
  example,
  index,
  maxActivation,
}) => {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'success'>('idle');

  // Determine the max activation to use for intensity calculation
  const effectiveMaxActivation = maxActivation ?? example.max_activation;

  // Get prime token/word info
  const getPrimeTokenInfo = () => {
    // Enhanced format: use word reconstruction to get the full word
    if (example.prefix_tokens && example.prime_token !== undefined && example.suffix_tokens) {
      const primeWord = getPrimeWord(
        example.prefix_tokens,
        example.prime_token,
        example.suffix_tokens
      );
      return {
        token: primeWord,
        rawToken: cleanToken(example.prime_token),
        primeIndex: example.prime_activation_index ?? example.prefix_tokens.length,
        isFragment: primeWord !== cleanToken(example.prime_token),
      };
    }
    // Legacy format: find max activation index (no reconstruction available)
    if (example.tokens && example.activations) {
      const maxIdx = example.activations.indexOf(Math.max(...example.activations));
      const token = cleanToken(example.tokens[maxIdx] || '');
      return {
        token,
        rawToken: token,
        primeIndex: maxIdx,
        isFragment: false,
      };
    }
    return { token: '', rawToken: '', primeIndex: 0, isFragment: false };
  };

  const { token: primeWord, rawToken: rawPrimeToken, primeIndex, isFragment } = getPrimeTokenInfo();

  // Build all tokens array for rendering
  const getAllTokens = (): string[] => {
    if (example.prefix_tokens && example.prime_token !== undefined && example.suffix_tokens) {
      return [...example.prefix_tokens, example.prime_token, ...example.suffix_tokens];
    }
    return example.tokens || [];
  };

  const allTokens = getAllTokens();

  // Compute word boundaries for token grouping
  // This determines which tokens should be visually connected (no gap)
  const getTokenWordInfo = () => {
    if (allTokens.length === 0) return { tokenToWordMap: [], useHeuristics: false };

    // Check if tokens have BPE markers
    const useHeuristics = !hasMarkers(allTokens);

    // Build token-to-word mapping
    const tokenToWordMap: number[] = [];
    let currentWordIndex = 0;

    for (let i = 0; i < allTokens.length; i++) {
      const token = allTokens[i];
      const startsNewWord = i === 0 || isWordStart(token, i, allTokens, useHeuristics);

      if (startsNewWord && i > 0) {
        currentWordIndex++;
      }
      tokenToWordMap.push(currentWordIndex);
    }

    return { tokenToWordMap, useHeuristics };
  };

  const { tokenToWordMap } = getTokenWordInfo();

  // Check if token is the last in its word (used for spacing)
  const isLastTokenInWord = (idx: number): boolean => {
    if (idx >= allTokens.length - 1) return true;
    return tokenToWordMap[idx] !== tokenToWordMap[idx + 1];
  };

  // Handle copy
  const handleCopy = async () => {
    const text = formatSingleExampleForClipboard(example, index);
    const success = await copyToClipboard(text);
    if (success) {
      setCopyStatus('success');
      setTimeout(() => setCopyStatus('idle'), 2000);
    }
  };

  return (
    <div className="flex items-start gap-3 py-3 border-b border-slate-800 last:border-b-0 hover:bg-slate-800/30 transition-colors group">
      {/* Left column: Prime word + activation (fixed width) */}
      <div className="flex-shrink-0 w-24 flex flex-col items-center text-center">
        {/* Prime word (reconstructed from BPE tokens) */}
        <div
          className="px-2 py-1 rounded text-sm font-mono font-semibold text-white truncate max-w-full"
          style={{ backgroundColor: 'rgba(16, 185, 129, 0.7)' }}
          title={isFragment
            ? `Word: "${primeWord}" (token: "${rawPrimeToken}")`
            : primeWord}
        >
          {primeWord || '·'}
        </div>
        {/* Fragment indicator - shows if prime token was part of a larger word */}
        {isFragment && (
          <div className="text-[10px] text-slate-400 mt-0.5 truncate max-w-full" title={`Token: "${rawPrimeToken}"`}>
            ← {rawPrimeToken}
          </div>
        )}
        {/* Activation value */}
        <div className="text-xs text-emerald-400 font-mono mt-1">
          {formatActivation(example.max_activation)}
        </div>
      </div>

      {/* Center: Token context (wrapping) */}
      <div className="flex-1 min-w-0">
        <div className="flex flex-wrap font-mono text-sm leading-relaxed">
          {allTokens.map((token, idx) => {
            const activation = example.activations[idx] || 0;
            const intensity = calculateIntensity(activation, effectiveMaxActivation);
            const backgroundColor = getBackgroundColor(intensity);
            const isPrime = idx === primeIndex;
            const cleanedToken = cleanToken(token);
            const lastInWord = isLastTokenInWord(idx);

            // Text color based on intensity
            const textColorClass = intensity > 0.6 ? 'text-white' : 'text-slate-300';
            // Border for high activations
            const borderClass = intensity > 0.7 ? 'border border-emerald-500' : '';
            // Prime token styling
            const primeClass = isPrime ? 'ring-2 ring-emerald-400 ring-offset-1 ring-offset-slate-900 font-bold' : '';
            // Spacing: only add margin after the last token in a word
            const spacingClass = lastInWord ? 'mr-1' : 'mr-0';
            // Rounding: round edges differently for word fragments
            const roundingClass = lastInWord ? 'rounded-r' : '';
            const roundingLeftClass = idx === 0 || tokenToWordMap[idx] !== tokenToWordMap[idx - 1] ? 'rounded-l' : '';

            return (
              <span
                key={idx}
                className={`relative px-1.5 py-0.5 cursor-help ${textColorClass} ${borderClass} ${primeClass} ${spacingClass} ${roundingClass} ${roundingLeftClass}`}
                style={{ backgroundColor }}
                title={`Activation: ${activation.toFixed(3)}${isPrime ? ' (PRIME)' : ''}`}
              >
                {cleanedToken || '·'}
              </span>
            );
          })}
        </div>
        {/* Sample index */}
        <div className="text-xs text-slate-500 mt-1">
          Sample #{example.sample_index}
        </div>
      </div>

      {/* Right: Copy button */}
      <div className="flex-shrink-0">
        <button
          onClick={handleCopy}
          className={`p-1.5 rounded transition-colors ${
            copyStatus === 'success'
              ? 'bg-emerald-600 text-white'
              : 'text-slate-500 hover:text-slate-300 hover:bg-slate-700 opacity-0 group-hover:opacity-100'
          }`}
          title="Copy example"
        >
          {copyStatus === 'success' ? (
            <Check className="w-4 h-4" />
          ) : (
            <Copy className="w-4 h-4" />
          )}
        </button>
      </div>
    </div>
  );
};

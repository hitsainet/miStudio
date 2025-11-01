/**
 * MaxActivatingExamples Component
 *
 * Displays max-activating examples for a feature.
 * Shows token sequences with activation-based highlighting.
 *
 * Implements Mock UI specification (lines 2728-2800+):
 * - Example cards with bg-slate-800/30
 * - Example number and max activation display
 * - Token highlighting using TokenHighlight component
 */

import React, { useEffect, useState } from 'react';
import { TokenHighlight } from './TokenHighlight';
import { FeatureActivationExample } from '../../types/features';
import { fetchAPI, buildQueryString } from '../../api/client';

interface MaxActivatingExamplesProps {
  featureId: string;
  limit?: number;
}

/**
 * MaxActivatingExamples component.
 * Fetches and displays max-activating examples for a feature.
 */
export const MaxActivatingExamples: React.FC<MaxActivatingExamplesProps> = ({
  featureId,
  limit = 100,
}) => {
  const [examples, setExamples] = useState<FeatureActivationExample[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchExamples = async () => {
      try {
        setLoading(true);
        setError(null);
        const query = buildQueryString({ limit });
        const data = await fetchAPI<FeatureActivationExample[]>(
          `/features/${featureId}/examples?${query}`
        );
        setExamples(data);
      } catch (err) {
        console.error('Error fetching feature examples:', err);
        setError('Failed to load examples');
      } finally {
        setLoading(false);
      }
    };

    fetchExamples();
  }, [featureId, limit]);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-slate-400">Loading examples...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-red-400">{error}</div>
      </div>
    );
  }

  if (examples.length === 0) {
    return (
      <div className="flex items-center justify-center py-8">
        <div className="text-slate-400">No examples found</div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="text-sm text-slate-400">
        Showing {examples.length} example{examples.length !== 1 ? 's' : ''}
      </div>

      {/* Example Cards */}
      <div className="space-y-4">
        {examples.map((example, index) => (
          <div
            key={index}
            className="bg-slate-800/30 rounded-lg p-4 space-y-2"
          >
            {/* Example Header */}
            <div className="flex items-center justify-between">
              <div className="text-sm font-medium text-slate-300">
                Example {index + 1}
              </div>
              <div className="text-sm">
                <span className="text-slate-400">Max Activation: </span>
                <span className="text-emerald-400 font-mono">
                  {example.max_activation.toFixed(3)}
                </span>
              </div>
            </div>

            {/* Token Sequence with Highlighting */}
            <TokenHighlight
              tokens={example.tokens}
              activations={example.activations}
              maxActivation={example.max_activation}
              className="py-2"
            />

            {/* Sample Index (optional debugging info) */}
            {import.meta.env.DEV && (
              <div className="text-xs text-slate-500">
                Sample index: {example.sample_index}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

/**
 * Compact version of MaxActivatingExamples for modal sidebars.
 * Shows only top 5 examples with truncated tokens.
 */
export const MaxActivatingExamplesCompact: React.FC<MaxActivatingExamplesProps> = ({
  featureId,
  limit = 5,
}) => {
  const [examples, setExamples] = useState<FeatureActivationExample[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchExamples = async () => {
      try {
        setLoading(true);
        const query = buildQueryString({ limit });
        const data = await fetchAPI<FeatureActivationExample[]>(
          `/features/${featureId}/examples?${query}`
        );
        setExamples(data);
      } catch (err) {
        console.error('Error fetching feature examples:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchExamples();
  }, [featureId, limit]);

  if (loading) {
    return <div className="text-xs text-slate-400">Loading...</div>;
  }

  if (examples.length === 0) {
    return <div className="text-xs text-slate-400">No examples</div>;
  }

  return (
    <div className="space-y-3">
      {examples.map((example, index) => (
        <div key={index} className="space-y-1">
          <div className="flex items-center justify-between text-xs">
            <span className="text-slate-400">#{index + 1}</span>
            <span className="text-emerald-400 font-mono">
              {example.max_activation.toFixed(2)}
            </span>
          </div>
          <TokenHighlight
            tokens={example.tokens.slice(0, 15)}
            activations={example.activations.slice(0, 15)}
            maxActivation={example.max_activation}
          />
          {example.tokens.length > 15 && (
            <div className="text-xs text-slate-500">
              ... +{example.tokens.length - 15} tokens
            </div>
          )}
        </div>
      ))}
    </div>
  );
};

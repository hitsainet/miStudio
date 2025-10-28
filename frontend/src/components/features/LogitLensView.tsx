/**
 * LogitLensView Component
 *
 * Displays logit lens analysis for a feature.
 * Shows top predicted tokens and semantic interpretation.
 *
 * Analysis: What the feature contributes to model predictions by passing
 * a synthetic activation through the SAE decoder and model head.
 */

import React, { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { Loader2 } from 'lucide-react';

interface LogitLensResponse {
  top_tokens: string[];
  probabilities: number[];
  interpretation: string;
  computed_at: string;
}

interface LogitLensViewProps {
  featureId: string;
}

/**
 * LogitLensView component.
 * Fetches and displays logit lens analysis for a feature.
 */
export const LogitLensView: React.FC<LogitLensViewProps> = ({ featureId }) => {
  const [data, setData] = useState<LogitLensResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchLogitLens = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await api.get<LogitLensResponse>(
          `/features/${featureId}/logit-lens`
        );
        setData(response.data);
      } catch (err: any) {
        console.error('Error fetching logit lens:', err);
        setError(err.response?.data?.detail || 'Failed to load logit lens analysis');
      } finally {
        setLoading(false);
      }
    };

    fetchLogitLens();
  }, [featureId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-3">
        <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
        <div className="text-slate-400 text-sm">Computing logit lens analysis...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-red-400">{error}</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-slate-400">No data available</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Semantic Interpretation */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">Interpretation</h3>
        <p className="text-sm text-slate-400">{data.interpretation}</p>
      </div>

      {/* Top Predicted Tokens */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-4">
          Top Predicted Tokens
        </h3>
        <div className="space-y-3">
          {data.top_tokens.map((token, index) => {
            const probability = data.probabilities[index];
            const percentage = (probability * 100).toFixed(1);
            const width = Math.max(5, probability * 100); // Minimum 5% width for visibility

            return (
              <div key={index} className="space-y-1">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-mono text-slate-300">{token}</span>
                  <span className="text-emerald-400">{percentage}%</span>
                </div>
                {/* Probability Bar */}
                <div className="w-full bg-slate-700/50 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-emerald-500 to-emerald-400 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${width}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Metadata */}
      <div className="text-xs text-slate-500">
        Computed: {new Date(data.computed_at).toLocaleString()}
      </div>
    </div>
  );
};

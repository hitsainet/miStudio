/**
 * AblationAnalysis Component
 *
 * Displays ablation analysis for a feature.
 * Shows feature's impact on model performance.
 *
 * Analysis: Measures feature importance by comparing model performance
 * with the feature active vs. ablated (set to zero).
 */

import React, { useEffect, useState } from 'react';
import { api } from '../../api/client';
import { Loader2, Zap, AlertCircle } from 'lucide-react';

interface AblationResponse {
  perplexity_delta: number;
  impact_score: number;
  baseline_perplexity: number;
  ablated_perplexity: number;
  computed_at: string;
}

interface AblationAnalysisProps {
  featureId: string;
}

/**
 * Get impact level description and color based on impact score.
 */
const getImpactLevel = (score: number): { label: string; color: string; description: string } => {
  if (score >= 0.7) {
    return {
      label: 'High Impact',
      color: 'text-red-400',
      description: 'This feature significantly affects model performance',
    };
  } else if (score >= 0.4) {
    return {
      label: 'Medium Impact',
      color: 'text-yellow-400',
      description: 'This feature moderately affects model performance',
    };
  } else if (score >= 0.2) {
    return {
      label: 'Low Impact',
      color: 'text-blue-400',
      description: 'This feature has a minor effect on model performance',
    };
  } else {
    return {
      label: 'Minimal Impact',
      color: 'text-slate-400',
      description: 'This feature has negligible effect on model performance',
    };
  }
};

/**
 * AblationAnalysis component.
 * Fetches and displays ablation analysis for a feature.
 */
export const AblationAnalysis: React.FC<AblationAnalysisProps> = ({ featureId }) => {
  const [data, setData] = useState<AblationResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAblation = async () => {
      try {
        setLoading(true);
        setError(null);
        const response = await api.get<AblationResponse>(
          `/features/${featureId}/ablation`
        );
        setData(response.data);
      } catch (err: any) {
        console.error('Error fetching ablation:', err);
        setError(err.response?.data?.detail || 'Failed to load ablation analysis');
      } finally {
        setLoading(false);
      }
    };

    fetchAblation();
  }, [featureId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-3">
        <Loader2 className="h-8 w-8 text-purple-500 animate-spin" />
        <div className="text-slate-400 text-sm">Computing ablation analysis...</div>
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

  const impact = getImpactLevel(data.impact_score);
  const impactPercent = (data.impact_score * 100).toFixed(1);

  return (
    <div className="space-y-6">
      {/* Impact Overview */}
      <div className="bg-slate-800/30 rounded-lg p-6 space-y-4">
        {/* Impact Score with Icon */}
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Zap className={`h-8 w-8 ${impact.color}`} />
            <div>
              <div className="text-sm text-slate-400">Impact Score</div>
              <div className={`text-3xl font-bold ${impact.color}`}>
                {impactPercent}%
              </div>
            </div>
          </div>
          <div className="text-right">
            <div className={`text-lg font-medium ${impact.color}`}>
              {impact.label}
            </div>
            <div className="text-xs text-slate-500 mt-1">
              0-100% scale
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-slate-700/50 rounded-full h-3">
          <div
            className="bg-gradient-to-r from-purple-500 to-purple-400 h-3 rounded-full transition-all duration-500"
            style={{ width: `${Math.max(2, data.impact_score * 100)}%` }}
          />
        </div>

        {/* Description */}
        <div className="flex items-start space-x-2 bg-slate-800/50 rounded p-3">
          <AlertCircle className="h-5 w-5 text-slate-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-slate-400">{impact.description}</p>
        </div>
      </div>

      {/* Perplexity Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Baseline Perplexity */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-400 mb-1">Baseline Perplexity</div>
          <div className="text-2xl font-bold text-emerald-400">
            {data.baseline_perplexity.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500 mt-1">Feature active</div>
        </div>

        {/* Ablated Perplexity */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-400 mb-1">Ablated Perplexity</div>
          <div className="text-2xl font-bold text-orange-400">
            {data.ablated_perplexity.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500 mt-1">Feature removed</div>
        </div>

        {/* Perplexity Delta */}
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-400 mb-1">Perplexity Delta</div>
          <div className="text-2xl font-bold text-purple-400">
            +{data.perplexity_delta.toFixed(1)}
          </div>
          <div className="text-xs text-slate-500 mt-1">Increase when ablated</div>
        </div>
      </div>

      {/* Explanation */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <h3 className="text-sm font-medium text-slate-300 mb-2">What This Means</h3>
        <div className="text-sm text-slate-400 space-y-2">
          <p>
            <strong>Baseline Perplexity:</strong> Model performance with this feature active.
            Lower is better.
          </p>
          <p>
            <strong>Ablated Perplexity:</strong> Model performance when this feature is removed (set to zero).
          </p>
          <p>
            <strong>Perplexity Delta:</strong> The increase in perplexity when the feature is removed.
            Higher delta indicates greater feature importance.
          </p>
          <p>
            <strong>Impact Score:</strong> Normalized measure (0-100%) of how much this feature
            contributes to model performance. Based on perplexity change.
          </p>
        </div>
      </div>

      {/* Metadata */}
      <div className="text-xs text-slate-500">
        Computed: {new Date(data.computed_at).toLocaleString()}
      </div>
    </div>
  );
};

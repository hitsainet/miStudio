/**
 * FeatureCorrelations Component
 *
 * Displays feature correlation analysis.
 * Shows features with similar activation patterns.
 *
 * Analysis: Finds features with similar activation patterns by computing
 * Pearson correlation coefficients on activation vectors.
 */

import React, { useEffect, useState } from 'react';
import { fetchAPI } from '../../api/client';
import { Loader2, TrendingUp } from 'lucide-react';

interface CorrelatedFeature {
  feature_id: string;
  feature_name: string;
  correlation: number;
}

interface CorrelationsResponse {
  correlated_features: CorrelatedFeature[];
  computed_at: string;
}

interface FeatureCorrelationsProps {
  featureId: string;
  onFeatureClick?: (featureId: string) => void;
}

/**
 * FeatureCorrelations component.
 * Fetches and displays correlation analysis for a feature.
 */
export const FeatureCorrelations: React.FC<FeatureCorrelationsProps> = ({
  featureId,
  onFeatureClick,
}) => {
  const [data, setData] = useState<CorrelationsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchCorrelations = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchAPI<CorrelationsResponse>(
          `/features/${featureId}/correlations`
        );
        setData(data);
      } catch (err: any) {
        console.error('Error fetching correlations:', err);
        setError(err.detail || err.message || 'Failed to load correlations analysis');
      } finally {
        setLoading(false);
      }
    };

    fetchCorrelations();
  }, [featureId]);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-3">
        <Loader2 className="h-8 w-8 text-blue-500 animate-spin" />
        <div className="text-slate-400 text-sm">Computing correlation analysis...</div>
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

  if (data.correlated_features.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-2">
        <TrendingUp className="h-12 w-12 text-slate-600" />
        <div className="text-slate-400">No correlated features found</div>
        <div className="text-xs text-slate-500">
          Correlation threshold: 0.5
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <p className="text-sm text-slate-400">
          Features with similar activation patterns (correlation ≥ 0.5).
          Higher correlation indicates features that activate on similar inputs.
        </p>
      </div>

      {/* Correlations Table */}
      <div className="bg-slate-800/30 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-slate-800/50">
            <tr>
              <th className="text-left px-4 py-3 text-xs font-medium text-slate-400">
                Feature ID
              </th>
              <th className="text-left px-4 py-3 text-xs font-medium text-slate-400">
                Label
              </th>
              <th className="text-right px-4 py-3 text-xs font-medium text-slate-400">
                Correlation
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-700/50">
            {data.correlated_features.map((feature) => {
              const correlationPercent = (feature.correlation * 100).toFixed(1);
              const barWidth = Math.max(5, feature.correlation * 100);

              return (
                <tr
                  key={feature.feature_id}
                  className="hover:bg-slate-800/30 transition-colors"
                >
                  <td className="px-4 py-3">
                    {onFeatureClick ? (
                      <button
                        onClick={() => onFeatureClick(feature.feature_id)}
                        className="font-mono text-sm text-blue-400 hover:text-blue-300 underline"
                      >
                        {feature.feature_id}
                      </button>
                    ) : (
                      <span className="font-mono text-sm text-slate-300">
                        {feature.feature_id}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <div className="text-sm text-slate-300 truncate max-w-xs">
                      {feature.feature_name}
                    </div>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end space-x-3">
                      {/* Bar visualization */}
                      <div className="flex-1 max-w-[100px] bg-slate-700/50 rounded-full h-2">
                        <div
                          className="bg-gradient-to-r from-blue-500 to-blue-400 h-2 rounded-full transition-all duration-300"
                          style={{ width: `${barWidth}%` }}
                        />
                      </div>
                      {/* Percentage */}
                      <span className="text-sm text-blue-400 font-mono min-w-[3.5rem] text-right">
                        {correlationPercent}%
                      </span>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {/* Metadata */}
      <div className="text-xs text-slate-500">
        Computed: {new Date(data.computed_at).toLocaleString()}
      </div>
    </div>
  );
};

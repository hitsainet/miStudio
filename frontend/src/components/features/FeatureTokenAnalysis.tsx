/**
 * FeatureTokenAnalysis Component
 *
 * Displays token analysis for a feature's activation examples with granular filter controls.
 * Shows which tokens most frequently appear in examples that activate this feature.
 *
 * Analysis: Analyzes all tokens from the feature's max-activating examples,
 * filters out junk tokens based on user-selected categories, and displays ranked statistics.
 */

import React, { useEffect, useState } from 'react';
import { Loader2, Hash, Settings } from 'lucide-react';
import { useFeaturesStore } from '../../stores/featuresStore';

interface FeatureTokenAnalysisProps {
  featureId: string;
}

/**
 * FeatureTokenAnalysis component.
 * Fetches and displays token analysis for a feature with toggleable filters.
 */
export const FeatureTokenAnalysis: React.FC<FeatureTokenAnalysisProps> = ({ featureId }) => {
  const { featureTokenAnalysis, isLoadingTokenAnalysis, fetchFeatureTokenAnalysis } = useFeaturesStore();

  // Filter states
  const [filterSpecial, setFilterSpecial] = useState(true);
  const [filterSingleChar, setFilterSingleChar] = useState(true);
  const [filterPunctuation, setFilterPunctuation] = useState(true);
  const [filterNumbers, setFilterNumbers] = useState(true);
  const [filterFragments, setFilterFragments] = useState(true);
  const [filterStopWords, setFilterStopWords] = useState(true);
  const [showFilters, setShowFilters] = useState(false);

  // Fetch data with current filters
  const fetchData = () => {
    fetchFeatureTokenAnalysis(featureId, {
      applyFilters: true,
      filterSpecial,
      filterSingleChar,
      filterPunctuation,
      filterNumbers,
      filterFragments,
      filterStopWords,
    });
  };

  // Load on mount
  useEffect(() => {
    fetchData();
  }, [featureId]);

  // Refetch when filters change
  useEffect(() => {
    fetchData();
  }, [filterSpecial, filterSingleChar, filterPunctuation, filterNumbers, filterFragments, filterStopWords]);

  if (isLoadingTokenAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-3">
        <Loader2 className="h-8 w-8 text-emerald-500 animate-spin" />
        <div className="text-slate-400 text-sm">Analyzing tokens...</div>
      </div>
    );
  }

  if (!featureTokenAnalysis) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="text-slate-400">No data available</div>
      </div>
    );
  }

  const { summary, tokens } = featureTokenAnalysis;

  if (tokens.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-12 space-y-2">
        <Hash className="h-12 w-12 text-slate-600" />
        <div className="text-slate-400">No meaningful tokens found</div>
        <div className="text-xs text-slate-500">
          All tokens were filtered as junk ({summary.junk_removed} removed)
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Description */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <p className="text-sm text-slate-400">
          Tokens that appear in examples that activate this feature.
          Shows which input tokens most frequently trigger this feature's activation.
        </p>
      </div>

      {/* Filter Controls */}
      <div className="bg-slate-800/30 rounded-lg p-4">
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 text-sm font-medium text-slate-300 hover:text-white transition-colors"
        >
          <Settings className="w-4 h-4" />
          <span>Filter Options</span>
          <span className="text-xs text-slate-500">
            ({summary.junk_removed} tokens filtered)
          </span>
        </button>

        {showFilters && (
          <div className="mt-4 grid grid-cols-2 md:grid-cols-3 gap-3">
            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={filterSpecial}
                onChange={(e) => setFilterSpecial(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>Special tokens</span>
              <span className="text-xs text-slate-500">(&lt;s&gt;, &lt;/s&gt;)</span>
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={filterSingleChar}
                onChange={(e) => setFilterSingleChar(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>Single characters</span>
              <span className="text-xs text-slate-500">(a, b, 1)</span>
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={filterPunctuation}
                onChange={(e) => setFilterPunctuation(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>Punctuation</span>
              <span className="text-xs text-slate-500">(., !, ?)</span>
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={filterNumbers}
                onChange={(e) => setFilterNumbers(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>Numbers</span>
              <span className="text-xs text-slate-500">(123, 2024)</span>
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={filterFragments}
                onChange={(e) => setFilterFragments(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>Word fragments</span>
              <span className="text-xs text-slate-500">(tion, ing, ed)</span>
            </label>

            <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer hover:text-white">
              <input
                type="checkbox"
                checked={filterStopWords}
                onChange={(e) => setFilterStopWords(e.target.checked)}
                className="rounded border-slate-600 bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900"
              />
              <span>Stop words</span>
              <span className="text-xs text-slate-500">(the, and, is)</span>
            </label>
          </div>
        )}
      </div>

      {/* Summary Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-500 mb-1">Examples</div>
          <div className="text-2xl font-semibold text-slate-200">{summary.total_examples}</div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-500 mb-1">Unique Tokens</div>
          <div className="text-2xl font-semibold text-emerald-400">{summary.filtered_token_count}</div>
          <div className="text-xs text-slate-500 mt-1">
            {summary.junk_removed} filtered
          </div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-500 mb-1">Total Occurrences</div>
          <div className="text-2xl font-semibold text-slate-200">{summary.filtered_token_occurrences}</div>
        </div>
        <div className="bg-slate-800/30 rounded-lg p-4">
          <div className="text-xs text-slate-500 mb-1">Diversity</div>
          <div className="text-2xl font-semibold text-blue-400">{summary.diversity_percent.toFixed(1)}%</div>
        </div>
      </div>

      {/* Tokens Table */}
      <div className="bg-slate-800/30 rounded-lg overflow-hidden">
        <div className="overflow-x-auto" style={{ maxHeight: '600px' }}>
          <table className="w-full">
            <thead className="bg-slate-800/50 sticky top-0 z-10">
              <tr>
                <th className="text-right px-4 py-3 text-xs font-medium text-slate-400 w-16">
                  Rank
                </th>
                <th className="text-left px-4 py-3 text-xs font-medium text-slate-400">
                  Token
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-slate-400 w-24">
                  Count
                </th>
                <th className="text-right px-4 py-3 text-xs font-medium text-slate-400 w-32">
                  Percentage
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {tokens.map((tokenData) => {
                const barWidth = Math.max(2, tokenData.percentage);

                return (
                  <tr
                    key={tokenData.rank}
                    className="hover:bg-slate-800/30 transition-colors"
                  >
                    <td className="px-4 py-3 text-right">
                      <span className="text-sm text-slate-500 font-mono">
                        {tokenData.rank}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <span className="font-mono text-sm text-slate-200 bg-slate-900/50 px-2 py-1 rounded">
                        {tokenData.token}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="text-sm text-slate-300 font-mono">
                        {tokenData.count}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end space-x-3">
                        {/* Bar visualization */}
                        <div className="flex-1 max-w-[80px] bg-slate-700/50 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-emerald-500 to-emerald-400 h-2 rounded-full transition-all duration-300"
                            style={{ width: `${barWidth}%` }}
                          />
                        </div>
                        {/* Percentage */}
                        <span className="text-sm text-emerald-400 font-mono min-w-[3.5rem] text-right">
                          {tokenData.percentage.toFixed(2)}%
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

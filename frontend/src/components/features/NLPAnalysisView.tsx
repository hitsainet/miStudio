/**
 * NLPAnalysisView Component
 *
 * Displays pre-computed NLP analysis for a feature including:
 * - Prime token analysis (POS tags, NER, token types)
 * - Context patterns (n-grams, syntactic patterns)
 * - Activation statistics
 * - Semantic clusters
 */

import React, { useState } from 'react';
import { Brain, Tag, Hash, TrendingUp, Layers, ChevronDown, ChevronUp, Sparkles } from 'lucide-react';
import type { NLPAnalysis, NLPSemanticCluster } from '../../types/features';

interface NLPAnalysisViewProps {
  nlpAnalysis: NLPAnalysis | null;
  nlpProcessedAt: string | null;
  featureId: string;
}

export const NLPAnalysisView: React.FC<NLPAnalysisViewProps> = ({
  nlpAnalysis,
  nlpProcessedAt,
  featureId: _featureId,
}) => {
  const [expandedSections, setExpandedSections] = useState<Set<string>>(
    new Set(['summary', 'prime', 'context', 'activation'])
  );

  const toggleSection = (section: string) => {
    setExpandedSections((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(section)) {
        newSet.delete(section);
      } else {
        newSet.add(section);
      }
      return newSet;
    });
  };

  if (!nlpAnalysis) {
    return (
      <div className="flex flex-col items-center justify-center py-12 text-slate-400">
        <Brain className="w-12 h-12 mb-4 text-slate-600" />
        <p className="text-lg font-medium mb-2">No NLP Analysis Available</p>
        <p className="text-sm text-slate-500 text-center max-w-md">
          NLP analysis has not been computed for this feature yet.
          Use the "Process NLP" button on the extraction card to analyze all features.
        </p>
      </div>
    );
  }

  const { prime_token_analysis, context_patterns, activation_stats, semantic_clusters, summary_for_prompt, num_examples_analyzed, computed_at } = nlpAnalysis;

  // Section Header Component
  const SectionHeader: React.FC<{
    id: string;
    icon: React.ReactNode;
    title: string;
    subtitle?: string;
  }> = ({ id, icon, title, subtitle }) => {
    const isExpanded = expandedSections.has(id);
    return (
      <button
        onClick={() => toggleSection(id)}
        className="w-full flex items-center justify-between p-3 bg-slate-800/50 hover:bg-slate-800/70 rounded-lg transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="text-cyan-400">{icon}</div>
          <div className="text-left">
            <div className="font-medium text-slate-200">{title}</div>
            {subtitle && <div className="text-xs text-slate-400">{subtitle}</div>}
          </div>
        </div>
        {isExpanded ? (
          <ChevronUp className="w-5 h-5 text-slate-400" />
        ) : (
          <ChevronDown className="w-5 h-5 text-slate-400" />
        )}
      </button>
    );
  };

  // Format percentage
  const formatPct = (value: number) => `${(value * 100).toFixed(1)}%`;

  // Top items display helper
  const TopItems: React.FC<{ items: Record<string, number>; limit?: number; label?: string }> = ({
    items,
    limit = 10,
    label = 'items',
  }) => {
    const sorted = Object.entries(items)
      .sort(([, a], [, b]) => b - a)
      .slice(0, limit);

    if (sorted.length === 0) {
      return <span className="text-slate-500 italic">No {label} found</span>;
    }

    return (
      <div className="flex flex-wrap gap-2">
        {sorted.map(([item, count]) => (
          <span
            key={item}
            className="px-2 py-1 bg-slate-700/50 text-slate-300 text-xs rounded flex items-center gap-1"
          >
            <span className="font-mono">{item}</span>
            <span className="text-slate-500">({count})</span>
          </span>
        ))}
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Metadata Header */}
      <div className="flex items-center justify-between text-sm text-slate-400 mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-4 h-4 text-cyan-400" />
          <span>Analyzed {num_examples_analyzed} examples</span>
        </div>
        {nlpProcessedAt && (
          <span>
            Computed: {new Date(computed_at || nlpProcessedAt).toLocaleString()}
          </span>
        )}
      </div>

      {/* Summary Section */}
      <div className="space-y-2">
        <SectionHeader
          id="summary"
          icon={<Sparkles className="w-5 h-5" />}
          title="Summary"
          subtitle="AI-generated feature description"
        />
        {expandedSections.has('summary') && (
          <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50">
            <p className="text-slate-300 whitespace-pre-wrap leading-relaxed">
              {summary_for_prompt || 'No summary available.'}
            </p>
          </div>
        )}
      </div>

      {/* Prime Token Analysis */}
      <div className="space-y-2">
        <SectionHeader
          id="prime"
          icon={<Tag className="w-5 h-5" />}
          title="Prime Token Analysis"
          subtitle={`${prime_token_analysis.unique_count} unique tokens, ${formatPct(prime_token_analysis.concentration_ratio)} concentration`}
        />
        {expandedSections.has('prime') && (
          <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50 space-y-4">
            {/* Most Common Token */}
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Most Common Token</div>
              <div className="flex items-center gap-2">
                <span className="px-3 py-1.5 bg-cyan-900/30 text-cyan-400 rounded font-mono text-lg">
                  "{prime_token_analysis.most_common_token[0]}"
                </span>
                <span className="text-slate-400">
                  ({prime_token_analysis.most_common_token[1]} occurrences)
                </span>
              </div>
            </div>

            {/* POS Distribution */}
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Part-of-Speech Distribution</div>
              <TopItems items={prime_token_analysis.pos_distribution} label="POS tags" />
            </div>

            {/* Token Types */}
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Token Types</div>
              <TopItems items={prime_token_analysis.token_types} label="token types" />
            </div>

            {/* Named Entities */}
            {prime_token_analysis.ner_entities.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Named Entities</div>
                <div className="flex flex-wrap gap-2">
                  {prime_token_analysis.ner_entities.slice(0, 10).map((entity, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-purple-900/30 text-purple-400 text-xs rounded"
                    >
                      {entity.text} <span className="text-purple-300/60">({entity.label})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Token Frequency */}
            <div>
              <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Top Prime Tokens (Case-Sensitive)</div>
              <TopItems items={prime_token_analysis.frequency_distribution} limit={15} label="tokens" />
            </div>
          </div>
        )}
      </div>

      {/* Context Patterns */}
      <div className="space-y-2">
        <SectionHeader
          id="context"
          icon={<Hash className="w-5 h-5" />}
          title="Context Patterns"
          subtitle="N-grams and syntactic patterns from surrounding context"
        />
        {expandedSections.has('context') && (
          <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50 space-y-4">
            {/* Immediately Before/After */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Immediately Before</div>
                <TopItems items={context_patterns.immediately_before} limit={8} label="tokens" />
              </div>
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Immediately After</div>
                <TopItems items={context_patterns.immediately_after} limit={8} label="tokens" />
              </div>
            </div>

            {/* Prefix/Suffix Bigrams */}
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Prefix Bigrams</div>
                {context_patterns.prefix_bigrams.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {context_patterns.prefix_bigrams.slice(0, 6).map((ngram, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-emerald-900/30 text-emerald-400 text-xs rounded font-mono"
                      >
                        {ngram.tokens.join(' ')} ({ngram.count})
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-500 italic text-sm">No bigrams found</span>
                )}
              </div>
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Suffix Bigrams</div>
                {context_patterns.suffix_bigrams.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {context_patterns.suffix_bigrams.slice(0, 6).map((ngram, idx) => (
                      <span
                        key={idx}
                        className="px-2 py-1 bg-blue-900/30 text-blue-400 text-xs rounded font-mono"
                      >
                        {ngram.tokens.join(' ')} ({ngram.count})
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-500 italic text-sm">No bigrams found</span>
                )}
              </div>
            </div>

            {/* Syntactic Patterns */}
            {context_patterns.syntactic_patterns.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Syntactic Patterns</div>
                <div className="flex flex-wrap gap-2">
                  {context_patterns.syntactic_patterns.slice(0, 8).map((pattern, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-amber-900/30 text-amber-400 text-xs rounded font-mono"
                    >
                      {pattern.pattern} ({pattern.count})
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Activation Statistics */}
      <div className="space-y-2">
        <SectionHeader
          id="activation"
          icon={<TrendingUp className="w-5 h-5" />}
          title="Activation Statistics"
          subtitle={`Distribution: ${activation_stats.distribution_type}, CV: ${activation_stats.coefficient_of_variation.toFixed(2)}`}
        />
        {expandedSections.has('activation') && (
          <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50 space-y-4">
            {/* Stats Grid */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Mean</div>
                <div className="text-lg font-semibold text-emerald-400">{activation_stats.mean.toFixed(3)}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Std Dev</div>
                <div className="text-lg font-semibold text-blue-400">{activation_stats.std.toFixed(3)}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Median</div>
                <div className="text-lg font-semibold text-purple-400">{activation_stats.median.toFixed(3)}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Min</div>
                <div className="text-lg font-semibold text-slate-300">{activation_stats.min.toFixed(3)}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Max</div>
                <div className="text-lg font-semibold text-slate-300">{activation_stats.max.toFixed(3)}</div>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3">
                <div className="text-xs text-slate-400 mb-1">Skewness</div>
                <div className="text-lg font-semibold text-amber-400">{activation_stats.skewness.toFixed(3)}</div>
              </div>
            </div>

            {/* High Activation Tokens */}
            {activation_stats.high_activation_tokens.length > 0 && (
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">High Activation Tokens</div>
                <div className="flex flex-wrap gap-2">
                  {activation_stats.high_activation_tokens.slice(0, 10).map((item, idx) => (
                    <span
                      key={idx}
                      className="px-2 py-1 bg-red-900/30 text-red-400 text-xs rounded font-mono"
                    >
                      "{item.token}" <span className="text-red-300/60">({item.activation.toFixed(2)})</span>
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Activation Range Buckets */}
            {Object.keys(activation_stats.activation_range_buckets).length > 0 && (
              <div>
                <div className="text-xs text-slate-400 uppercase tracking-wide mb-2">Activation Distribution</div>
                <div className="space-y-1">
                  {Object.entries(activation_stats.activation_range_buckets)
                    .sort(([a], [b]) => a.localeCompare(b))
                    .map(([range, count]) => {
                      const total = Object.values(activation_stats.activation_range_buckets).reduce((a, b) => a + b, 0);
                      const pct = (count / total) * 100;
                      return (
                        <div key={range} className="flex items-center gap-2">
                          <span className="text-xs text-slate-400 w-24 font-mono">{range}</span>
                          <div className="flex-1 h-4 bg-slate-700/30 rounded overflow-hidden">
                            <div
                              className="h-full bg-cyan-600/50"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                          <span className="text-xs text-slate-400 w-16 text-right">
                            {count} ({pct.toFixed(0)}%)
                          </span>
                        </div>
                      );
                    })}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Semantic Clusters */}
      {semantic_clusters && semantic_clusters.length > 0 && (
        <div className="space-y-2">
          <SectionHeader
            id="clusters"
            icon={<Layers className="w-5 h-5" />}
            title="Semantic Clusters"
            subtitle={`${semantic_clusters.length} clusters identified`}
          />
          {expandedSections.has('clusters') && (
            <div className="p-4 bg-slate-800/30 rounded-lg border border-slate-700/50 space-y-3">
              {semantic_clusters.map((cluster: NLPSemanticCluster, idx: number) => (
                <div
                  key={idx}
                  className="p-3 bg-slate-700/30 rounded-lg"
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-slate-200">{cluster.label}</span>
                    <span className="text-xs text-slate-400">
                      {cluster.size} examples, avg activation: {cluster.avg_activation.toFixed(2)}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {cluster.representative_tokens.map((token, tidx) => (
                      <span
                        key={tidx}
                        className="px-2 py-0.5 bg-slate-600/50 text-slate-300 text-xs rounded font-mono"
                      >
                        {token}
                      </span>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

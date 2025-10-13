/**
 * ActivationExtractionHistory - Display completed activation extractions with statistics.
 *
 * Features:
 * - List of completed extractions with metadata
 * - Detailed statistics for each layer
 * - File information and storage sizes
 * - Timestamps and configuration details
 */

import { useState, useEffect } from 'react';
import { FileText, Download, Calendar, Layers, Database, TrendingUp, Activity } from 'lucide-react';
import { Model } from '../../types/model';
import { getModelExtractions } from '../../api/models';

interface ExtractionStatistics {
  shape: number[];
  mean_magnitude: number;
  max_activation: number;
  min_activation: number;
  std_activation: number;
  sparsity_percent: number;
  size_mb: number;
}

interface Extraction {
  extraction_id: string;
  model_id: string;
  architecture: string;
  quantization: string;
  dataset_path: string;
  layer_indices: number[];
  hook_types: string[];
  max_samples: number;
  batch_size: number;
  num_samples_processed: number;
  created_at: string;
  saved_files: string[];
  statistics: Record<string, ExtractionStatistics>;
}

interface ActivationExtractionHistoryProps {
  model: Model;
  onClose: () => void;
}

export function ActivationExtractionHistory({
  model,
  onClose
}: ActivationExtractionHistoryProps) {
  const [extractions, setExtractions] = useState<Extraction[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedExtraction, setSelectedExtraction] = useState<Extraction | null>(null);

  useEffect(() => {
    fetchExtractions();
  }, [model.id]);

  const fetchExtractions = async () => {
    try {
      setLoading(true);
      const data = await getModelExtractions(model.id);
      setExtractions(data.extractions || []);
    } catch (error) {
      console.error('[ActivationExtractionHistory] Failed to fetch:', error);
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString();
  };

  const formatNumber = (num: number, decimals: number = 4) => {
    return num.toFixed(decimals);
  };

  const getTotalSize = (extraction: Extraction) => {
    const totalMB = Object.values(extraction.statistics).reduce(
      (sum, stat) => sum + stat.size_mb, 0
    );
    return totalMB >= 1024
      ? `${(totalMB / 1024).toFixed(2)} GB`
      : `${totalMB.toFixed(2)} MB`;
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-6xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-2xl font-semibold text-emerald-400">Activation Extractions</h2>
            <p className="text-sm text-slate-400 mt-1">
              {model.name} - {extractions.length} extraction{extractions.length !== 1 ? 's' : ''}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-300 transition-colors"
            aria-label="Close"
          >
            <span className="text-2xl">×</span>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500"></div>
              <p className="text-slate-400 mt-4">Loading extractions...</p>
            </div>
          )}

          {!loading && extractions.length === 0 && (
            <div className="text-center py-12">
              <Database className="w-16 h-16 text-slate-600 mx-auto mb-4" />
              <p className="text-slate-400 text-lg">No extractions yet</p>
              <p className="text-slate-500 mt-2">
                Extract activations from this model to see results here
              </p>
            </div>
          )}

          {!loading && extractions.length > 0 && (
            <div className="space-y-4">
              {extractions.map((extraction) => (
                <div
                  key={extraction.extraction_id}
                  className="bg-slate-800/50 border border-slate-700 rounded-lg p-6 hover:border-emerald-500/50 transition-colors cursor-pointer"
                  onClick={() => setSelectedExtraction(extraction === selectedExtraction ? null : extraction)}
                >
                  {/* Extraction Header */}
                  <div className="flex items-start justify-between mb-4">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <Activity className="w-5 h-5 text-emerald-400" />
                        <h3 className="text-lg font-semibold text-slate-100">
                          {extraction.extraction_id}
                        </h3>
                      </div>
                      <div className="flex items-center gap-4 text-sm text-slate-400">
                        <div className="flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          <span>{formatDate(extraction.created_at)}</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Layers className="w-4 h-4" />
                          <span>{extraction.layer_indices.length} layers</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <Database className="w-4 h-4" />
                          <span>{extraction.num_samples_processed.toLocaleString()} samples</span>
                        </div>
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm font-mono text-emerald-400">
                        {getTotalSize(extraction)}
                      </div>
                      <div className="text-xs text-slate-500 mt-1">
                        {extraction.saved_files.length} files
                      </div>
                    </div>
                  </div>

                  {/* Configuration Summary */}
                  <div className="grid grid-cols-4 gap-4 mb-4">
                    <div className="bg-slate-900/50 rounded p-3">
                      <div className="text-xs text-slate-500 mb-1">Layers</div>
                      <div className="text-sm font-mono text-slate-300">
                        {extraction.layer_indices.join(', ')}
                      </div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-3">
                      <div className="text-xs text-slate-500 mb-1">Hook Types</div>
                      <div className="text-sm text-slate-300 capitalize">
                        {extraction.hook_types.join(', ')}
                      </div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-3">
                      <div className="text-xs text-slate-500 mb-1">Batch Size</div>
                      <div className="text-sm font-mono text-slate-300">
                        {extraction.batch_size}
                      </div>
                    </div>
                    <div className="bg-slate-900/50 rounded p-3">
                      <div className="text-xs text-slate-500 mb-1">Quantization</div>
                      <div className="text-sm font-mono text-slate-300">
                        {extraction.quantization}
                      </div>
                    </div>
                  </div>

                  {/* Expanded Statistics */}
                  {selectedExtraction === extraction && (
                    <div className="border-t border-slate-700 pt-4 mt-4 space-y-4">
                      <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-emerald-400" />
                        Layer Statistics
                      </h4>

                      {Object.entries(extraction.statistics).map(([layerName, stats]) => (
                        <div key={layerName} className="bg-slate-900 border border-slate-700 rounded-lg p-4">
                          <div className="flex items-center justify-between mb-3">
                            <h5 className="font-mono text-emerald-400 text-sm">
                              {layerName}
                            </h5>
                            <span className="text-xs text-slate-500 font-mono">
                              Shape: [{stats.shape.join(', ')}]
                            </span>
                          </div>

                          <div className="grid grid-cols-3 gap-4">
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Mean Magnitude</div>
                              <div className="text-sm font-mono text-slate-300">
                                {formatNumber(stats.mean_magnitude)}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Std Deviation</div>
                              <div className="text-sm font-mono text-slate-300">
                                {formatNumber(stats.std_activation)}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Sparsity</div>
                              <div className="text-sm font-mono text-slate-300">
                                {formatNumber(stats.sparsity_percent, 2)}%
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Max Activation</div>
                              <div className="text-sm font-mono text-slate-300">
                                {formatNumber(stats.max_activation)}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-slate-500 mb-1">Min Activation</div>
                              <div className="text-sm font-mono text-slate-300">
                                {stats.min_activation === 0 ? '0.0000' : formatNumber(stats.min_activation, 9)}
                              </div>
                            </div>
                            <div>
                              <div className="text-xs text-slate-500 mb-1">File Size</div>
                              <div className="text-sm font-mono text-slate-300">
                                {stats.size_mb.toFixed(2)} MB
                              </div>
                            </div>
                          </div>

                          {/* Activation Range Visualization */}
                          <div className="mt-4">
                            <div className="text-xs text-slate-500 mb-2">Activation Range</div>
                            <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-blue-500 via-emerald-500 to-yellow-500"
                                style={{ width: '100%' }}
                              ></div>
                            </div>
                            <div className="flex justify-between text-xs text-slate-500 mt-1">
                              <span>{formatNumber(stats.min_activation, 4)}</span>
                              <span>{formatNumber(stats.mean_magnitude)}</span>
                              <span>{formatNumber(stats.max_activation)}</span>
                            </div>
                          </div>
                        </div>
                      ))}

                      {/* Files List */}
                      <div className="border-t border-slate-700 pt-4">
                        <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2 mb-3">
                          <FileText className="w-4 h-4 text-emerald-400" />
                          Saved Files ({extraction.saved_files.length})
                        </h4>
                        <div className="grid grid-cols-2 gap-2">
                          {extraction.saved_files.map((file) => (
                            <div
                              key={file}
                              className="bg-slate-900/50 border border-slate-700 rounded px-3 py-2 font-mono text-xs text-slate-300"
                            >
                              {file}
                            </div>
                          ))}
                        </div>
                      </div>

                      {/* Dataset Info */}
                      <div className="border-t border-slate-700 pt-4">
                        <h4 className="text-sm font-semibold text-slate-300 mb-2">Dataset</h4>
                        <div className="bg-slate-900/50 rounded px-3 py-2 font-mono text-xs text-slate-400 break-all">
                          {extraction.dataset_path}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t border-slate-800 p-4 bg-slate-900/50">
          <div className="flex items-center justify-between text-sm text-slate-400">
            <div>
              Click an extraction to view detailed statistics
            </div>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded transition-colors text-slate-300"
            >
              Close
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * DatasetPreviewModal - Preview dataset information before downloading.
 *
 * Features:
 * - Display dataset splits with row counts and sizes
 * - Show dataset features/columns with types
 * - Display metadata (description, license, citation)
 * - Allow split selection for download
 * - Show total download size
 */

import { useState, useEffect } from 'react';
import { X, Database, FileText, Info, Download } from 'lucide-react';
import { getDatasetInfo, formatBytes, DatasetInfo } from '../../api/huggingface';

interface DatasetPreviewModalProps {
  repoId: string;
  config?: string;
  onClose: () => void;
  onSelectSplit?: (split: string) => void;
  onDownloadAll?: () => void;
}

export function DatasetPreviewModal({
  repoId,
  config,
  onClose,
  onSelectSplit,
  onDownloadAll,
}: DatasetPreviewModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [selectedSplit, setSelectedSplit] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasetInfo();
  }, [repoId, config]);

  const fetchDatasetInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      const info = await getDatasetInfo(repoId, config);
      console.log('[DatasetPreviewModal] Received dataset info:', info);
      setDatasetInfo(info);
    } catch (err) {
      console.error('[DatasetPreviewModal] Failed to fetch dataset info:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch dataset information');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadSplit = () => {
    if (selectedSplit && onSelectSplit) {
      onSelectSplit(selectedSplit);
      onClose();
    }
  };

  const handleDownloadAll = () => {
    if (onDownloadAll) {
      onDownloadAll();
      onClose();
    }
  };

  const splitEntries = datasetInfo?.splits ? Object.entries(datasetInfo.splits) : [];
  const totalSize = datasetInfo?.dataset_size || 0;
  const downloadSize = datasetInfo?.download_size || 0;

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <Database className="w-6 h-6 text-emerald-400" />
            <div>
              <h2 className="text-2xl font-semibold text-emerald-400">Dataset Preview</h2>
              <p className="text-sm text-slate-400 mt-1 font-mono">{repoId}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-slate-400 hover:text-slate-300 transition-colors"
            aria-label="Close"
          >
            <X className="w-6 h-6" />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {loading && (
            <div className="text-center py-12">
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-emerald-500"></div>
              <p className="text-slate-400 mt-4">Loading dataset information...</p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Failed to load dataset information</p>
                  <p className="text-sm mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {!loading && !error && datasetInfo && (
            <>
              {/* Description */}
              {datasetInfo.description && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <FileText className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-lg font-semibold text-slate-100">Description</h3>
                  </div>
                  <p className="text-slate-300 text-sm leading-relaxed">
                    {datasetInfo.description}
                  </p>
                </div>
              )}

              {/* Metadata Row */}
              <div className="grid grid-cols-2 gap-4">
                {datasetInfo.license && (
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-xs text-slate-500 mb-1">License</div>
                    <div className="text-sm text-slate-300 font-medium">{datasetInfo.license}</div>
                  </div>
                )}
                {datasetInfo.homepage && (
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-xs text-slate-500 mb-1">Homepage</div>
                    <a
                      href={datasetInfo.homepage}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-sm text-emerald-400 hover:text-emerald-300 underline"
                    >
                      View
                    </a>
                  </div>
                )}
              </div>

              {/* Splits Table */}
              <div>
                <div className="flex items-center gap-2 mb-3">
                  <Database className="w-5 h-5 text-emerald-400" />
                  <h3 className="text-lg font-semibold text-slate-100">Available Splits</h3>
                </div>
                <div className="bg-slate-800/30 border border-slate-700 rounded-lg overflow-hidden">
                  <table className="w-full">
                    <thead className="bg-slate-800">
                      <tr>
                        <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">
                          Select
                        </th>
                        <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">
                          Split
                        </th>
                        <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                          Examples
                        </th>
                        <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                          Size
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {splitEntries.map(([splitName, splitData]) => (
                        <tr
                          key={splitName}
                          className="border-t border-slate-700 hover:bg-slate-800/50 transition-colors"
                        >
                          <td className="px-4 py-3">
                            <input
                              type="radio"
                              name="split"
                              value={splitName}
                              checked={selectedSplit === splitName}
                              onChange={(e) => setSelectedSplit(e.target.value)}
                              className="w-4 h-4 text-emerald-600 bg-slate-700 border-slate-600 focus:ring-emerald-500 focus:ring-2"
                            />
                          </td>
                          <td className="px-4 py-3">
                            <span className="text-slate-100 font-mono text-sm">{splitName}</span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="text-slate-300 text-sm">
                              {splitData.num_examples.toLocaleString()}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-right">
                            <span className="text-slate-300 text-sm font-mono">
                              {formatBytes(splitData.num_bytes)}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                    <tfoot className="bg-slate-800/70 border-t-2 border-slate-600">
                      <tr>
                        <td colSpan={2} className="px-4 py-3 text-sm font-medium text-slate-300">
                          Total
                        </td>
                        <td className="px-4 py-3 text-right text-sm font-medium text-emerald-400">
                          {splitEntries
                            .reduce((sum, [, data]) => sum + data.num_examples, 0)
                            .toLocaleString()}
                        </td>
                        <td className="px-4 py-3 text-right text-sm font-medium text-emerald-400 font-mono">
                          {formatBytes(totalSize)}
                        </td>
                      </tr>
                    </tfoot>
                  </table>
                </div>
              </div>

              {/* Features */}
              {datasetInfo.features && Object.keys(datasetInfo.features).length > 0 && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <FileText className="w-5 h-5 text-emerald-400" />
                    <h3 className="text-lg font-semibold text-slate-100">Features</h3>
                  </div>
                  <div className="bg-slate-800/30 border border-slate-700 rounded-lg p-4">
                    <div className="grid grid-cols-2 gap-3">
                      {Object.entries(datasetInfo.features).map(([name, type]) => (
                        <div key={name} className="flex items-center justify-between">
                          <span className="text-sm text-slate-300 font-mono">{name}</span>
                          <span className="text-xs text-slate-500">
                            {typeof type === 'object' ? JSON.stringify(type).slice(0, 30) : String(type)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              {/* Download Size Info */}
              <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg p-4">
                <div className="flex items-start gap-3">
                  <Download className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-emerald-400 font-medium">Download Size</p>
                    <p className="text-xs text-emerald-300/70 mt-1">
                      Total download: <span className="font-mono font-medium">{formatBytes(downloadSize)}</span>
                      {' • '}
                      Dataset size: <span className="font-mono font-medium">{formatBytes(totalSize)}</span>
                    </p>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        {/* Footer */}
        {!loading && !error && datasetInfo && (
          <div className="border-t border-slate-800 p-4 bg-slate-900/50">
            <div className="flex items-center justify-between">
              <button
                onClick={onClose}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
              >
                Cancel
              </button>
              <div className="flex gap-3">
                {onSelectSplit && (
                  <button
                    onClick={handleDownloadSplit}
                    disabled={!selectedSplit}
                    className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium"
                  >
                    Download Selected Split
                  </button>
                )}
                {onDownloadAll && (
                  <button
                    onClick={handleDownloadAll}
                    className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 rounded-lg transition-colors text-white font-medium flex items-center gap-2"
                  >
                    <Download className="w-4 h-4" />
                    Download All Splits
                  </button>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

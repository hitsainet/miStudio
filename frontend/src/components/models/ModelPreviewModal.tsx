/**
 * ModelPreviewModal - Preview model information before downloading.
 *
 * Features:
 * - Display model metadata (description, tags, license)
 * - Show model architecture and configuration
 * - Display download stats (downloads, likes)
 * - Show memory requirements for different quantizations
 * - Allow quantization selection before download
 */

import { useState, useEffect } from 'react';
import { X, Cpu, Download, Info, Heart, TrendingUp } from 'lucide-react';
import { getModelInfo, calculateMemoryRequirement, formatBytes, ModelInfo } from '../../api/huggingface';
import { QuantizationFormat } from '../../types/model';

interface ModelPreviewModalProps {
  repoId: string;
  onClose: () => void;
  onDownload?: (quantization: string, trustRemoteCode: boolean) => void;
}

export function ModelPreviewModal({
  repoId,
  onClose,
  onDownload,
}: ModelPreviewModalProps) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [selectedQuantization, setSelectedQuantization] = useState<string>(QuantizationFormat.Q4);
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);

  useEffect(() => {
    fetchModelInfo();
  }, [repoId]);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      setError(null);
      const info = await getModelInfo(repoId);
      console.log('[ModelPreviewModal] Received model info:', info);
      setModelInfo(info);
    } catch (err) {
      console.error('[ModelPreviewModal] Failed to fetch model info:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch model information');
    } finally {
      setLoading(false);
    }
  };

  const handleDownload = () => {
    if (onDownload) {
      onDownload(selectedQuantization, trustRemoteCode);
      onClose();
    }
  };

  // Extract parameter count from model ID or config (rough estimation)
  const estimateParamCount = (): number | null => {
    if (!modelInfo) return null;

    // Try to extract from model ID (e.g., "TinyLlama-1.1B", "Llama-2-7b")
    const match = modelInfo.id.match(/(\d+\.?\d*)[BM]/i);
    if (match) {
      const value = parseFloat(match[1]);
      const unit = match[0].slice(-1).toUpperCase();
      return unit === 'B' ? value * 1e9 : value * 1e6;
    }

    // Fallback: assume 1B parameters if we can't determine
    return 1e9;
  };

  const paramCount = estimateParamCount();

  // Calculate memory requirements for different quantizations
  const quantizations = [
    { format: QuantizationFormat.FP32, label: 'FP32 (Full Precision)' },
    { format: QuantizationFormat.FP16, label: 'FP16 (Half Precision)' },
    { format: QuantizationFormat.Q8, label: 'Q8 (8-bit)' },
    { format: QuantizationFormat.Q4, label: 'Q4 (4-bit) - Recommended' },
    { format: QuantizationFormat.Q2, label: 'Q2 (2-bit)' },
  ];

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <Cpu className="w-6 h-6 text-purple-400" />
            <div>
              <h2 className="text-2xl font-semibold text-purple-400">Model Preview</h2>
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
              <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-slate-700 border-t-purple-500"></div>
              <p className="text-slate-400 mt-4">Loading model information...</p>
            </div>
          )}

          {error && (
            <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400">
              <div className="flex items-start gap-2">
                <Info className="w-5 h-5 flex-shrink-0 mt-0.5" />
                <div>
                  <p className="font-medium">Failed to load model information</p>
                  <p className="text-sm mt-1">{error}</p>
                </div>
              </div>
            </div>
          )}

          {!loading && !error && modelInfo && (
            <>
              {/* Unsupported Architecture Error */}
              {modelInfo.unsupportedArchitecture && (
                <div className="p-4 bg-red-500/10 border border-red-500/30 rounded-lg">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-red-300">Unsupported Architecture</p>
                      <p className="text-sm text-red-200/80 mt-1">
                        This model uses the <span className="font-mono">{modelInfo.unsupportedArchitecture}</span> architecture,
                        which is not currently supported. Supported architectures are: falcon, gpt2, gpt_neox, llama,
                        mistral, mixtral, phi, phi3, phi3_v, pythia, qwen, qwen2, qwen3.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Trust Remote Code Warning */}
              {modelInfo.requiresTrustRemoteCode && !modelInfo.unsupportedArchitecture && (
                <div className="p-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                  <div className="flex items-start gap-3">
                    <Info className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-yellow-300">Trust Remote Code Required</p>
                      <p className="text-sm text-yellow-200/80 mt-1">
                        This model requires executing custom code from the repository. You'll need to
                        enable "Trust Remote Code" below. Only proceed if you trust the model source.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Stats Row */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-slate-500 mb-1">
                    <TrendingUp className="w-4 h-4" />
                    <div className="text-xs">Downloads</div>
                  </div>
                  <div className="text-xl font-semibold text-slate-100">
                    {modelInfo.downloads.toLocaleString()}
                  </div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="flex items-center gap-2 text-slate-500 mb-1">
                    <Heart className="w-4 h-4" />
                    <div className="text-xs">Likes</div>
                  </div>
                  <div className="text-xl font-semibold text-slate-100">
                    {modelInfo.likes.toLocaleString()}
                  </div>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-4">
                  <div className="text-xs text-slate-500 mb-1">Pipeline</div>
                  <div className="text-sm font-medium text-slate-300">
                    {modelInfo.pipeline_tag || 'N/A'}
                  </div>
                </div>
              </div>

              {/* Model Tags */}
              {modelInfo.tags && modelInfo.tags.length > 0 && (
                <div>
                  <div className="text-sm font-medium text-slate-300 mb-2">Tags</div>
                  <div className="flex flex-wrap gap-2">
                    {modelInfo.tags.slice(0, 10).map((tag) => (
                      <span
                        key={tag}
                        className="px-2 py-1 bg-purple-500/10 border border-purple-500/30 rounded text-xs text-purple-300"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Architecture Info */}
              {modelInfo.config?.architectures && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="w-5 h-5 text-purple-400" />
                    <h3 className="text-lg font-semibold text-slate-100">Architecture</h3>
                  </div>
                  <div className="bg-slate-800/30 border border-slate-700 rounded-lg p-4">
                    <div className="space-y-2">
                      <div>
                        <div className="text-xs text-slate-500">Model Type</div>
                        <div className="text-sm text-slate-300 font-mono">
                          {modelInfo.config.model_type || 'Unknown'}
                        </div>
                      </div>
                      <div>
                        <div className="text-xs text-slate-500">Architectures</div>
                        <div className="text-sm text-slate-300 font-mono">
                          {modelInfo.config.architectures.join(', ')}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* License & Language */}
              <div className="grid grid-cols-2 gap-4">
                {modelInfo.cardData?.license && (
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-xs text-slate-500 mb-1">License</div>
                    <div className="text-sm text-slate-300 font-medium">{modelInfo.cardData.license}</div>
                  </div>
                )}
                {modelInfo.cardData?.language && (
                  <div className="bg-slate-800/50 rounded-lg p-4">
                    <div className="text-xs text-slate-500 mb-1">Language</div>
                    <div className="text-sm text-slate-300 font-medium">
                      {Array.isArray(modelInfo.cardData.language)
                        ? modelInfo.cardData.language.join(', ')
                        : modelInfo.cardData.language}
                    </div>
                  </div>
                )}
              </div>

              {/* Memory Requirements */}
              {paramCount && (
                <div>
                  <div className="flex items-center gap-2 mb-3">
                    <Download className="w-5 h-5 text-purple-400" />
                    <h3 className="text-lg font-semibold text-slate-100">Memory Requirements</h3>
                  </div>
                  <div className="bg-slate-800/30 border border-slate-700 rounded-lg overflow-hidden">
                    <table className="w-full">
                      <thead className="bg-slate-800">
                        <tr>
                          <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">
                            Select
                          </th>
                          <th className="text-left px-4 py-3 text-sm font-medium text-slate-300">
                            Quantization
                          </th>
                          <th className="text-right px-4 py-3 text-sm font-medium text-slate-300">
                            Est. Memory
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {quantizations.map(({ format, label }) => {
                          const memory = calculateMemoryRequirement(paramCount, format as any);
                          return (
                            <tr
                              key={format}
                              className="border-t border-slate-700 hover:bg-slate-800/50 transition-colors"
                            >
                              <td className="px-4 py-3">
                                <input
                                  type="radio"
                                  name="quantization"
                                  value={format}
                                  checked={selectedQuantization === format}
                                  onChange={(e) => setSelectedQuantization(e.target.value)}
                                  className="w-4 h-4 text-purple-600 bg-slate-700 border-slate-600 focus:ring-purple-500 focus:ring-2"
                                />
                              </td>
                              <td className="px-4 py-3">
                                <span className="text-slate-100 text-sm">{label}</span>
                              </td>
                              <td className="px-4 py-3 text-right">
                                <span className="text-slate-300 text-sm font-mono">
                                  {formatBytes(memory)}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-xs text-slate-500 mt-2">
                    * Memory estimates include 20% overhead for inference. Actual requirements may vary.
                  </p>
                </div>
              )}
            </>
          )}
        </div>

        {/* Footer */}
        {!loading && !error && modelInfo && (
          <div className="border-t border-slate-800 p-4 bg-slate-900/50 space-y-4">
            {/* Trust Remote Code Checkbox */}
            {modelInfo.requiresTrustRemoteCode && !modelInfo.unsupportedArchitecture && (
              <div className="flex items-start gap-3 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
                <input
                  type="checkbox"
                  id="preview-trust-remote-code"
                  checked={trustRemoteCode}
                  onChange={(e) => setTrustRemoteCode(e.target.checked)}
                  className="mt-1 w-4 h-4 rounded border-yellow-500/50 bg-slate-900 text-yellow-500 focus:ring-yellow-500 focus:ring-offset-slate-950"
                />
                <div className="flex-1">
                  <label htmlFor="preview-trust-remote-code" className="block text-sm font-medium text-yellow-300 cursor-pointer">
                    Trust Remote Code
                  </label>
                  <p className="mt-1 text-xs text-yellow-200/80">
                    I understand this model requires executing custom code and I trust the source.
                  </p>
                </div>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex items-center justify-between">
              <button
                onClick={onClose}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors text-slate-300"
              >
                Cancel
              </button>
              {onDownload && (
                <button
                  onClick={handleDownload}
                  disabled={!!modelInfo.unsupportedArchitecture || (modelInfo.requiresTrustRemoteCode && !trustRemoteCode)}
                  className="px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg transition-colors text-white font-medium flex items-center gap-2"
                >
                  <Download className="w-4 h-4" />
                  {modelInfo.unsupportedArchitecture ? 'Unsupported Architecture' : `Download with ${selectedQuantization}`}
                </button>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

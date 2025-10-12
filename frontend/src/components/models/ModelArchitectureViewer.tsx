/**
 * ModelArchitectureViewer - Modal for viewing model architecture details.
 *
 * Features:
 * - Model overview stats (layers, hidden dim, attention heads, parameters)
 * - Detailed layer list with types and dimensions
 * - Model configuration display
 * - Scrollable view for large architectures
 */

import { X } from 'lucide-react';
import { Model } from '../../types/model';
import { useEffect, useState } from 'react';

interface ModelArchitectureViewerProps {
  model: Model;
  onClose: () => void;
}

interface LayerInfo {
  type: string;
  size?: string;
  attention?: string;
  mlp?: string;
}

export function ModelArchitectureViewer({ model, onClose }: ModelArchitectureViewerProps) {
  const [architectureData, setArchitectureData] = useState<{
    layers: LayerInfo[];
    numLayers: number;
    hiddenDim: number;
    numHeads: number;
    mlpDim: number;
    vocabSize: number;
  } | null>(null);

  useEffect(() => {
    // Generate architecture data from model.architecture_config
    const config = model.architecture_config;
    const numLayers = config?.num_layers || config?.num_hidden_layers || 12;
    const hiddenDim = config?.hidden_size || 768;
    const numHeads = config?.num_attention_heads || 12;
    const headDim = Math.floor(hiddenDim / numHeads);
    const mlpDim = config?.intermediate_size || hiddenDim * 4;
    const vocabSize = config?.vocab_size || 50257;

    const layers: LayerInfo[] = [
      { type: 'Embedding', size: `${vocabSize} × ${hiddenDim}` },
      ...Array.from({ length: numLayers }, (_, i) => ({
        type: `TransformerBlock_${i}`,
        attention: `${numHeads} heads × ${headDim} dims`,
        mlp: `${hiddenDim} → ${mlpDim} → ${hiddenDim}`
      })),
      { type: 'LayerNorm', size: `${hiddenDim}` },
      { type: 'Output', size: `${hiddenDim} × ${vocabSize}` }
    ];

    setArchitectureData({
      layers,
      numLayers,
      hiddenDim,
      numHeads,
      mlpDim,
      vocabSize
    });
  }, [model]);

  const formatParams = (count: number): string => {
    if (count >= 1_000_000_000) {
      return `${(count / 1_000_000_000).toFixed(1)}B`;
    } else if (count >= 1_000_000) {
      return `${Math.round(count / 1_000_000)}M`;
    }
    return count.toString();
  };

  if (!architectureData) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-lg max-w-4xl w-full max-h-[80vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-slate-800">
          <div>
            <h2 className="text-2xl font-semibold text-emerald-400">{model.name} Architecture</h2>
            <p className="text-sm text-slate-400 mt-1">
              {formatParams(model.params_count)} parameters • {model.quantization} quantization
              {model.architecture && ` • ${model.architecture}`}
            </p>
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
        <div className="flex-1 overflow-y-auto p-6">
          {/* Model Overview Stats */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Total Layers</div>
              <div className="text-2xl font-semibold text-emerald-400">
                {architectureData.layers.length}
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Hidden Dimension</div>
              <div className="text-2xl font-semibold text-purple-400">
                {architectureData.hiddenDim}
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Attention Heads</div>
              <div className="text-2xl font-semibold text-blue-400">
                {architectureData.numHeads}
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <div className="text-sm text-slate-400 mb-1">Parameters</div>
              <div className="text-2xl font-semibold text-yellow-400">
                {formatParams(model.params_count)}
              </div>
            </div>
          </div>

          {/* Layer List */}
          <div className="space-y-2">
            <h3 className="text-lg font-semibold mb-3 text-slate-100">Model Layers</h3>
            {architectureData.layers.map((layer, idx) => (
              <div
                key={idx}
                className="bg-slate-800/30 border border-slate-700 rounded-lg p-4 hover:border-slate-600 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <span className="text-slate-500 font-mono text-sm w-8">{idx}</span>
                    <div>
                      <div className="font-medium text-slate-200">{layer.type}</div>
                      {layer.attention && layer.mlp && (
                        <div className="text-sm text-slate-400 mt-1">
                          Attention: {layer.attention} | MLP: {layer.mlp}
                        </div>
                      )}
                      {layer.size && (
                        <div className="text-sm text-slate-400 mt-1">
                          Shape: {layer.size}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Model Configuration */}
          <div className="mt-6 bg-slate-800/30 border border-slate-700 rounded-lg p-4">
            <h3 className="text-lg font-semibold mb-3 text-slate-100">Model Configuration</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-slate-400">Vocabulary Size:</span>
                <span className="font-mono ml-2 text-slate-200">{architectureData.vocabSize}</span>
              </div>
              <div>
                <span className="text-slate-400">Max Position:</span>
                <span className="font-mono ml-2 text-slate-200">
                  {model.architecture_config?.max_position_embeddings || 1024}
                </span>
              </div>
              <div>
                <span className="text-slate-400">MLP Ratio:</span>
                <span className="font-mono ml-2 text-slate-200">
                  {Math.round(architectureData.mlpDim / architectureData.hiddenDim)}x
                </span>
              </div>
              <div>
                <span className="text-slate-400">Architecture:</span>
                <span className="font-mono ml-2 text-slate-200">
                  {model.architecture || 'Decoder-only'}
                </span>
              </div>
              {model.architecture_config?.num_key_value_heads && (
                <div>
                  <span className="text-slate-400">KV Heads:</span>
                  <span className="font-mono ml-2 text-slate-200">
                    {model.architecture_config.num_key_value_heads}
                  </span>
                </div>
              )}
              {model.architecture_config?.rope_theta && (
                <div>
                  <span className="text-slate-400">RoPE Theta:</span>
                  <span className="font-mono ml-2 text-slate-200">
                    {model.architecture_config.rope_theta}
                  </span>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

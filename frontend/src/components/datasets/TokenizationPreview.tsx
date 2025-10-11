/**
 * TokenizationPreview - Live tokenization preview component.
 *
 * Allows users to test tokenization settings on sample text before
 * processing the entire dataset. Displays tokens with color-coded special tokens.
 */

import { useState } from 'react';
import { Zap, AlertCircle } from 'lucide-react';
import { API_BASE_URL } from '../../config/api';

interface TokenInfo {
  id: number;
  text: string;
  type: 'special' | 'regular';
  position: number;
}

interface TokenizePreviewResponse {
  tokens: TokenInfo[];
  attention_mask?: number[];
  token_count: number;
  sequence_length: number;
  special_token_count: number;
}

interface TokenizationPreviewProps {
  tokenizerName: string;
  maxLength: number;
  paddingStrategy: 'max_length' | 'longest' | 'do_not_pad';
  truncationStrategy: 'longest_first' | 'only_first' | 'only_second' | 'do_not_truncate';
  disabled?: boolean;
}

export function TokenizationPreview({
  tokenizerName,
  maxLength,
  paddingStrategy,
  truncationStrategy,
  disabled = false,
}: TokenizationPreviewProps) {
  const [text, setText] = useState('');
  const [previewResult, setPreviewResult] = useState<TokenizePreviewResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handlePreview = async () => {
    if (!text.trim()) {
      setError('Please enter some text to preview');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/datasets/tokenize-preview`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          tokenizer_name: tokenizerName,
          text: text,
          max_length: maxLength,
          padding: paddingStrategy,
          truncation: truncationStrategy,
          add_special_tokens: true,
          return_attention_mask: true,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to tokenize');
      }

      const data: TokenizePreviewResponse = await response.json();
      setPreviewResult(data);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to preview tokenization';
      setError(errorMessage);
      setPreviewResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const charCount = text.length;
  const isOverLimit = charCount > 1000;

  return (
    <div className="bg-slate-800/30 border border-slate-700 rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-semibold text-slate-300 flex items-center gap-2">
          <Zap className="w-4 h-4 text-emerald-400" />
          Tokenization Preview
        </h4>
      </div>

      {/* Input Section */}
      <div className="space-y-4">
        <div>
          <div className="flex items-center justify-between mb-2">
            <label htmlFor="preview-text" className="block text-sm font-medium text-slate-300">
              Sample Text
            </label>
            <span
              className={`text-xs font-mono ${
                isOverLimit ? 'text-red-400' : 'text-slate-500'
              }`}
            >
              {charCount} / 1000
            </span>
          </div>
          <textarea
            id="preview-text"
            value={text}
            onChange={(e) => setText(e.target.value.slice(0, 1000))}
            placeholder="Enter sample text to see how it will be tokenized..."
            disabled={disabled || isLoading}
            rows={4}
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded text-slate-100 placeholder-slate-500 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed resize-none"
          />
          <p className="text-xs text-slate-500 mt-1">
            Enter up to 1000 characters to preview tokenization with current settings
          </p>
        </div>

        {/* Preview Button */}
        <button
          onClick={handlePreview}
          disabled={disabled || isLoading || !text.trim() || isOverLimit}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:bg-slate-800 disabled:cursor-not-allowed text-slate-200 text-sm font-medium rounded transition-colors"
        >
          <Zap className="w-4 h-4" />
          {isLoading ? 'Tokenizing...' : 'Preview Tokenization'}
        </button>

        {/* Error Message */}
        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded flex items-start gap-2">
            <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Preview Results */}
        {previewResult && (
          <div className="space-y-4 pt-4 border-t border-slate-700">
            {/* Statistics */}
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-slate-800/50 rounded p-3">
                <div className="text-xs text-slate-400">Total Tokens</div>
                <div className="text-lg font-bold text-slate-200 font-mono">
                  {previewResult.token_count}
                </div>
              </div>
              <div className="bg-slate-800/50 rounded p-3">
                <div className="text-xs text-slate-400">Special Tokens</div>
                <div className="text-lg font-bold text-emerald-400 font-mono">
                  {previewResult.special_token_count}
                </div>
              </div>
              <div className="bg-slate-800/50 rounded p-3">
                <div className="text-xs text-slate-400">Sequence Length</div>
                <div className="text-lg font-bold text-slate-200 font-mono">
                  {previewResult.sequence_length}
                </div>
              </div>
            </div>

            {/* Token Display */}
            <div>
              <div className="text-xs font-medium text-slate-400 mb-2">Tokens:</div>
              <div className="bg-slate-900/50 border border-slate-700 rounded p-4 max-h-64 overflow-y-auto">
                <div className="flex flex-wrap gap-2">
                  {previewResult.tokens.map((token) => (
                    <div
                      key={token.position}
                      className={`inline-flex items-center px-2 py-1 rounded text-xs font-mono ${
                        token.type === 'special'
                          ? 'bg-emerald-500/20 border border-emerald-500/30 text-emerald-300'
                          : 'bg-slate-700/50 border border-slate-600 text-slate-300'
                      }`}
                      title={`Position: ${token.position}, ID: ${token.id}, Type: ${token.type}`}
                    >
                      {token.text}
                    </div>
                  ))}
                </div>
              </div>
              <p className="text-xs text-slate-500 mt-2">
                <span className="inline-block w-3 h-3 bg-emerald-500/20 border border-emerald-500/30 rounded mr-1"></span>
                Special tokens (BOS, EOS, PAD, etc.)
                <span className="inline-block w-3 h-3 bg-slate-700/50 border border-slate-600 rounded mx-1 ml-3"></span>
                Regular tokens
              </p>
            </div>

            {/* Attention Mask (if available) */}
            {previewResult.attention_mask && (
              <div>
                <div className="text-xs font-medium text-slate-400 mb-2">Attention Mask:</div>
                <div className="bg-slate-900/50 border border-slate-700 rounded p-3 max-h-32 overflow-y-auto">
                  <div className="flex flex-wrap gap-1">
                    {previewResult.attention_mask.map((mask, idx) => (
                      <span
                        key={idx}
                        className={`inline-block w-6 h-6 text-center text-xs font-mono leading-6 rounded ${
                          mask === 1
                            ? 'bg-emerald-500/20 text-emerald-300'
                            : 'bg-slate-700/50 text-slate-500'
                        }`}
                        title={`Position ${idx}: ${mask === 1 ? 'Attend' : 'Ignore'}`}
                      >
                        {mask}
                      </span>
                    ))}
                  </div>
                </div>
                <p className="text-xs text-slate-500 mt-2">
                  1 = attend to token, 0 = ignore (padding)
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

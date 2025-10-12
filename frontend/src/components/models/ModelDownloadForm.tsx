/**
 * ModelDownloadForm - Form for downloading models from HuggingFace.
 *
 * Features:
 * - HuggingFace repository ID input with validation
 * - Quantization format selector (FP32, FP16, Q8, Q4, Q2)
 * - Optional access token for gated models
 * - Form validation and error handling
 */

import { useState } from 'react';
import { Download } from 'lucide-react';
import { QuantizationFormat } from '../../types/model';

interface ModelDownloadFormProps {
  onDownload: (repoId: string, quantization: string, accessToken?: string, trustRemoteCode?: boolean) => Promise<void>;
}

export function ModelDownloadForm({ onDownload }: ModelDownloadFormProps) {
  const [hfModelRepo, setHfModelRepo] = useState('');
  const [quantization, setQuantization] = useState<string>(QuantizationFormat.Q4);
  const [accessToken, setAccessToken] = useState('');
  const [trustRemoteCode, setTrustRemoteCode] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);

  const validateRepoId = (repoId: string): boolean => {
    if (!repoId.trim()) {
      setValidationError('Repository ID is required');
      return false;
    }

    // HuggingFace repo format: username/repo-name or org/repo-name
    const repoIdPattern = /^[\w-]+\/[\w.-]+$/;
    if (!repoIdPattern.test(repoId)) {
      setValidationError('Invalid repository format. Use: username/repo-name');
      return false;
    }

    setValidationError(null);
    return true;
  };

  const handleSubmit = async () => {
    if (!validateRepoId(hfModelRepo)) {
      return;
    }

    setIsSubmitting(true);
    try {
      await onDownload(
        hfModelRepo.trim(),
        quantization,
        accessToken.trim() || undefined,
        trustRemoteCode
      );
      // Keep form values after successful download for convenience
      setValidationError(null);
    } catch (error) {
      console.error('[ModelDownloadForm] Download failed:', error);
      setValidationError(error instanceof Error ? error.message : 'Download failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !isSubmitting) {
      handleSubmit();
    }
  };

  return (
    <div className="bg-slate-900/50 border border-slate-800 rounded-lg p-6 space-y-4">
      <div className="grid grid-cols-2 gap-4">
        {/* HuggingFace Repository Input */}
        <div>
          <label className="block text-sm font-medium text-slate-300 mb-2">
            HuggingFace Model Repository
          </label>
          <input
            type="text"
            placeholder="e.g., TinyLlama/TinyLlama-1.1B"
            value={hfModelRepo}
            onChange={(e) => {
              setHfModelRepo(e.target.value);
              setValidationError(null);
            }}
            onKeyPress={handleKeyPress}
            disabled={isSubmitting}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          />
        </div>

        {/* Quantization Format Selector */}
        <div>
          <label htmlFor="model-quantization" className="block text-sm font-medium text-slate-300 mb-2">
            Quantization Format
          </label>
          <select
            id="model-quantization"
            value={quantization}
            onChange={(e) => setQuantization(e.target.value)}
            disabled={isSubmitting}
            className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <option value={QuantizationFormat.FP32}>FP32 (Full Precision)</option>
            <option value={QuantizationFormat.FP16}>FP16 (Half Precision)</option>
            <option value={QuantizationFormat.Q8}>Q8 (8-bit Quantization)</option>
            <option value={QuantizationFormat.Q4}>Q4 (4-bit Quantization) - Recommended</option>
            <option value={QuantizationFormat.Q2}>Q2 (2-bit Quantization)</option>
          </select>
        </div>
      </div>

      {/* Access Token Input */}
      <div>
        <label className="block text-sm font-medium text-slate-300 mb-2">
          Access Token <span className="text-slate-500">(optional, for gated models)</span>
        </label>
        <input
          type="password"
          placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
          value={accessToken}
          onChange={(e) => setAccessToken(e.target.value)}
          onKeyPress={handleKeyPress}
          disabled={isSubmitting}
          className="w-full px-4 py-2 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm text-slate-100 placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        />
        <p className="mt-1 text-xs text-slate-500">
          Required for gated models like Llama, Gemma, or other restricted access models
        </p>
      </div>

      {/* Trust Remote Code Checkbox */}
      <div className="flex items-start gap-3 p-3 bg-yellow-500/10 border border-yellow-500/30 rounded-lg">
        <input
          type="checkbox"
          id="trust-remote-code"
          checked={trustRemoteCode}
          onChange={(e) => setTrustRemoteCode(e.target.checked)}
          disabled={isSubmitting}
          className="mt-1 w-4 h-4 rounded border-yellow-500/50 bg-slate-900 text-yellow-500 focus:ring-yellow-500 focus:ring-offset-slate-950 disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="flex-1">
          <label htmlFor="trust-remote-code" className="block text-sm font-medium text-yellow-300 cursor-pointer">
            Trust Remote Code
          </label>
          <p className="mt-1 text-xs text-yellow-200/80">
            Some models (like Phi-4, CodeLlama, etc.) require executing custom code from the repository.
            Only enable this if you trust the model source.
          </p>
        </div>
      </div>

      {/* Validation Error */}
      {validationError && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {validationError}
        </div>
      )}

      {/* Download Button */}
      <button
        type="button"
        onClick={handleSubmit}
        disabled={!hfModelRepo || isSubmitting}
        className="w-full px-6 py-3 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-700 disabled:cursor-not-allowed rounded-lg flex items-center justify-center gap-2 transition-colors font-medium text-white"
      >
        {isSubmitting ? (
          <>
            <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
            Initiating Download...
          </>
        ) : (
          <>
            <Download className="w-5 h-5" />
            Download Model from HuggingFace
          </>
        )}
      </button>
    </div>
  );
}

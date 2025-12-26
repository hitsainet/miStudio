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
import { Download, Eye, EyeOff } from 'lucide-react';
import { QuantizationFormat } from '../../types/model';
import { ModelPreviewModal } from './ModelPreviewModal';
import { COMPONENTS } from '../../config/brand';

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
  const [showPreview, setShowPreview] = useState(false);
  const [showToken, setShowToken] = useState(false);

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

  const handlePreview = () => {
    if (!validateRepoId(hfModelRepo)) {
      return;
    }
    setShowPreview(true);
  };

  const handlePreviewDownload = async (selectedQuantization: string, previewTrustRemoteCode: boolean) => {
    // Update quantization and trustRemoteCode from preview selection
    setQuantization(selectedQuantization);
    setTrustRemoteCode(previewTrustRemoteCode);
    // Trigger download with selected quantization and trustRemoteCode
    setIsSubmitting(true);
    try {
      await onDownload(
        hfModelRepo.trim(),
        selectedQuantization,
        accessToken.trim() || undefined,
        previewTrustRemoteCode
      );
      setValidationError(null);
    } catch (error) {
      console.error('[ModelDownloadForm] Download failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Download failed';

      // Detect specific error types and provide helpful messages
      if (errorMessage.includes('trust_remote_code') || errorMessage.includes('trust remote code')) {
        setValidationError('This model requires executing custom code. Please enable "Trust Remote Code" below and try again.');
      } else if (errorMessage.includes('Unsupported architecture')) {
        setValidationError(`${errorMessage}\n\nThis model architecture is not yet supported. Please check the supported architectures list or try a different model.`);
      } else {
        setValidationError(errorMessage);
      }
    } finally {
      setIsSubmitting(false);
    }
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
      const errorMessage = error instanceof Error ? error.message : 'Download failed';

      // Detect specific error types and provide helpful messages
      if (errorMessage.includes('trust_remote_code') || errorMessage.includes('trust remote code')) {
        setValidationError('This model requires executing custom code. Please enable "Trust Remote Code" below and try again.');
      } else if (errorMessage.includes('Unsupported architecture')) {
        setValidationError(`${errorMessage}\n\nThis model architecture is not yet supported. Please check the supported architectures list or try a different model.`);
      } else {
        setValidationError(errorMessage);
      }
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
    <div className={`${COMPONENTS.card.base} p-6 space-y-4`}>
      <div className="grid grid-cols-2 gap-4">
        {/* HuggingFace Repository Input */}
        <div>
          <label htmlFor="model-repo" className="block text-sm font-medium text-slate-300 mb-2">
            HuggingFace Model Repository
          </label>
          <input
            id="model-repo"
            type="text"
            autoComplete="off"
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
        <label htmlFor="access-token" className="block text-sm font-medium text-slate-300 mb-2">
          Access Token <span className="text-slate-500">(optional, for gated models)</span>
        </label>
        <div className="relative">
          <input
            id="access-token"
            name="model-hf-credential-input"
            type="text"
            autoComplete="off"
            data-lpignore="true"
            data-1p-ignore="true"
            data-form-type="other"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
            value={accessToken}
            onChange={(e) => setAccessToken(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isSubmitting}
            className="w-full px-4 py-2 pr-10 bg-slate-900 border border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm text-slate-100 placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            style={{ WebkitTextSecurity: showToken ? 'none' : 'disc' } as React.CSSProperties}
          />
          {accessToken && (
            <button
              type="button"
              onMouseDown={() => setShowToken(true)}
              onMouseUp={() => setShowToken(false)}
              onMouseLeave={() => setShowToken(false)}
              onTouchStart={() => setShowToken(true)}
              onTouchEnd={() => setShowToken(false)}
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-slate-400 hover:text-slate-300 transition-colors"
              title="Hold to reveal token"
              tabIndex={-1}
            >
              {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          )}
        </div>
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

      {/* Action Buttons */}
      <div className="grid grid-cols-2 gap-3">
        <button
          type="button"
          onClick={handlePreview}
          disabled={!hfModelRepo || isSubmitting}
          className={`py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.secondary}`}
        >
          <Eye className="w-5 h-5" />
          Preview
        </button>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!hfModelRepo || isSubmitting}
          className={`py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.primary}`}
        >
          {isSubmitting ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-2 border-white border-t-transparent"></div>
              Downloading...
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              Download
            </>
          )}
        </button>
      </div>

      {/* Model Preview Modal */}
      {showPreview && (
        <ModelPreviewModal
          repoId={hfModelRepo}
          onClose={() => setShowPreview(false)}
          onDownload={handlePreviewDownload}
        />
      )}
    </div>
  );
}

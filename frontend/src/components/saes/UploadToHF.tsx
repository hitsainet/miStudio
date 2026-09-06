/**
 * UploadToHF - Modal for uploading SAEs to HuggingFace.
 *
 * Features:
 * - Display SAE info (name, source, size)
 * - HuggingFace repository ID input with validation
 * - File path input for target location in repo
 * - Required access token input (masked with reveal)
 * - Option to create new repository if it doesn't exist
 * - Option to make repository private
 * - Custom commit message
 * - Upload progress and success state with link to HuggingFace
 */

import { useState, useEffect } from 'react';
import { Upload, X, Cloud, Eye, EyeOff, ExternalLink, Loader, CheckCircle, FileCode } from 'lucide-react';
import { SAE, SAESource, SAEUploadRequest, SAEUploadResponse } from '../../types/sae';
import { useSAEsStore } from '../../stores/saesStore';
import { useModelsStore } from '../../stores/modelsStore';
import { COMPONENTS } from '../../config/brand';

/** Abbreviate hook type names for compact filepaths */
const HOOK_ABBREVIATIONS: Record<string, string> = {
  'residual': 'res',
  'hook_resid_pre': 'res',
  'hook_resid_post': 'res',
  'mlp': 'mlp',
  'hook_mlp_out': 'mlp',
  'attention': 'attn',
  'hook_attn_out': 'attn',
};

/** Get abbreviated hook type, falling back to first 4 chars */
function getHookAbbrev(hookType: string): string {
  const lower = hookType.toLowerCase();
  return HOOK_ABBREVIATIONS[lower] ?? lower.slice(0, 4);
}

/** Determine zero-padding width from total layer count */
function getLayerPadWidth(totalLayers: number | undefined): number {
  if (!totalLayers || totalLayers <= 0) return 2; // default 2 digits
  return Math.max(2, totalLayers.toString().length);
}

/** Format latent width as compact string: 8192 → "8k", 16384 → "16k", 65536 → "65k" */
function formatWidth(nFeatures: number): string {
  if (nFeatures >= 1000 && nFeatures % 1000 === 0) {
    return `${nFeatures / 1000}k`;
  }
  if (nFeatures >= 1024 && nFeatures % 1024 === 0) {
    return `${nFeatures / 1024}k`;
  }
  return nFeatures.toString();
}

interface UploadToHFProps {
  sae: SAE;
  isOpen: boolean;
  onClose: () => void;
  onUploadComplete?: (response: SAEUploadResponse) => void;
}

export function UploadToHF({ sae, isOpen, onClose, onUploadComplete }: UploadToHFProps) {
  const [repoId, setRepoId] = useState('');
  const [filepath, setFilepath] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [createRepo, setCreateRepo] = useState(false);
  const [isPrivate, setIsPrivate] = useState(false);
  const [commitMessage, setCommitMessage] = useState('');
  const [showToken, setShowToken] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<SAEUploadResponse | null>(null);

  const { uploadSAE } = useSAEsStore();
  const { models } = useModelsStore();

  // Set default filepath and commit message based on SAE fields when modal opens
  useEffect(() => {
    if (isOpen && sae) {
      if (sae.layer !== null && sae.layer !== undefined && sae.hook_type) {
        // Structured path: use SAE fields for clean, sortable filepath
        const model = sae.model_id ? models.find(m => m.id === sae.model_id) : undefined;
        const totalLayers = model?.architecture_config?.num_hidden_layers;
        const padWidth = getLayerPadWidth(totalLayers);
        const paddedLayer = sae.layer.toString().padStart(padWidth, '0');
        const hookAbbrev = getHookAbbrev(sae.hook_type);

        const widthSegment = sae.n_features ? `width_${formatWidth(sae.n_features)}/` : '';
        setFilepath(`layer_${paddedLayer}/${widthSegment}${hookAbbrev}`);
        const widthLabel = sae.n_features ? ` ${formatWidth(sae.n_features)}` : '';
        setCommitMessage(`info: push l${paddedLayer}${widthLabel} ${hookAbbrev} sae trained by miStudio`);
      } else {
        // Fallback: sanitize full SAE name
        const safeName = sae.name.toLowerCase().replace(/[^a-z0-9-_]/g, '-').replace(/-+/g, '-').replace(/-$/, '');
        setFilepath(`saes/${safeName}`);
        setCommitMessage('Upload SAE via miStudio');
      }
    }
  }, [isOpen, sae, models]);

  // Reset form when modal closes
  useEffect(() => {
    if (!isOpen) {
      setRepoId('');
      setFilepath('');
      setAccessToken('');
      setCreateRepo(false);
      setIsPrivate(false);
      setCommitMessage('');
      setShowToken(false);
      setValidationError(null);
      setIsUploading(false);
      setUploadResult(null);
    }
  }, [isOpen]);

  const validateForm = (): boolean => {
    // Check repo ID format
    if (!repoId.trim()) {
      setValidationError('Repository ID is required');
      return false;
    }

    const repoIdPattern = /^[\w-]+\/[\w.-]+$/;
    if (!repoIdPattern.test(repoId)) {
      setValidationError('Invalid repository format. Use: username/repo-name');
      return false;
    }

    // Check filepath
    if (!filepath.trim()) {
      setValidationError('File path is required');
      return false;
    }

    // Check access token
    if (!accessToken.trim()) {
      setValidationError('Access token is required for uploads');
      return false;
    }

    setValidationError(null);
    return true;
  };

  const handleUpload = async () => {
    if (!validateForm()) {
      return;
    }

    setIsUploading(true);
    setValidationError(null);

    try {
      const request: SAEUploadRequest = {
        sae_id: sae.id,
        repo_id: repoId.trim(),
        filepath: filepath.trim(),
        access_token: accessToken.trim(),
        create_repo: createRepo,
        private: isPrivate,
        commit_message: commitMessage.trim() || undefined,
      };

      const response = await uploadSAE(request);
      setUploadResult(response);
      onUploadComplete?.(response);
    } catch (error) {
      console.error('[UploadToHF] Upload failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Upload failed';
      setValidationError(errorMessage);
    } finally {
      setIsUploading(false);
    }
  };

  const formatFileSize = (bytes: number | null): string => {
    if (bytes === null) return 'Unknown size';
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${Math.round(mb)} MB`;
  };

  const getSourceLabel = (source: SAESource): string => {
    switch (source) {
      case SAESource.TRAINED:
        return 'Trained locally';
      case SAESource.LOCAL:
        return 'Local file';
      case SAESource.HUGGINGFACE:
        return 'From HuggingFace';
      default:
        return source;
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/60 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className={`relative ${COMPONENTS.card.base} w-full max-w-lg mx-4 p-6 max-h-[90vh] overflow-y-auto`}>
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Cloud className="w-5 h-5 text-yellow-400" />
            <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Upload to HuggingFace</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Success State */}
        {uploadResult ? (
          <div className="space-y-4">
            <div className="flex items-center gap-3 p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
              <CheckCircle className="w-6 h-6 text-emerald-400 flex-shrink-0" />
              <div>
                <p className="text-emerald-400 font-medium">Upload successful!</p>
                <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
                  Your SAE has been uploaded to HuggingFace.
                </p>
              </div>
            </div>

            <div className="p-4 bg-slate-100 dark:bg-slate-800/50 rounded-lg space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Repository</span>
                <span className="text-sm text-slate-800 dark:text-slate-200 font-mono">{uploadResult.repo_id}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-slate-600 dark:text-slate-400">Path</span>
                <span className="text-sm text-slate-800 dark:text-slate-200 font-mono">{uploadResult.filepath}</span>
              </div>
              {uploadResult.commit_hash && (
                <div className="flex items-center justify-between">
                  <span className="text-sm text-slate-600 dark:text-slate-400">Commit</span>
                  <span className="text-sm text-slate-800 dark:text-slate-200 font-mono">
                    {uploadResult.commit_hash.substring(0, 8)}
                  </span>
                </div>
              )}
            </div>

            <a
              href={uploadResult.url}
              target="_blank"
              rel="noopener noreferrer"
              className={`w-full py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.primary}`}
            >
              <ExternalLink className="w-5 h-5" />
              View on HuggingFace
            </a>

            <button
              onClick={onClose}
              className={`w-full py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.secondary}`}
            >
              Close
            </button>
          </div>
        ) : (
          <div className="space-y-4">
            {/* SAE Info */}
            <div className="p-4 bg-slate-100 dark:bg-slate-800/50 rounded-lg">
              <div className="flex items-start gap-3">
                <FileCode className="w-5 h-5 text-slate-600 dark:text-slate-400 mt-0.5 flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="font-medium text-slate-800 dark:text-slate-200 truncate">{sae.name}</p>
                  <div className="flex items-center gap-3 mt-1 text-xs text-slate-500">
                    <span>{getSourceLabel(sae.source)}</span>
                    <span>•</span>
                    <span>{formatFileSize(sae.file_size_bytes)}</span>
                    {sae.n_features && (
                      <>
                        <span>•</span>
                        <span>{sae.n_features.toLocaleString()} features</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* Repository ID Input */}
            <div>
              <label htmlFor="upload-repo" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Target Repository <span className="text-red-400">*</span>
              </label>
              <input
                id="upload-repo"
                type="text"
                placeholder="e.g., your-username/my-saes"
                value={repoId}
                onChange={(e) => {
                  setRepoId(e.target.value);
                  setValidationError(null);
                }}
                disabled={isUploading}
                className="w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
            </div>

            {/* File Path Input */}
            <div>
              <label htmlFor="upload-filepath" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                File Path in Repository <span className="text-red-400">*</span>
              </label>
              <input
                id="upload-filepath"
                type="text"
                placeholder="e.g., saes/my-sae"
                value={filepath}
                onChange={(e) => {
                  setFilepath(e.target.value);
                  setValidationError(null);
                }}
                disabled={isUploading}
                className="w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-mono text-sm"
              />
              <p className="text-xs text-slate-500 mt-1">
                Directory path where SAE files will be stored
              </p>
            </div>

            {/* Access Token Input */}
            <div>
              <label htmlFor="upload-token" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                HuggingFace Access Token <span className="text-red-400">*</span>
              </label>
              <div className="relative">
                <input
                  id="upload-token"
                  type={showToken ? 'text' : 'password'}
                  placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
                  value={accessToken}
                  onChange={(e) => {
                    setAccessToken(e.target.value);
                    setValidationError(null);
                  }}
                  disabled={isUploading}
                  className="w-full px-4 py-2 pr-10 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                />
                {accessToken && (
                  <button
                    type="button"
                    onMouseDown={() => setShowToken(true)}
                    onMouseUp={() => setShowToken(false)}
                    onMouseLeave={() => setShowToken(false)}
                    onTouchStart={() => setShowToken(true)}
                    onTouchEnd={() => setShowToken(false)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
                    title="Hold to reveal token"
                    tabIndex={-1}
                  >
                    {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                )}
              </div>
              <p className="text-xs text-slate-500 mt-1">
                Requires write access. Get a token at{' '}
                <a
                  href="https://huggingface.co/settings/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-emerald-400 hover:text-emerald-300"
                >
                  huggingface.co/settings/tokens
                </a>
              </p>
            </div>

            {/* Repository Options */}
            <div className="space-y-3">
              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={createRepo}
                  onChange={(e) => setCreateRepo(e.target.checked)}
                  disabled={isUploading}
                  className="rounded border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900 disabled:opacity-50"
                />
                <div>
                  <span className="text-sm text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white">
                    Create repository if it doesn't exist
                  </span>
                </div>
              </label>

              <label className="flex items-center gap-3 cursor-pointer group">
                <input
                  type="checkbox"
                  checked={isPrivate}
                  onChange={(e) => setIsPrivate(e.target.checked)}
                  disabled={isUploading || !createRepo}
                  className="rounded border-slate-300 dark:border-slate-600 bg-slate-100 dark:bg-slate-700 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900 disabled:opacity-50"
                />
                <div>
                  <span className={`text-sm ${createRepo ? 'text-slate-700 dark:text-slate-300 group-hover:text-slate-900 dark:group-hover:text-white' : 'text-slate-500'}`}>
                    Make repository private
                  </span>
                  {!createRepo && (
                    <span className="text-xs text-slate-500 ml-2">
                      (only for new repositories)
                    </span>
                  )}
                </div>
              </label>
            </div>

            {/* Commit Message Input */}
            <div>
              <label htmlFor="upload-commit" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
                Commit Message <span className="text-slate-500">(optional)</span>
              </label>
              <input
                id="upload-commit"
                type="text"
                placeholder="Upload SAE from miStudio"
                value={commitMessage}
                onChange={(e) => setCommitMessage(e.target.value)}
                disabled={isUploading}
                className="w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              />
            </div>

            {/* Validation Error */}
            {validationError && (
              <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
                {validationError}
              </div>
            )}

            {/* Upload Button */}
            <button
              type="button"
              onClick={handleUpload}
              disabled={isUploading}
              className={`w-full py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.primary}`}
            >
              {isUploading ? (
                <>
                  <Loader className="w-5 h-5 animate-spin" />
                  Uploading...
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5" />
                  Upload to HuggingFace
                </>
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * DownloadFromHF - Form for downloading SAEs from HuggingFace.
 *
 * Features:
 * - HuggingFace repository ID input with validation
 * - Preview repository to discover SAE files (SAELens format)
 * - Select specific SAE files to download
 * - Optional access token for gated repos
 * - Support for custom SAE names
 */

import { useState, useEffect, useMemo } from 'react';
import { Download, Search, CheckCircle, Cloud, FolderOpen, Loader, Box, Eye, EyeOff } from 'lucide-react';
import { HFFileInfo } from '../../types/sae';
import { useSAEsStore } from '../../stores/saesStore';
import { useModelsStore } from '../../stores/modelsStore';
import { COMPONENTS } from '../../config/brand';

/** An SAE directory group containing all files for one SAE */
interface SAEGroup {
  dirPath: string;
  files: HFFileInfo[];
  totalSizeBytes: number;
  layer: number | undefined;
  nFeatures: number | undefined;
}

/** Group individual SAE files into directory-level SAE groups */
function groupFilesIntoSAEs(saeFiles: HFFileInfo[]): SAEGroup[] {
  const groups = new Map<string, SAEGroup>();
  for (const file of saeFiles) {
    const lastSlash = file.filepath.lastIndexOf('/');
    const dirPath = lastSlash > 0 ? file.filepath.substring(0, lastSlash) : '.';
    if (!groups.has(dirPath)) {
      groups.set(dirPath, {
        dirPath, files: [], totalSizeBytes: 0,
        layer: file.layer ?? undefined,
        nFeatures: file.n_features ?? undefined,
      });
    }
    const group = groups.get(dirPath)!;
    group.files.push(file);
    group.totalSizeBytes += file.size_bytes || 0;
    if (group.layer === undefined && file.layer != null) group.layer = file.layer;
    if (!group.nFeatures && file.n_features) group.nFeatures = file.n_features;
  }
  return Array.from(groups.values());
}

interface DownloadFromHFProps {
  onDownloadComplete?: () => void;
}

export function DownloadFromHF({ onDownloadComplete }: DownloadFromHFProps) {
  const [repoId, setRepoId] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [customName, setCustomName] = useState('');
  const [selectedDirs, setSelectedDirs] = useState<Set<string>>(new Set());
  const [selectedModelId, setSelectedModelId] = useState<string>('');
  const [validationError, setValidationError] = useState<string | null>(null);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState({ current: 0, total: 0 });
  const [showToken, setShowToken] = useState(false);

  const {
    hfPreview,
    hfPreviewLoading,
    hfPreviewError,
    previewHFRepository,
    clearHFPreview,
    downloadSAE,
  } = useSAEsStore();

  const { models, fetchModels } = useModelsStore();

  // Fetch models on mount for the model selector
  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  // Filter to only show ready models
  const readyModels = models.filter(m => m.status === 'ready');

  const validateRepoId = (input: string): boolean => {
    if (!input.trim()) {
      setValidationError('Repository ID is required');
      return false;
    }

    // HuggingFace repo format: username/repo-name or org/repo-name
    const repoIdPattern = /^[\w-]+\/[\w.-]+$/;
    if (!repoIdPattern.test(input)) {
      setValidationError('Invalid repository format. Use: username/repo-name');
      return false;
    }

    setValidationError(null);
    return true;
  };

  const handlePreview = async () => {
    if (!validateRepoId(repoId)) {
      return;
    }

    setSelectedDirs(new Set());
    try {
      await previewHFRepository({
        repo_id: repoId.trim(),
        access_token: accessToken.trim() || undefined,
      });
    } catch (error) {
      console.error('[DownloadFromHF] Preview failed:', error);
    }
  };

  // Compute SAE groups from preview files
  const saeGroups = useMemo(() => {
    if (!hfPreview) return [];
    return groupFilesIntoSAEs(hfPreview.sae_files);
  }, [hfPreview]);

  const toggleDirSelection = (dirPath: string) => {
    setSelectedDirs(prev => {
      const next = new Set(prev);
      if (next.has(dirPath)) {
        next.delete(dirPath);
      } else {
        next.add(dirPath);
      }
      return next;
    });
  };

  const selectAllInLayer = (groups: SAEGroup[]) => {
    setSelectedDirs(prev => {
      const next = new Set(prev);
      groups.forEach(g => next.add(g.dirPath));
      return next;
    });
  };

  const deselectAllInLayer = (groups: SAEGroup[]) => {
    setSelectedDirs(prev => {
      const next = new Set(prev);
      groups.forEach(g => next.delete(g.dirPath));
      return next;
    });
  };

  const isLayerFullySelected = (groups: SAEGroup[]) => {
    return groups.every(g => selectedDirs.has(g.dirPath));
  };

  const handleDownload = async () => {
    if (selectedDirs.size === 0 || !hfPreview) {
      setValidationError('Please select at least one SAE to download');
      return;
    }

    if (!selectedModelId) {
      setValidationError('Please select a model to link these SAEs with for steering');
      return;
    }

    const dirsToDownload = Array.from(selectedDirs);

    setIsDownloading(true);
    setDownloadProgress({ current: 0, total: dirsToDownload.length });

    try {
      // Download each selected SAE directory sequentially
      for (let i = 0; i < dirsToDownload.length; i++) {
        setDownloadProgress({ current: i + 1, total: dirsToDownload.length });

        await downloadSAE({
          repo_id: hfPreview.repo_id,
          filepath: dirsToDownload[i],
          name: customName.trim() || undefined,
          access_token: accessToken.trim() || undefined,
          model_id: selectedModelId,
        });
      }

      // Clear form on success
      setRepoId('');
      setAccessToken('');
      setCustomName('');
      setSelectedDirs(new Set());
      setSelectedModelId('');
      clearHFPreview();
      onDownloadComplete?.();
    } catch (error) {
      console.error('[DownloadFromHF] Download failed:', error);
      const errorMessage = error instanceof Error ? error.message : 'Download failed';
      setValidationError(errorMessage);
    } finally {
      setIsDownloading(false);
      setDownloadProgress({ current: 0, total: 0 });
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !hfPreviewLoading) {
      handlePreview();
    }
  };

  const formatFileSize = (bytes: number): string => {
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
      return `${(mb / 1024).toFixed(1)} GB`;
    }
    return `${Math.round(mb)} MB`;
  };

  return (
    <div className={`${COMPONENTS.card.base} p-6 space-y-4`}>
      <div className="flex items-center gap-2 mb-2">
        <Cloud className="w-5 h-5 text-cyan-400" />
        <h3 className="text-lg font-semibold text-slate-900 dark:text-slate-100">Download from HuggingFace</h3>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* HuggingFace Repository Input */}
        <div>
          <label htmlFor="sae-repo" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            HuggingFace Repository
          </label>
          <input
            id="sae-repo"
            type="text"
            autoComplete="off"
            placeholder="e.g., jbloom/GPT2-Small-SAEs-Reformatted"
            value={repoId}
            onChange={(e) => {
              setRepoId(e.target.value);
              setValidationError(null);
            }}
            onKeyPress={handleKeyPress}
            disabled={hfPreviewLoading || isDownloading}
            className="w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          />
        </div>

        {/* Custom Name Input */}
        <div>
          <label htmlFor="sae-name" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
            Custom Name <span className="text-slate-500">(optional)</span>
          </label>
          <input
            id="sae-name"
            type="text"
            autoComplete="off"
            placeholder="Auto-generated from file path"
            value={customName}
            onChange={(e) => setCustomName(e.target.value)}
            disabled={hfPreviewLoading || isDownloading}
            className="w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          />
        </div>
      </div>

      {/* Access Token Input */}
      <div>
        <label htmlFor="sae-access-token" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          Access Token <span className="text-slate-500">(optional, for gated repos)</span>
        </label>
        <div className="relative">
          <input
            id="sae-access-token"
            name="sae-hf-credential-input"
            type="text"
            autoComplete="off"
            data-lpignore="true"
            data-1p-ignore="true"
            data-form-type="other"
            placeholder="hf_xxxxxxxxxxxxxxxxxxxx"
            value={accessToken}
            onChange={(e) => setAccessToken(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={hfPreviewLoading || isDownloading}
            className="w-full px-4 py-2 pr-10 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 font-mono text-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
              className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-slate-600 dark:text-slate-400 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              title="Hold to reveal token"
              tabIndex={-1}
            >
              {showToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </button>
          )}
        </div>
      </div>

      {/* Model Selector - Required for steering */}
      <div>
        <label htmlFor="sae-model" className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2">
          <span className="flex items-center gap-2">
            <Box className="w-4 h-4" />
            Link to Model <span className="text-red-400">*</span>
          </span>
          <span className="text-xs text-slate-500 font-normal mt-1 block">
            Select a downloaded model to use with this SAE for steering
          </span>
        </label>
        <select
          id="sae-model"
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={hfPreviewLoading || isDownloading}
          className="w-full px-4 py-2 bg-white dark:bg-slate-900 border border-slate-300 dark:border-slate-700 rounded-lg focus:outline-none focus:border-emerald-500 text-slate-900 dark:text-slate-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <option value="">Select a model...</option>
          {readyModels.map((model) => (
            <option key={model.id} value={model.id}>
              {model.name} {model.repo_id ? `(${model.repo_id})` : ''}
            </option>
          ))}
        </select>
        {readyModels.length === 0 && (
          <p className="text-xs text-cyan-400 mt-1">
            No models available. Download a model from the Models panel first.
          </p>
        )}
      </div>

      {/* Preview Button */}
      <button
        type="button"
        onClick={handlePreview}
        disabled={!repoId.trim() || hfPreviewLoading || isDownloading}
        className={`w-full py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.secondary}`}
      >
        {hfPreviewLoading ? (
          <>
            <Loader className="w-5 h-5 animate-spin" />
            Scanning Repository...
          </>
        ) : (
          <>
            <Search className="w-5 h-5" />
            Preview Repository
          </>
        )}
      </button>

      {/* Preview Error */}
      {hfPreviewError && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {hfPreviewError}
        </div>
      )}

      {/* Preview Results */}
      {hfPreview && (
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h4 className="text-sm font-medium text-slate-700 dark:text-slate-300">
              Found {saeGroups.length} SAE{saeGroups.length !== 1 ? 's' : ''}
            </h4>
            {hfPreview.model_name && (
              <span className="text-xs text-slate-500">
                Model: {hfPreview.model_name}
              </span>
            )}
          </div>

          {saeGroups.length === 0 ? (
            <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-lg text-cyan-400 text-sm">
              No SAELens format files found in this repository. Make sure the repository contains
              cfg.json and/or sae_weights.safetensors files.
            </div>
          ) : (
            <div className="max-h-60 overflow-y-auto">
              {/* Group SAEs by layer and render with sticky headers */}
              {(() => {
                // Group SAE directories by layer
                const groupsByLayer = new Map<number | 'unknown', SAEGroup[]>();
                saeGroups.forEach((group) => {
                  const layerKey = group.layer ?? 'unknown';
                  if (!groupsByLayer.has(layerKey)) {
                    groupsByLayer.set(layerKey, []);
                  }
                  groupsByLayer.get(layerKey)!.push(group);
                });

                // Sort layers numerically (unknown last)
                const sortedLayers = Array.from(groupsByLayer.keys()).sort((a, b) => {
                  if (a === 'unknown') return 1;
                  if (b === 'unknown') return -1;
                  return (a as number) - (b as number);
                });

                return sortedLayers.map((layerKey) => {
                  const layerGroups = groupsByLayer.get(layerKey)!;
                  const allSelected = isLayerFullySelected(layerGroups);

                  return (
                    <div key={layerKey} className="relative">
                      {/* Sticky layer header with select all toggle */}
                      <div className="sticky top-0 z-10 bg-white/95 dark:bg-slate-900/95 backdrop-blur-sm border-b border-slate-200 dark:border-slate-700 px-3 py-2 flex items-center justify-between">
                        <span className="text-xs font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider">
                          {layerKey === 'unknown' ? 'Unknown Layer' : `Layer ${layerKey}`}
                        </span>
                        <button
                          type="button"
                          onClick={() => allSelected ? deselectAllInLayer(layerGroups) : selectAllInLayer(layerGroups)}
                          className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
                        >
                          {allSelected ? 'Deselect All' : 'Select All'}
                        </button>
                      </div>
                      {/* SAE directories in this layer */}
                      <div className="space-y-2 p-2">
                        {layerGroups.map((group) => {
                          const isSelected = selectedDirs.has(group.dirPath);
                          return (
                            <div
                              key={group.dirPath}
                              onClick={() => toggleDirSelection(group.dirPath)}
                              className={`p-3 rounded-lg cursor-pointer transition-colors ${
                                isSelected
                                  ? 'bg-emerald-500/20 border border-emerald-500/50'
                                  : 'bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 hover:border-slate-400 dark:hover:border-slate-600'
                              }`}
                            >
                              <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2 min-w-0">
                                  {isSelected ? (
                                    <CheckCircle className="w-4 h-4 text-emerald-400 flex-shrink-0" />
                                  ) : (
                                    <FolderOpen className="w-4 h-4 text-slate-600 dark:text-slate-400 flex-shrink-0" />
                                  )}
                                  <span className="text-sm text-slate-800 dark:text-slate-200 truncate font-mono">
                                    {group.dirPath}
                                  </span>
                                </div>
                                <span className="text-xs text-slate-500 flex-shrink-0 ml-2">
                                  {group.totalSizeBytes > 0 ? formatFileSize(group.totalSizeBytes) : `${group.files.length} files`}
                                </span>
                              </div>
                              <div className="mt-1 flex items-center gap-3 text-xs text-slate-500">
                                <span>{group.files.length} file{group.files.length !== 1 ? 's' : ''}</span>
                                {group.nFeatures && (
                                  <><span>•</span><span>{group.nFeatures.toLocaleString()} features</span></>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                });
              })()}
            </div>
          )}
        </div>
      )}

      {/* Validation Error */}
      {validationError && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg text-red-400 text-sm">
          {validationError}
        </div>
      )}

      {/* Download Button */}
      {hfPreview && hfPreview.sae_files.length > 0 && (
        <button
          type="button"
          onClick={handleDownload}
          disabled={selectedDirs.size === 0 || !selectedModelId || isDownloading}
          className={`w-full py-3 flex items-center justify-center gap-2 ${COMPONENTS.button.primary}`}
        >
          {isDownloading ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              Downloading {downloadProgress.current} of {downloadProgress.total}...
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              {selectedDirs.size === 0
                ? 'Select SAEs to Download'
                : `Download ${selectedDirs.size} Selected SAE${selectedDirs.size !== 1 ? 's' : ''}`}
            </>
          )}
        </button>
      )}
    </div>
  );
}

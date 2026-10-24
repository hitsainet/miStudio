/**
 * DownloadForm component for downloading datasets from HuggingFace.
 *
 * This component renders a form to input repository ID and access token.
 */

import React, { useState } from 'react';
import { Download, Eye, EyeOff } from 'lucide-react';
import { validateHfRepoId } from '../../utils/validators';
import { DatasetPreviewModal } from './DatasetPreviewModal';
import { COMPONENTS } from '../../config/brand';
import type { Dataset } from '../../types/dataset';
import { shortCommit } from './RevisionBadge';

interface DownloadFormProps {
  /** Resolves to the created dataset when the caller has it, so the form can show the pinned commit. */
  onDownload: (
    repoId: string,
    accessToken?: string,
    split?: string,
    config?: string,
    revision?: string,
  ) => Promise<Dataset | void | undefined>;
  className?: string;
}

export function DownloadForm({ onDownload, className = '' }: DownloadFormProps) {
  const [hfRepo, setHfRepo] = useState('');
  const [accessToken, setAccessToken] = useState('');
  const [split, setSplit] = useState('');
  const [config, setConfig] = useState('');
  const [revision, setRevision] = useState('');
  /** The commit the backend resolved for the last successful download (035 FR-8). */
  const [pinned, setPinned] = useState<{ repo: string; commit: string } | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showPreview, setShowPreview] = useState(false);
  const [showToken, setShowToken] = useState(false);

  const showPinned = (created: Dataset | void | undefined) => {
    if (created && created.revision) {
      setPinned({ repo: created.hf_repo_id || hfRepo, commit: created.revision });
    } else {
      setPinned(null);
    }
  };

  const handlePreview = () => {
    // Validate repository ID before showing preview
    const validation = validateHfRepoId(hfRepo);
    if (validation !== true) {
      setError(validation);
      return;
    }
    setError(null);
    setShowPreview(true);
  };

  const handleSelectSplit = (selectedSplit: string) => {
    // Update the split field with the selected split from preview
    setSplit(selectedSplit);
  };

  const handleDownloadAll = async () => {
    // Download all splits (no split parameter)
    setIsSubmitting(true);
    try {
      const created = await onDownload(
        hfRepo,
        accessToken || undefined,
        undefined, // No split = download all
        config || undefined,
        revision.trim() || undefined
      );
      showPinned(created);
      // Reset form on success
      setHfRepo('');
      setAccessToken('');
      setSplit('');
      setConfig('');
      setRevision('');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to download dataset';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setPinned(null);

    // Validate repository ID
    const validation = validateHfRepoId(hfRepo);
    if (validation !== true) {
      setError(validation);
      return;
    }

    setIsSubmitting(true);

    try {
      const created = await onDownload(
        hfRepo,
        accessToken || undefined,
        split || undefined,
        config || undefined,
        revision.trim() || undefined
      );
      showPinned(created);
      // Reset form on success
      setHfRepo('');
      setAccessToken('');
      setSplit('');
      setConfig('');
      setRevision('');
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to download dataset';
      setError(errorMessage);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      className={`${COMPONENTS.card.base} p-6 ${className}`}
    >
      <h2 className="text-lg font-semibold text-slate-900 dark:text-slate-100 mb-4">
        Download from HuggingFace
      </h2>

      <div className="space-y-4">
        <div>
          <label
            htmlFor="hf-repo"
            className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2"
          >
            Repository ID
          </label>
          <input
            id="hf-repo"
            type="text"
            autoComplete="off"
            value={hfRepo}
            onChange={(e) => setHfRepo(e.target.value)}
            placeholder="publisher/dataset-name"
            className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            disabled={isSubmitting}
            required
          />
        </div>

        <div>
          <label
            htmlFor="access-token"
            className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2"
          >
            Access Token (optional)
          </label>
          <div className="relative">
            <input
              id="access-token"
              name="dataset-hf-credential-input"
              type="text"
              autoComplete="off"
              data-lpignore="true"
              data-1p-ignore="true"
              data-form-type="other"
              value={accessToken}
              onChange={(e) => setAccessToken(e.target.value)}
              placeholder="hf_..."
              className="w-full px-4 py-2 pr-10 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 font-mono text-sm focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={isSubmitting}
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
          <p className="text-xs text-slate-500 mt-1">
            Required for private or gated datasets
          </p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label
              htmlFor="split"
              className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2"
            >
              Split (optional)
            </label>
            <input
              id="split"
              type="text"
              autoComplete="off"
              value={split}
              onChange={(e) => setSplit(e.target.value)}
              placeholder="train, validation, test"
              className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={isSubmitting}
            />
            <p className="text-xs text-slate-500 mt-1">
              Dataset split to download
            </p>
          </div>

          <div>
            <label
              htmlFor="config"
              className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2"
            >
              Config (optional)
            </label>
            <input
              id="config"
              type="text"
              autoComplete="off"
              value={config}
              onChange={(e) => setConfig(e.target.value)}
              placeholder="en, zh, etc."
              className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
              disabled={isSubmitting}
            />
            <p className="text-xs text-slate-500 mt-1">
              Dataset configuration
            </p>
          </div>
        </div>

        <div>
          <label
            htmlFor="revision"
            className="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-2"
          >
            Revision (commit, branch or tag)
          </label>
          <input
            id="revision"
            type="text"
            autoComplete="off"
            value={revision}
            onChange={(e) => setRevision(e.target.value)}
            placeholder="main, v1.0, or a commit"
            className="w-full px-4 py-2 bg-white dark:bg-slate-800 border border-slate-300 dark:border-slate-700 rounded-sm text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 font-mono text-sm focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-transparent"
            disabled={isSubmitting}
          />
          <p className="text-xs text-slate-500 mt-1">
            Leave empty to pin the current head. A branch or tag is resolved to its commit.
          </p>
        </div>

        {pinned && (
          <div
            className="p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-sm text-sm text-emerald-400"
            data-testid="pinned-commit"
          >
            {pinned.repo} pinned to commit{' '}
            <code className="font-mono" title={pinned.commit}>
              {shortCommit(pinned.commit)}
            </code>
          </div>
        )}

        {error && (
          <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-sm text-sm text-red-400">
            {error}
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <button
            type="button"
            onClick={handlePreview}
            disabled={isSubmitting || !hfRepo}
            className={`flex items-center justify-center gap-2 ${COMPONENTS.button.secondary}`}
          >
            <Eye className="w-4 h-4" />
            Preview
          </button>
          <button
            type="submit"
            disabled={isSubmitting || !hfRepo}
            className={`flex items-center justify-center gap-2 ${COMPONENTS.button.primary}`}
          >
            <Download className="w-4 h-4" />
            {isSubmitting ? 'Downloading...' : 'Download'}
          </button>
        </div>
      </div>

      {/* Dataset Preview Modal */}
      {showPreview && (
        <DatasetPreviewModal
          repoId={hfRepo}
          config={config || undefined}
          accessToken={accessToken || undefined}
          onClose={() => setShowPreview(false)}
          onSelectSplit={handleSelectSplit}
          onDownloadAll={handleDownloadAll}
        />
      )}
    </form>
  );
}

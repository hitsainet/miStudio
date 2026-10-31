/**
 * RevisionBadge — the Hugging Face commit a dataset was downloaded at (Feature 035, FR-5/FR-8).
 *
 * A pinned row shows the first 12 characters in monospace, the full commit in the tooltip, and a
 * copy button. A row with no recorded commit says exactly that — "revision not recorded" — and
 * never "latest" or "head": the row predates pinned downloads, so which commit it holds is not
 * known, and a display that guessed would turn a missing fact into a false one.
 */
import { useState } from 'react';
import { Check, Copy, GitCommit } from 'lucide-react';

export const REVISION_NOT_RECORDED = 'revision not recorded';
export const SHORT_COMMIT_LENGTH = 12;

export function shortCommit(revision: string): string {
  return revision.slice(0, SHORT_COMMIT_LENGTH);
}

interface RevisionBadgeProps {
  revision?: string | null;
  className?: string;
}

export function RevisionBadge({ revision, className = '' }: RevisionBadgeProps) {
  const [copied, setCopied] = useState(false);

  if (!revision) {
    return (
      <span
        className={`inline-flex items-center gap-1 text-xs text-slate-500 italic ${className}`}
        data-testid="revision-badge"
        title="Downloaded before miStudio recorded commits; the exact revision is unknown"
      >
        <GitCommit className="w-3 h-3" />
        {REVISION_NOT_RECORDED}
      </span>
    );
  }

  const copy = async (e: React.MouseEvent) => {
    e.stopPropagation(); // the badge sits inside clickable cards
    try {
      await navigator.clipboard?.writeText(revision);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard refused (insecure context): the full value stays in the tooltip.
    }
  };

  return (
    <span
      className={`inline-flex items-center gap-1 text-xs text-slate-600 dark:text-slate-400 ${className}`}
      data-testid="revision-badge"
    >
      <GitCommit className="w-3 h-3" />
      <code className="font-mono" title={revision}>
        {shortCommit(revision)}
      </code>
      <button
        type="button"
        onClick={copy}
        className="p-0.5 rounded-control hover:text-slate-900 dark:hover:text-slate-100"
        title="Copy full commit"
        aria-label="Copy full commit"
      >
        {copied ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
      </button>
    </span>
  );
}

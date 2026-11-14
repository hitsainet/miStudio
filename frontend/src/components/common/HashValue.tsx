/**
 * HashValue — a 64-hex digest shown as its first 16 characters (Feature 039 FR-20, G2 (a)).
 *
 * The full value is in the tooltip and one click copies it whole: a digest is a join key, and a
 * reader comparing two of them needs the exact string, not the 16 characters that fit on a row. A
 * missing digest says "not recorded" — never a placeholder hash, which would read as a real one.
 * The clipboard pattern is `datasets/RevisionBadge.tsx`'s.
 */
import { useState } from 'react';
import { Check, Copy } from 'lucide-react';

export const HASH_NOT_RECORDED = 'not recorded';
export const SHORT_HASH_LENGTH = 16;

interface HashValueProps {
  value?: string | null;
  /** What the hash is, for the copy button's accessible name ("Copy <label>"). */
  label: string;
  testId?: string;
}

export function HashValue({ value, label, testId }: HashValueProps) {
  const [copied, setCopied] = useState(false);

  if (!value) {
    return (
      <span className="text-xs italic text-slate-500" data-testid={testId}>
        {HASH_NOT_RECORDED}
      </span>
    );
  }

  const copy = async (event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      await navigator.clipboard?.writeText(value);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard refused (insecure context): the full value stays in the tooltip.
    }
  };

  return (
    <span className="inline-flex items-center gap-1" data-testid={testId}>
      <code className="font-mono text-slate-300" title={value}>
        {value.slice(0, SHORT_HASH_LENGTH)}…
      </code>
      <button
        type="button"
        onClick={copy}
        className="rounded-control p-0.5 text-slate-500 hover:text-slate-100"
        aria-label={`Copy ${label}`}
        title={`Copy ${label}`}
      >
        {copied ? <Check className="h-3 w-3" /> : <Copy className="h-3 w-3" />}
      </button>
    </span>
  );
}

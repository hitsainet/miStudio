// A "Help" link: opens the Help screen at its topic. Beside it, "Manual" opens the topic's page
// in the public manual (config/help.ts `manual`).
// Origin: miDataworks frontend/src/components/help/HelpLink.tsx @ 32cf406 (ENH-004).
import { CircleHelp } from 'lucide-react';

import { helpTopic } from '../../config/help';
import { ManualLink } from './ManualLink';
import { openHelp } from './openHelp';

export function HelpLink({ topic, label = 'Help' }: { topic: string; label?: string }) {
  const manual = helpTopic(topic)?.manual;
  return (
    <span className="inline-flex items-center gap-3">
      <button
        type="button"
        data-testid={`help-link-${topic}`}
        onClick={() => openHelp(topic)}
        className="inline-flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-400 hover:underline rounded-control focus:outline-hidden focus-visible:ring-2 focus-visible:ring-emerald-500"
      >
        <CircleHelp className="w-3.5 h-3.5" aria-hidden="true" />
        {label}
      </button>
      {manual && <ManualLink page={manual} label="Manual" testId={`manual-link-${topic}`} />}
    </span>
  );
}

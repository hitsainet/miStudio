// The Help screen (ENH-004). One card per topic: a short summary in plain language, then "Read
// more in the manual". The topic a Help link or a `#/help/<topic>` address asked for is scrolled
// into view and marked current; the chips at the top are real `#/help/<topic>` links.
// Origin: miDataworks frontend/src/components/help/HelpPanel.tsx @ 32cf406 (ENH-004).
import { useEffect } from 'react';
import { CircleHelp } from 'lucide-react';

import { HELP_TOPICS } from '../../config/help';
import { helpHash, replaceHash } from '../../config/helpRoute';
import { ManualLink } from './ManualLink';
import { useHelpTopic } from './helpTopicStore';

export function HelpPanel() {
  const topic = useHelpTopic(s => s.topic);
  const open = useHelpTopic(s => s.open);

  useEffect(() => {
    if (topic) document.getElementById(`help-${topic}`)?.scrollIntoView?.({ block: 'start' });
  }, [topic]);

  return (
    <div className="max-w-4xl mx-auto px-6 py-6" data-testid="screen-help">
      <h1 className="text-2xl font-semibold text-slate-900 dark:text-slate-100 mb-2 flex items-center gap-2">
        <CircleHelp className="w-6 h-6 text-emerald-500" aria-hidden="true" />
        Help
      </h1>
      <p className="mb-4 text-sm text-slate-700 dark:text-slate-300">
        What each screen does, in short. These are summaries: the{' '}
        <ManualLink page="" label="miStudio manual" testId="manual-home" /> explains every screen in
        full.
      </p>
      <nav aria-label="Help topics" className="mb-6 flex flex-wrap gap-2 text-xs">
        {HELP_TOPICS.map(t => (
          <a
            key={t.id}
            href={helpHash(t.id)}
            data-testid={`help-chip-${t.id}`}
            onClick={e => {
              e.preventDefault();
              open(t.id);
              replaceHash(helpHash(t.id));
            }}
            className="rounded-full border border-slate-200 dark:border-slate-700 px-3 py-1 text-slate-700 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-800 focus:outline-hidden focus-visible:ring-2 focus-visible:ring-emerald-500"
          >
            {t.title}
          </a>
        ))}
      </nav>
      <div className="space-y-6">
        {HELP_TOPICS.map(t => (
          <section
            key={t.id}
            id={`help-${t.id}`}
            aria-labelledby={`help-${t.id}-title`}
            data-current={t.id === topic ? 'true' : undefined}
            className={`rounded-card border p-5 bg-white dark:bg-slate-900/50 scroll-mt-20 ${
              t.id === topic ? 'border-emerald-500' : 'border-slate-200 dark:border-slate-800'
            }`}
          >
            <h2
              id={`help-${t.id}-title`}
              className="font-semibold text-slate-900 dark:text-slate-100 mb-2"
            >
              {t.title}
            </h2>
            {t.body.map((p, i) => (
              <p key={i} className="text-sm text-slate-700 dark:text-slate-300 mb-2">
                {p}
              </p>
            ))}
            {t.steps && (
              <ol className="list-decimal pl-5 text-sm text-slate-700 dark:text-slate-300 space-y-1">
                {t.steps.map((s, i) => (
                  <li key={i}>{s}</li>
                ))}
              </ol>
            )}
            <div className="mt-3">
              <ManualLink
                page={t.manual}
                label="Read more in the manual"
                testId={`help-manual-${t.id}`}
              />
            </div>
          </section>
        ))}
      </div>
    </div>
  );
}

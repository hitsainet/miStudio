// Linkable Help topics (ENH-004, decision D2): `#/help` opens the Help screen and
// `#/help/<topic>` opens it at one topic, so a topic can be bookmarked or linked from the manual
// or a chat. miStudio has no router; this is the only part of the address it reads besides the
// J-Lens permalink (`#jlens?...`), which these functions never match and never overwrite.

const PREFIX = '#/help';

/** `{ topic }` when the fragment asks for the Help screen (topic null for the screen itself). */
export function parseHelpHash(hash: string | null | undefined): { topic: string | null } | null {
  if (!hash) return null;
  if (hash === PREFIX || hash === `${PREFIX}/`) return { topic: null };
  if (!hash.startsWith(`${PREFIX}/`)) return null;
  const raw = hash.slice(PREFIX.length + 1).split(/[/?#]/)[0];
  let topic: string;
  try {
    topic = decodeURIComponent(raw);
  } catch {
    return { topic: null };
  }
  return { topic: topic || null };
}

/** The fragment for the Help screen at a topic. */
export function helpHash(topic: string | null | undefined): string {
  return topic ? `${PREFIX}/${encodeURIComponent(topic)}` : PREFIX;
}

/** Write the fragment without adding a history entry or firing `hashchange`. */
export function replaceHash(hash: string): void {
  const { pathname, search } = window.location;
  window.history.replaceState(window.history.state, '', `${pathname}${search}${hash}`);
}

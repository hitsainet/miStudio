// Where the public miStudio manual lives (manual/, published from the public mirror to GitHub
// Pages). The sidebar's book icon, the Help screen and every per-screen Help link point into it;
// the in-app Help text stays a short summary. This is the ONE place the address is written: set
// VITE_MANUAL_BASE_URL at build time to point at another copy (a self-hosted build of manual/,
// for example). manualLinks.test.tsx checks that the default agrees with
// manual/docusaurus.config.ts (url + baseUrl) and that every page and heading a Help topic names
// exists in the manual's sources.
//
// Origin: miDataworks frontend/src/config/manual.ts @ 32cf406 (ENH-004).

export const DEFAULT_MANUAL_BASE_URL = 'https://hitsainet.github.io/miStudio/';

/** The manual's base URL, always ending in "/". An empty or missing value takes the default. */
export function manualBaseUrl(
  raw: string | undefined = import.meta.env.VITE_MANUAL_BASE_URL as string | undefined
): string {
  const value = (raw ?? '').trim() || DEFAULT_MANUAL_BASE_URL;
  return value.endsWith('/') ? value : `${value}/`;
}

export const MANUAL_BASE_URL = manualBaseUrl();

/** A manual page, e.g. "core-workflow/circuits#the-loop", as an absolute URL. */
export function manualUrl(page: string, base: string = MANUAL_BASE_URL): string {
  return `${base}${page.replace(/^\/+/, '')}`;
}

/**
 * Where this manual is published, derived from where it is built (042 FR-042.1, PD-7, TPD-14).
 *
 * ONE config builds TWO sites (PD-7), so no second config file can drift and no workflow needs a
 * variable that someone forgets at rename time:
 *   - the public mirror (`GITHUB_REPOSITORY` owner `hitsainet`): the custom domain at its root,
 *     https://docs-miworkbench.hitsai.net/ with baseUrl '/';
 *   - the private repository (any other owner): GitHub Pages serves it under '/<repo name>/', so the
 *     base path follows the repository's name by itself ('/miStudio/' before ENH-009 P1, '/miWorkbench/'
 *     after, R10). `url` stays the public domain, so the private build names no private owner
 *     (`check_operator_strings.py` refuses one; Docusaurus serialises the site config into the bundle,
 *     which is also why `organizationName` is the public owner in every case);
 *   - a local build (no `GITHUB_REPOSITORY`): the public values.
 *
 * Pure: no Node imports, so the frontend's Vitest can import it (frontend/src/test/manualLinks.test.tsx
 * asserts the five values of each case and that the in-app Help default equals the public url + baseUrl).
 */

export const PUBLIC_URL = 'https://docs-miworkbench.hitsai.net';
export const PUBLIC_OWNER = 'hitsainet';
/** The public repository's name (miWorkbench since ENH-009 P1; it was miStudio before). CI builds always take
 * the name from GITHUB_REPOSITORY, so only a local build uses this. */
export const LOCAL_REPOSITORY_NAME = 'miWorkbench';

export interface SiteAddress {
  url: string;
  baseUrl: string;
  organizationName: string;
  projectName: string;
  editUrl: string;
}

export function siteAddress(env: {GITHUB_REPOSITORY?: string}): SiteAddress {
  const slug = (env.GITHUB_REPOSITORY ?? '').trim();
  const [owner, name] = slug.includes('/') ? slug.split('/', 2) : ['', ''];
  const repo = name || LOCAL_REPOSITORY_NAME;
  const isPublic = !slug || owner.toLowerCase() === PUBLIC_OWNER;
  return {
    url: PUBLIC_URL,
    baseUrl: isPublic ? '/' : `/${repo}/`,
    organizationName: PUBLIC_OWNER,
    projectName: repo,
    editUrl: `https://github.com/${PUBLIC_OWNER}/${repo}/tree/main/manual/`,
  };
}

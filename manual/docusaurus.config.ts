import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

import {siteAddress} from './src/siteAddress';

// 042 FR-042.1 / PD-7: the five site values come from where this is built (GITHUB_REPOSITORY), so one
// config builds the public site (docs-miworkbench.hitsai.net, at its root) and the private repository's
// Pages site ('/<repo name>/'). frontend/src/test/manualLinks.test.tsx pins the values and this wiring.
const site = siteAddress(process.env);

const config: Config = {
  title: 'miWorkbench Manual',
  tagline: 'Interpretability Workbench',
  favicon: 'img/favicon.svg',

  // Published from the public hitsainet mirror via .github/workflows/deploy-manual.yml
  // (actions/deploy-pages) at the custom domain; see src/siteAddress.ts.
  url: site.url,
  baseUrl: site.baseUrl,
  organizationName: site.organizationName,
  projectName: site.projectName,
  trailingSlash: false,

  onBrokenLinks: 'throw',
  // ENH-004: the in-app Help screen deep-links to headings (`page#section`). A renamed heading
  // must fail the build here, not 404 to a page top in a user's browser.
  onBrokenAnchors: 'throw',

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'warn',
    },
  },
  themes: ['@docusaurus/theme-mermaid'],

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          editUrl: site.editUrl,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    navbar: {
      title: 'miWorkbench Manual',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'manualSidebar',
          position: 'left',
          label: 'Manual',
        },
        {to: '/getting-started/quickstart-tutorial', label: 'Quickstart', position: 'left'},
        {to: '/concepts/architecture', label: 'Concepts', position: 'left'},
        {to: '/reference/api/overview', label: 'Reference', position: 'left'},
        {
          href: `https://github.com/${site.organizationName}/${site.projectName}`,
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Manual',
          items: [
            {label: 'Quickstart Tutorial', to: '/getting-started/quickstart-tutorial'},
            {label: 'Core Workflow', to: '/core-workflow/researcher-journey'},
            {label: 'Interpretability Primer', to: '/concepts/interpretability-primer'},
            {label: 'Troubleshooting', to: '/troubleshooting'},
          ],
        },
        {
          title: 'Reference',
          items: [
            {label: 'REST API', to: '/reference/api/overview'},
            {label: 'WebSocket Channels', to: '/reference/websocket-channels'},
            {label: 'Data Model', to: '/reference/data-model'},
          ],
        },
        {
          title: 'Resources',
          items: [
            {label: 'GitHub', href: `https://github.com/${site.organizationName}/${site.projectName}`},
            {label: 'Neuronpedia', href: 'https://neuronpedia.org'},
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Human in the Stream, LLC. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'typescript', 'nginx', 'yaml', 'docker'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

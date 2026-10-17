import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'miStudio Manual',
  tagline: 'MechInterp Studio — End-to-End Mechanistic Interpretability Platform',
  favicon: 'img/favicon.svg',

  // GitHub Pages deployment — published from the public hitsainet mirror via
  // .github/workflows/deploy-manual.yml (actions/deploy-pages), not `yarn deploy`.
  url: 'https://hitsainet.github.io',
  baseUrl: '/miStudio/',

  // GitHub Pages config
  organizationName: 'hitsainet',
  projectName: 'miStudio',
  trailingSlash: false,

  onBrokenLinks: 'throw',

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
          editUrl: 'https://github.com/hitsainet/miStudio/tree/main/manual/',
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
      title: 'miStudio Manual',
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
          href: 'https://github.com/hitsainet/miStudio',
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
            {label: 'GitHub', href: 'https://github.com/hitsainet/miStudio'},
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

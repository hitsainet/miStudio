# miStudio Manual

The miStudio user manual — a [Docusaurus 3](https://docusaurus.io/) site published at
**<https://hitsainet.github.io/miStudio/>**.

## Editing content

All pages live under `docs/` as Markdown. Screenshots go in `static/img/` (JPG,
`snake_case` names matching the existing files). The sidebar is defined in
`sidebars.ts`; site config in `docusaurus.config.ts`.

## Local development

Requires Node.js ≥ 20.

```bash
npm install
npm run start     # dev server with live reload at http://localhost:3000/miStudio/
```

## Build (also the link check)

```bash
npm run build
```

`onBrokenLinks: 'throw'` means a green build guarantees no broken internal links or
anchors. Always build before pushing.

## Deployment (automatic — do not run `npm run deploy`)

Pushing to `main` with changes under `manual/**` triggers the pipeline:

1. The sync workflow mirrors this repo to the public `hitsainet/miStudio` repo.
2. `.github/workflows/deploy-manual.yml` in the mirror builds the site and deploys it
   to GitHub Pages.
3. The live site updates within a few minutes.

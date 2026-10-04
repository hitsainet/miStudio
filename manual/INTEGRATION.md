# miStudio Manual — Docusaurus Integration Guide

> **⚠️ Historical document.** The manual is now published to **GitHub Pages** at
> <https://hitsainet.github.io/miStudio/> — the nginx `/manual/` embedding described
> below was the original integration plan and is kept for reference only.
>
> **How publishing actually works today:**
> 1. Edit files under `manual/docs/` and push to `main`.
> 2. The sync workflow mirrors the repo to the public `hitsainet/miStudio` mirror.
> 3. `.github/workflows/deploy-manual.yml` (in the mirror) runs on any `manual/**`
>    change: `npm ci` → `npm run build` → `actions/deploy-pages`.
> 4. The site updates at hitsainet.github.io/miStudio a few minutes later.
>
> **Verify locally before pushing:** `cd manual && npm run build` — broken links
> fail the build (`onBrokenLinks: 'throw'`), so a green build is the link check.
> Config values quoted below (e.g. `mistudio.hitsai.local`, `baseUrl: '/manual/'`)
> are **stale**; `docusaurus.config.ts` is the source of truth.

This directory contains a complete Docusaurus 3.x static documentation site for the miStudio user manual. The sections below describe the original (now unused) nginx integration.

---

## 1. Place the Directory

Copy/move the entire `manual/` directory into the **root** of the miStudio application repository:

```
miStudio/
├── manual/              <-- this directory
│   ├── docs/            <-- markdown content (the actual manual pages)
│   ├── src/             <-- Docusaurus theme/CSS customizations
│   ├── static/          <-- static assets (images, favicon)
│   ├── docusaurus.config.ts
│   ├── sidebars.ts
│   ├── package.json
│   └── tsconfig.json
├── backend/
├── frontend/
├── docker-compose.yml
└── ...
```

## 2. Update .gitignore

Add these lines to the **repo root** `.gitignore`:

```gitignore
# Docusaurus manual
manual/node_modules/
manual/build/
manual/.docusaurus/
```

## 3. Install Dependencies

Requires **Node.js >= 20**.

```bash
cd manual
npm install
```

## 4. Build the Static Site

```bash
cd manual
npx docusaurus build
```

This produces `manual/build/` containing a fully self-contained static site (HTML, CSS, JS). The site is configured to be served at the `/manual/` URL path.

## 5. Serve via Nginx (Production)

The built static files need to be served by the existing nginx reverse proxy. Add a location block to the nginx configuration:

```nginx
# miStudio User Manual (static Docusaurus site)
location /manual/ {
    alias /path/to/manual/build/;
    index index.html;
    try_files $uri $uri/ $uri.html /manual/index.html;
}
```

Replace `/path/to/manual/build/` with the actual path inside the container or host where the built files reside.

### Option A: Serve from the existing frontend container

If the React frontend is served by nginx, mount the built manual into the same container:

```yaml
# In docker-compose.yml, add to the frontend service volumes:
volumes:
  - ./manual/build:/usr/share/nginx/html/manual:ro
```

Then add the nginx location block above to the frontend's nginx config.

### Option B: Add a dedicated manual build step to Docker

Add a build stage to the frontend Dockerfile (or create a separate one):

```dockerfile
# Manual build stage
FROM node:20-alpine AS manual-builder
WORKDIR /app/manual
COPY manual/package*.json ./
RUN npm ci
COPY manual/ ./
RUN npx docusaurus build

# Then in the final nginx stage, copy the built files:
COPY --from=manual-builder /app/manual/build /usr/share/nginx/html/manual
```

## 6. Configuration Reference

Key settings in `docusaurus.config.ts`:

| Setting | Current Value | Purpose |
|---------|--------------|---------|
| `url` | `https://mistudio.hitsai.local` | Production URL of the miStudio instance |
| `baseUrl` | `/manual/` | URL path prefix — all manual pages live under `/manual/` |
| `docs.routeBasePath` | `/` | Docs are the root content (no `/docs/` prefix within `/manual/`) |
| `blog` | `false` | Blog feature disabled |

**URL structure in production:** `https://mistudio.hitsai.local/manual/` → manual homepage.

If the production URL or path changes, update `url` and `baseUrl` in `docusaurus.config.ts` and rebuild.

## 7. Local Development / Preview

```bash
cd manual
npx docusaurus start
```

Opens a hot-reloading dev server at `http://localhost:3000/manual/`. Edits to markdown files in `docs/` are reflected instantly.

## 8. Editing the Manual

All manual content is in `manual/docs/`:

```
docs/
├── intro.md                          # Landing page (served at /manual/)
├── getting-started/
│   ├── introduction.md               # What miStudio is
│   ├── installation.md               # Hardware/software requirements
│   └── dashboard.md                  # UI navigation guide
├── core-workflow/
│   ├── researcher-journey.md         # The 5-stage pipeline overview
│   ├── data-model-management.md      # Models, quantization, hook points
│   ├── sae-training.md               # SAE frameworks & hyperparameters
│   ├── feature-extraction.md         # Extraction config & token filtering
│   ├── auto-labeling.md              # LLM-powered labeling system
│   └── steering.md                   # Feature intervention & matrix testing
├── advanced/
│   ├── templates.md                  # Template ecosystem
│   ├── external-saes.md              # HuggingFace SAE integration
│   ├── multi-gpu.md                  # Multi-GPU partitioning
│   ├── exporting.md                  # Export & Neuronpedia integration
│   └── multi-dataset.md              # Multi-dataset training
└── troubleshooting.md                # Common issues & key formulas
```

- Each file uses standard Markdown with Docusaurus [admonitions](https://docusaurus.io/docs/markdown-features/admonitions) (`:::info`, `:::tip`, `:::warning`, `:::danger`)
- Sidebar order is controlled by `sidebar_position` in each file's frontmatter
- Sidebar structure is defined in `sidebars.ts`
- **MDX note:** Bare `<` characters in text (not in code blocks/backticks) must be escaped as `&lt;` — Docusaurus uses MDX which interprets `<` as JSX

## 9. Adding a Link from the Main Application

Add a link in the miStudio React frontend sidebar or help menu that points to `/manual/`:

```tsx
<a href="/manual/" target="_blank" rel="noopener noreferrer">
  User Manual
</a>
```

This opens the manual in a new tab while keeping the user's miStudio session active.

## 10. CI/CD Considerations

If using GitHub Actions or similar:

```yaml
# Example GitHub Actions step
- name: Build manual
  working-directory: manual
  run: |
    npm ci
    npx docusaurus build
```

The `manual/build/` output can then be included in the Docker image or deployed to static hosting.

## Dependencies

- **Node.js >= 20** (build-time only)
- **No runtime dependencies** — the output is pure static HTML/CSS/JS
- The `manual/package.json` contains all Docusaurus dependencies; they are isolated from the main application

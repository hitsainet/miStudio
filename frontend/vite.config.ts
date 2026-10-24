import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { readFileSync } from 'fs';

// The app version shown in the sidebar foot (src/config/version.ts): package.json is the one
// source of truth, read here rather than written into a component.
const pkg = JSON.parse(readFileSync(path.resolve(import.meta.dirname, 'package.json'), 'utf8')) as { version: string };

// A top-level .ts file of any `manual/` directory (tests read them as `?raw` text).
const MANUAL_TOP_LEVEL_TS = /[\\/]manual[\\/][^\\/]+\.ts$/;

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  // In tests, plugin-react must also leave the manual's top-level .ts files alone: since Vite 8
  // its JSX-refresh filter (which matches `.ts` WITH a query) sends a `.ts?raw` module through
  // Oxc even when `oxc.exclude` names it. See the `oxc` block below.
  plugins: [react(mode === 'test' ? { exclude: [/\/node_modules\//, MANUAL_TOP_LEVEL_TS] } : {})],
  define: { __APP_VERSION__: JSON.stringify(pkg.version) },
  resolve: {
    alias: {
      '@': path.resolve(import.meta.dirname, './src'),
    },
  },
  server: {
    // Tests only (vitest runs in mode "test"): manualLinks.test.tsx and helpPlainLanguage.test.ts
    // read the manual's Markdown, and version.test.ts the repository's VERSION file, through
    // Vite's `?raw` imports, which Vite denies outside the frontend unless allowed (a single file
    // cannot be allowed in this Vite version, so the repository root is). The dev server keeps
    // Vite's default and never serves anything from the rest of the repository.
    fs: mode === 'test'
      ? { allow: [path.resolve(import.meta.dirname, '..')] }
      : undefined,
    port: 3000,
    host: '0.0.0.0',
    // Allow all hosts for development - required for nginx proxy and open-source portability
    // Production builds serve static files so this only affects the dev server
    allowedHosts: ['mistudio.hitsai.local', 'dev-mistudio.hitsai.local', 'k8s-mistudio.hitsai.local', 'k8s-mistudio.hitsai.net', 'localhost', 'host.docker.internal'],
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true,
      },
    },
  },
  // Tests only: manualLinks.test.tsx reads manual/docusaurus.config.ts and manual/sidebars.ts
  // as text (`?raw`), but Vite still runs its TypeScript transform (Oxc since Vite 8, esbuild
  // before) on a `.ts?raw` module and resolves the nearest tsconfig.json for it.
  // manual/tsconfig.json extends "@docusaurus/tsconfig", which GitHub's Frontend CI never
  // installs (it runs `npm ci` in frontend/ only), so that test file failed to load there. Keep
  // the transform off the top-level .ts files of a manual/ directory in tests; /\.js$/ is Vite's
  // own default exclude, kept. Guarded by src/test/rawTsOutsideTheApp.test.ts.
  ...(mode === 'test'
    ? { oxc: { exclude: [/\.js$/, MANUAL_TOP_LEVEL_TS] } }
    : {}),
  build: {
    outDir: 'dist',
    // MIS-E2E-020. Sourcemaps were unconditionally on, so every production
    // build shipped 12 `.map` files reconstructing the full original source —
    // including comments — to anyone who opened devtools. Keep them for a dev
    // build, where they are the point, and drop them from a production one.
    sourcemap: mode !== 'production',
    chunkSizeWarningLimit: 600,
    rolldownOptions: {
      // 364 `console.log`/`debug`/`info` calls reach the browser otherwise, some of them
      // printing request payloads. Marking them pure lets the bundler drop them from a
      // production build. `console.warn` and `console.error` are deliberately absent: those are
      // the ones a user is ever asked to read back. (MIS-E2E-020)
      //
      // VITE 8 (2026-10-21): this was `esbuild.pure`. Vite 8 builds with Rolldown and Oxc and
      // IGNORES the `esbuild` block in a build (it only warns), so the calls would have shipped
      // again with every config-reading test still green. src/config/buildHardening.test.ts now
      // builds a fixture through this config and reads the output.
      treeshake: {
        manualPureFunctions: mode === 'production'
          ? ['console.log', 'console.debug', 'console.info', 'console.trace']
          : [],
      },
      output: {
        // VITE 8: Rollup's `manualChunks` is deprecated in Rolldown and, run as-is, put recharts
        // inside feature-jlens (a group pulls in its dependencies) and dropped vendor-charts,
        // -icons, -state and -socket. Two groups keep the old layout: every node_modules module
        // is claimed by the higher-priority vendor group first, so a feature group can only
        // take application code.
        codeSplitting: {
          groups: [
            {
              debugName: 'vendor',
              priority: 20,
              test: /node_modules/,
              name: (id: string) => {
                // React and zustand must be in the same chunk to avoid useState undefined error
                if (id.includes('react') || id.includes('react-dom') || id.includes('zustand') || id.includes('use-sync-external-store')) {
                  return 'vendor-react';
                }
                if (id.includes('recharts')) {
                  return 'vendor-charts';
                }
                if (id.includes('lucide-react')) {
                  return 'vendor-icons';
                }
                if (id.includes('axios')) {
                  return 'vendor-state';
                }
                if (id.includes('socket.io-client')) {
                  return 'vendor-socket';
                }
                // Other node_modules go into vendor chunk
                return 'vendor';
              },
            },
            {
              // Feature-based code splitting
              debugName: 'features',
              priority: 10,
              name: (id: string) => {
                if (id.includes('/components/datasets/') || id.includes('/stores/datasetsStore')) {
                  return 'feature-datasets';
                }
                if (id.includes('/components/models/') || id.includes('/stores/modelsStore')) {
                  return 'feature-models';
                }
                if (id.includes('/components/training/') || id.includes('/stores/trainingsStore')) {
                  return 'feature-training';
                }
                if (id.includes('/components/extractionTemplates/') || id.includes('/stores/extractionTemplatesStore')) {
                  return 'feature-templates';
                }
                if (id.includes('/components/SystemMonitor/')) {
                  return 'feature-monitor';
                }
                // These rules match SUB-COMPONENT directories, never
                // /components/panels/ — which is why the J-Lens trajectory chart
                // (recharts) lives under /components/jlens/ rather than in the panel.
                if (id.includes('/components/jlens/') || id.includes('/stores/jlensStore')) {
                  return 'feature-jlens';
                }
                return null;
              },
            },
          ],
        },
      },
    },
  },
  test: {
    globals: true,
    environment: 'jsdom',
    setupFiles: './src/test/setup.ts',
    css: true,
  },
}));

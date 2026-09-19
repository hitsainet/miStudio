import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
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
  // 364 `console.log`/`debug`/`info` calls reach the browser otherwise, some
  // of them printing request payloads. Marking them pure lets the default
  // esbuild minifier drop them from a production build. `console.warn` and
  // `console.error` are deliberately absent: those are the ones a user is ever
  // asked to read back. (MIS-E2E-020)
  esbuild: {
    pure: mode === 'production'
      ? ['console.log', 'console.debug', 'console.info', 'console.trace']
      : [],
  },
  build: {
    outDir: 'dist',
    // MIS-E2E-020. Sourcemaps were unconditionally on, so every production
    // build shipped 12 `.map` files reconstructing the full original source —
    // including comments — to anyone who opened devtools. Keep them for a dev
    // build, where they are the point, and drop them from a production one.
    sourcemap: mode !== 'production',
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor chunks - split large dependencies
          if (id.includes('node_modules')) {
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
          }

          // Feature-based code splitting
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

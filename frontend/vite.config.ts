import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    host: '0.0.0.0',
    // Allow all hosts for development access from 192.168.224.0/24 network
    allowedHosts: 'all',
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
  build: {
    outDir: 'dist',
    sourcemap: true,
    chunkSizeWarningLimit: 600,
    rollupOptions: {
      output: {
        manualChunks: (id) => {
          // Vendor chunks - split large dependencies
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('react-dom')) {
              return 'vendor-react';
            }
            if (id.includes('recharts')) {
              return 'vendor-charts';
            }
            if (id.includes('lucide-react')) {
              return 'vendor-icons';
            }
            if (id.includes('zustand') || id.includes('axios')) {
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
});

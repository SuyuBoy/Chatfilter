import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  base: './',
  build: {
    target: 'esnext',
    outDir: 'dist',
    rollupOptions: {
      input: {
        main: 'index.html',
        dashboard: 'dashboard.html',
      },
    },
  },
  plugins: [
    react(),
    viteStaticCopy({
      targets: [{
        src: 'node_modules/onnxruntime-web/dist/*.wasm',
        dest: 'assets',
      }],
    }),
  ],
  optimizeDeps: {
    entries: ['index.html', 'dashboard.html'],
  },
});

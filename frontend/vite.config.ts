import { defineConfig } from 'vite';
import { viteStaticCopy } from 'vite-plugin-static-copy';

export default defineConfig({
  base: './',
  build: {
    target: 'esnext',
    outDir: 'dist',
  },
  optimizeDeps: {
    exclude: ['@xenova/transformers'],
  },
  plugins: [
    viteStaticCopy({
      targets: [
        {
          src: 'node_modules/onnxruntime-web/dist/*.wasm',
          dest: 'assets',
        },
      ],
    }),
  ],
});

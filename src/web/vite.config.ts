import { defineConfig } from 'vite'; // v4.1.0
import react from '@vitejs/plugin-react'; // v3.1.0
import { resolve } from 'path';

// Vite configuration for TALD UNIA web frontend
export default defineConfig({
  // Development server configuration
  server: {
    port: 3000,
    host: true, // Listen on all local IPs
    open: true, // Auto-open browser on start
    cors: true, // Enable CORS for development
    hmr: {
      overlay: true, // Show errors as overlay
    },
  },

  // Build configuration
  build: {
    outDir: 'dist',
    sourcemap: true, // Enable source maps for debugging
    minify: 'terser', // Use Terser for optimal minification
    target: 'esnext', // Target modern browsers
    chunkSizeWarningLimit: 2000, // Increase chunk size limit for LiDAR visualization
    rollupOptions: {
      output: {
        manualChunks: {
          // Split vendor chunks for optimal caching
          'react-vendor': ['react', 'react-dom'],
          'mui-vendor': ['@mui/material'],
          'three-vendor': ['three'],
          'rxjs-vendor': ['rxjs'],
        },
      },
    },
  },

  // Path resolution configuration
  resolve: {
    alias: {
      '@': resolve(__dirname, 'src'),
      '@components': resolve(__dirname, 'src/components'),
      '@hooks': resolve(__dirname, 'src/hooks'),
      '@utils': resolve(__dirname, 'src/utils'),
      '@store': resolve(__dirname, 'src/store'),
      '@api': resolve(__dirname, 'src/api'),
      '@types': resolve(__dirname, 'src/types'),
      '@assets': resolve(__dirname, 'src/assets'),
      '@styles': resolve(__dirname, 'src/styles'),
      '@config': resolve(__dirname, 'src/config'),
    },
  },

  // Plugin configuration
  plugins: [
    react({
      // Enable fast refresh for React components
      fastRefresh: true,
      // Include runtime JSX transforms
      jsxRuntime: 'automatic',
      // Enable babel plugins for optimal development experience
      babel: {
        plugins: [
          ['@babel/plugin-transform-runtime'],
        ],
      },
    }),
  ],

  // Dependency optimization
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      '@mui/material',
      'three',
      'rxjs',
    ],
    // Force dependency pre-bundling
    force: true,
  },

  // CSS configuration
  css: {
    modules: {
      localsConvention: 'camelCase',
    },
    preprocessorOptions: {
      scss: {
        additionalData: '@import "@styles/variables";',
      },
    },
  },

  // Performance optimizations
  esbuild: {
    logOverride: { 'this-is-undefined-in-esm': 'silent' },
    target: 'esnext',
    treeShaking: true,
  },

  // Preview configuration
  preview: {
    port: 3000,
    host: true,
  },
});
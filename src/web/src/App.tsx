import React, { useEffect } from 'react';
import { Provider } from 'react-redux'; // v8.1.0
import { ThemeProvider, CssBaseline } from '@mui/material'; // v5.13.0
import { BrowserRouter, Routes, Route } from 'react-router-dom'; // v6.11.2
import * as Sentry from '@sentry/react'; // v7.0.0
import { Analytics } from '@segment/analytics-next'; // v1.51.0
import { ErrorBoundary } from 'react-error-boundary'; // v4.0.11

import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import { store } from './store';
import { lightTheme } from './styles/theme';

// Constants
const APP_TITLE = 'TALD UNIA';
const DEFAULT_ROUTE = '/';
const ERROR_BOUNDARY_FALLBACK = '/error';
const SENTRY_DSN = process.env.REACT_APP_SENTRY_DSN;
const ANALYTICS_KEY = process.env.REACT_APP_ANALYTICS_KEY;
const API_RATE_LIMIT = '100';
const CACHE_DURATION = '3600';

// Initialize error tracking
Sentry.init({
  dsn: SENTRY_DSN,
  tracesSampleRate: 1.0,
  integrations: [
    new Sentry.BrowserTracing({
      tracingOrigins: ['localhost', 'tald.unia'],
    }),
  ],
});

// Initialize analytics
const analytics = Analytics.load({ writeKey: ANALYTICS_KEY });

// Error Boundary Fallback Component
const ErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = ({
  error,
  resetErrorBoundary
}) => (
  <div role="alert">
    <h2>Something went wrong:</h2>
    <pre>{error.message}</pre>
    <button onClick={resetErrorBoundary}>Try again</button>
  </div>
);

// Service Worker Registration
const registerServiceWorker = async (): Promise<void> => {
  if ('serviceWorker' in navigator) {
    try {
      const registration = await navigator.serviceWorker.register('/service-worker.js');
      console.log('ServiceWorker registration successful:', registration);
    } catch (error) {
      console.error('ServiceWorker registration failed:', error);
    }
  }
};

// Main App Component
const App: React.FC = () => {
  useEffect(() => {
    // Register service worker
    registerServiceWorker();

    // Set content security policy
    const meta = document.createElement('meta');
    meta.httpEquiv = 'Content-Security-Policy';
    meta.content = `
      default-src 'self';
      script-src 'self' 'unsafe-inline';
      style-src 'self' 'unsafe-inline';
      img-src 'self' data: blob:;
      connect-src 'self' wss://*.tald.unia;
      worker-src 'self' blob:;
    `;
    document.head.appendChild(meta);

    // Configure rate limiting
    document.head.appendChild(
      Object.assign(document.createElement('meta'), {
        name: 'rate-limit',
        content: API_RATE_LIMIT
      })
    );

    // Configure caching
    document.head.appendChild(
      Object.assign(document.createElement('meta'), {
        name: 'cache-control',
        content: `max-age=${CACHE_DURATION}`
      })
    );

    // Track page load
    analytics.track('App Initialized', {
      timestamp: Date.now(),
      userAgent: navigator.userAgent
    });
  }, []);

  return (
    <ErrorBoundary
      FallbackComponent={ErrorFallback}
      onReset={() => window.location.href = ERROR_BOUNDARY_FALLBACK}
      onError={(error) => {
        Sentry.captureException(error);
        analytics.track('Error Occurred', {
          error: error.message,
          timestamp: Date.now()
        });
      }}
    >
      <Provider store={store}>
        <ThemeProvider theme={lightTheme}>
          <CssBaseline />
          <BrowserRouter>
            <MainLayout>
              <Routes>
                <Route path={DEFAULT_ROUTE} element={<Dashboard />} />
              </Routes>
            </MainLayout>
          </BrowserRouter>
        </ThemeProvider>
      </Provider>
    </ErrorBoundary>
  );
};

// Performance monitoring decorator
const withPerformanceMonitoring = Sentry.withProfiler(App);

// Error boundary decorator
const withErrorBoundary = Sentry.withErrorBoundary(withPerformanceMonitoring, {
  fallback: ErrorFallback
});

export default withErrorBoundary;
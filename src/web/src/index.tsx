import React from 'react';
import { createRoot } from 'react-dom/client';
import { Provider } from 'react-redux';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import * as Sentry from '@sentry/react';
import { ErrorBoundary, PerformanceMonitor } from '@sentry/react';

import App from './App';
import { store } from './store';
import { lightTheme } from './styles/theme';

// Constants based on technical specifications
const ROOT_ELEMENT_ID = 'root';
const PERFORMANCE_THRESHOLD = 16.67; // Target 60 FPS
const IS_DEVELOPMENT = process.env.NODE_ENV === 'development';
const SENTRY_DSN = process.env.REACT_APP_SENTRY_DSN;

// Initialize error tracking and performance monitoring
Sentry.init({
    dsn: SENTRY_DSN,
    environment: process.env.NODE_ENV,
    tracesSampleRate: 1.0,
    integrations: [
        new Sentry.BrowserTracing({
            tracingOrigins: ['localhost', 'tald.unia'],
        }),
    ],
    beforeSend(event) {
        if (IS_DEVELOPMENT) {
            console.error('Sentry Event:', event);
        }
        return event;
    },
    // Performance monitoring configuration
    performanceOptions: {
        captureInteractions: true,
        timeoutWarningLimit: PERFORMANCE_THRESHOLD,
        longtaskWarningLimit: PERFORMANCE_THRESHOLD,
    }
});

// Error Boundary Fallback Component
const ErrorFallback: React.FC<{ error: Error; resetErrorBoundary: () => void }> = ({
    error,
    resetErrorBoundary
}) => (
    <div role="alert" style={{ padding: '20px', textAlign: 'center' }}>
        <h2>Application Error</h2>
        <pre style={{ margin: '10px 0' }}>{error.message}</pre>
        <button 
            onClick={resetErrorBoundary}
            style={{ padding: '8px 16px', cursor: 'pointer' }}
        >
            Try Again
        </button>
    </div>
);

// Initialize and render the application
const renderApp = (): void => {
    const rootElement = document.getElementById(ROOT_ELEMENT_ID);
    if (!rootElement) {
        throw new Error(`Element with id '${ROOT_ELEMENT_ID}' not found`);
    }

    const root = createRoot(rootElement);

    root.render(
        <React.StrictMode>
            <ErrorBoundary
                fallback={ErrorFallback}
                onError={(error) => {
                    Sentry.captureException(error);
                    console.error('Application Error:', error);
                }}
            >
                <Provider store={store}>
                    <ThemeProvider theme={lightTheme}>
                        <CssBaseline />
                        <PerformanceMonitor
                            captureInteractions={true}
                            threshold={PERFORMANCE_THRESHOLD}
                            onError={(error) => {
                                console.warn('Performance degradation:', error);
                            }}
                        >
                            <App />
                        </PerformanceMonitor>
                    </ThemeProvider>
                </Provider>
            </ErrorBoundary>
        </React.StrictMode>
    );
};

// Register service worker for PWA support
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/service-worker.js')
            .then(registration => {
                console.log('ServiceWorker registration successful:', registration);
            })
            .catch(error => {
                console.error('ServiceWorker registration failed:', error);
            });
    });
}

// Initialize application
renderApp();
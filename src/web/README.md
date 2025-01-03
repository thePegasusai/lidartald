# TALD UNIA Web Frontend

Enterprise-grade React application providing the user interface for the TALD UNIA LiDAR-enabled gaming platform.

## Project Overview

TALD UNIA's web frontend implements a high-performance, Material Design 3.0 compliant interface for:
- Real-time LiDAR point cloud visualization
- Fleet ecosystem management 
- Social gaming interactions
- Environment mapping and game integration

### Key Features
- 60 FPS UI responsiveness
- Real-time WebSocket communication
- GPU-accelerated 3D rendering
- Responsive design for portrait/landscape modes
- Internationalization support for 12 languages

## Prerequisites

- Node.js >= 18.0.0
- npm >= 8.0.0
- VSCode with recommended extensions:
  - ESLint
  - Prettier
  - TypeScript and JavaScript Language Features
  - Material Icon Theme
  - Jest Runner

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd src/web
```

2. Install dependencies:
```bash
npm install
```

3. Create environment configuration:
```bash
cp .env.example .env.local
```

## Development

### Available Scripts

```bash
# Start development server with hot reload
npm run dev

# Build production bundle
npm run build

# Run unit tests
npm run test

# Run end-to-end tests
npm run test:e2e

# Run ESLint checks
npm run lint
```

### Code Organization

```
src/
├── components/       # Reusable UI components
├── features/        # Feature-specific modules
├── hooks/           # Custom React hooks
├── services/        # API and external services
├── store/           # Redux state management
├── styles/          # Global styles and themes
├── types/           # TypeScript definitions
└── utils/           # Utility functions
```

### Technology Stack

#### Core
- React v18.2.0 - UI framework
- TypeScript v4.9.5 - Type safety
- Material UI v5.0.0 - Component library

#### State Management
- Redux Toolkit v1.9.0 - Application state
- RxJS v7.8.0 - Reactive programming

#### Visualization
- Three.js v0.150.0 - 3D rendering

## Architecture

### Component Architecture

The application follows atomic design principles:
- Atoms: Basic UI elements
- Molecules: Composite components
- Organisms: Feature sections
- Templates: Page layouts
- Pages: Route components

### State Management

Redux store organization:
```typescript
interface RootState {
  lidar: LidarState;       // LiDAR processing state
  fleet: FleetState;       // Fleet management state
  game: GameState;         // Game session state
  user: UserState;         // User profile state
  ui: UIState;             // UI control state
}
```

### Performance Optimization

- Code splitting via React.lazy()
- Memoization of expensive computations
- WebGL rendering optimization
- Asset preloading
- Service Worker caching

## Testing

### Unit Testing
- Jest for component and utility testing
- React Testing Library for component interaction
- 80% minimum coverage requirement

### E2E Testing
- Cypress for critical user flows
- Automated performance benchmarking
- Cross-browser compatibility testing

## Deployment

### Build Process
1. TypeScript compilation
2. Asset optimization
3. Bundle size analysis
4. Environment configuration
5. Performance benchmarking

### Performance Requirements
- Initial load: < 2 seconds
- UI responsiveness: 60 FPS
- Bundle size: < 1000KB initial

## Contributing

### Code Style
- ESLint configuration
- Prettier formatting
- TypeScript strict mode
- Material Design guidelines

### Pull Request Process
1. Feature branch creation
2. Implementation with tests
3. Documentation updates
4. Code review
5. CI/CD pipeline validation

### Review Guidelines
- Performance impact assessment
- Accessibility compliance
- Security considerations
- Code quality metrics
- Test coverage requirements

## License

Copyright © 2023 TALD UNIA. All rights reserved.
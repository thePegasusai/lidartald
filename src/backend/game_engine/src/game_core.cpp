/**
 * @file game_core.cpp
 * @brief Implementation of the core game engine system for TALD UNIA
 * @version 1.0.0
 */

#include "game_core.hpp"
#include <chrono>
#include <spdlog/spdlog.h>  // v1.11.0
#include <glm/glm.hpp>      // v0.9.9.8
#include <yojimbo.h>        // v1.0.0

using namespace tald::game;
using namespace std::chrono;

GameEngine::GameEngine(const GameEngineConfig& config)
    : m_deltaTime(0.0)
    , m_isRunning(false)
    , m_fleetSyncEnabled(false)
    , m_config(config) {
    
    spdlog::info("Initializing TALD UNIA Game Engine v{}", ENGINE_VERSION);
    
    try {
        // Initialize core components with enhanced error handling
        m_physics = std::make_unique<PhysicsEngine>(config.physicsConfig);
        if (!m_physics) {
            throw std::runtime_error("Failed to initialize physics engine");
        }

        m_renderer = std::make_unique<Renderer>(config.renderConfig);
        if (!m_renderer) {
            throw std::runtime_error("Failed to initialize renderer");
        }

        m_stateManager = std::make_unique<GameStateManager>(config.stateConfig);
        if (!m_stateManager) {
            throw std::runtime_error("Failed to initialize state manager");
        }

        // Initialize performance monitoring
        m_metrics = PerformanceMetrics{
            .frameTime = 0.0,
            .physicsTime = 0.0,
            .renderTime = 0.0,
            .networkLatency = 0.0,
            .frameCount = 0
        };

        // Initialize fleet networking if enabled
        if (!config.fleetId.empty()) {
            m_fleetSyncEnabled = true;
            spdlog::info("Fleet synchronization enabled with ID: {}", config.fleetId);
        }

        // Initialize vector clock for CRDT
        m_vectorClock = boost::crdt::vclock();
        m_mergePolicy = config.crdtConfig.mergePolicy;

    } catch (const std::exception& e) {
        spdlog::error("Engine initialization failed: {}", e.what());
        throw;
    }
}

bool GameEngine::initialize() {
    try {
        // Validate system requirements
        if (!validateSystemRequirements()) {
            spdlog::error("System requirements validation failed");
            return false;
        }

        // Initialize physics system with thread safety
        if (!m_physics->initialize()) {
            spdlog::error("Physics initialization failed");
            return false;
        }

        // Initialize renderer with GPU optimization
        if (!m_renderer->initialize()) {
            spdlog::error("Renderer initialization failed");
            return false;
        }

        // Initialize state manager with CRDT support
        if (!m_stateManager->initialize()) {
            spdlog::error("State manager initialization failed");
            return false;
        }

        // Setup fleet synchronization if enabled
        if (m_fleetSyncEnabled) {
            if (!initializeFleetSync()) {
                spdlog::error("Fleet synchronization initialization failed");
                return false;
            }
        }

        // Initialize performance monitoring
        initializePerformanceMonitoring();

        m_isRunning = true;
        m_lastFrameTime = high_resolution_clock::now();
        spdlog::info("Game engine initialization complete");
        return true;

    } catch (const std::exception& e) {
        spdlog::error("Engine initialization failed: {}", e.what());
        return false;
    }
}

void GameEngine::update() {
    if (!m_isRunning) return;

    auto frameStart = high_resolution_clock::now();

    try {
        // Calculate frame timing
        auto currentTime = high_resolution_clock::now();
        m_deltaTime = duration<double>(currentTime - m_lastFrameTime).count();
        m_lastFrameTime = currentTime;

        // Update performance metrics
        m_metrics.frameTime = m_deltaTime;
        m_metrics.frameCount++;

        // Update vector clock for CRDT
        m_vectorClock.increment();

        // Update physics with timing
        auto physicsStart = high_resolution_clock::now();
        m_physics->update(m_deltaTime);
        m_metrics.physicsTime = duration<double>(high_resolution_clock::now() - physicsStart).count();

        // Update game state
        auto stateStart = high_resolution_clock::now();
        m_stateManager->update(m_deltaTime);

        // Perform CRDT state reconciliation if fleet sync is enabled
        if (m_fleetSyncEnabled) {
            syncFleetState();
        }
        m_metrics.networkLatency = duration<double>(high_resolution_clock::now() - stateStart).count();

        // Render frame with timing
        auto renderStart = high_resolution_clock::now();
        m_renderer->render(m_stateManager->getCurrentState(), m_physics->getDebugData());
        m_metrics.renderTime = duration<double>(high_resolution_clock::now() - renderStart).count();

        // Frame rate control
        limitFrameRate();

    } catch (const std::exception& e) {
        spdlog::error("Update cycle failed: {}", e.what());
    }
}

void GameEngine::shutdown() {
    spdlog::info("Shutting down game engine");
    
    m_isRunning = false;

    try {
        // Finalize CRDT state if fleet sync was enabled
        if (m_fleetSyncEnabled) {
            finalizeFleetSync();
        }

        // Cleanup core components in reverse initialization order
        if (m_renderer) {
            m_renderer->cleanup();
        }
        
        if (m_physics) {
            m_physics->cleanup();
        }
        
        if (m_stateManager) {
            m_stateManager->cleanup();
        }

        // Log final performance metrics
        logPerformanceMetrics();

        spdlog::info("Engine shutdown complete");

    } catch (const std::exception& e) {
        spdlog::error("Shutdown error: {}", e.what());
    }
}

// Private helper methods

bool GameEngine::validateSystemRequirements() {
    // Validate GPU capabilities
    if (!m_renderer->checkGPUSupport()) {
        spdlog::error("Required GPU features not supported");
        return false;
    }

    // Validate memory requirements
    if (!checkMemoryRequirements()) {
        spdlog::error("Insufficient system memory");
        return false;
    }

    // Validate thread support
    if (!checkThreadSupport()) {
        spdlog::error("Required thread support not available");
        return false;
    }

    return true;
}

bool GameEngine::initializeFleetSync() {
    try {
        m_stateManager->enableFleetSync(m_config.fleetId);
        m_stateManager->setMergePolicy(m_mergePolicy);
        return true;
    } catch (const std::exception& e) {
        spdlog::error("Fleet sync initialization failed: {}", e.what());
        return false;
    }
}

void GameEngine::syncFleetState() {
    try {
        m_stateManager->syncState();
        m_stateManager->mergeCRDT(m_mergePolicy);
    } catch (const std::exception& e) {
        spdlog::warn("Fleet state sync failed: {}", e.what());
    }
}

void GameEngine::finalizeFleetSync() {
    try {
        m_stateManager->finalizeState();
        m_stateManager->syncState();
    } catch (const std::exception& e) {
        spdlog::warn("Final state sync failed: {}", e.what());
    }
}

void GameEngine::limitFrameRate() {
    const double targetFrameTime = 1.0 / TARGET_FRAME_RATE;
    const auto frameEnd = high_resolution_clock::now();
    const double frameTime = duration<double>(frameEnd - m_lastFrameTime).count();
    
    if (frameTime < targetFrameTime) {
        std::this_thread::sleep_for(duration<double>(targetFrameTime - frameTime));
    }
}

void GameEngine::initializePerformanceMonitoring() {
    m_metrics = PerformanceMetrics{};
    spdlog::info("Performance monitoring initialized");
}

void GameEngine::logPerformanceMetrics() {
    if (m_metrics.frameCount > 0) {
        const double avgFrameTime = m_metrics.frameTime / m_metrics.frameCount;
        const double avgPhysicsTime = m_metrics.physicsTime / m_metrics.frameCount;
        const double avgRenderTime = m_metrics.renderTime / m_metrics.frameCount;
        const double avgNetworkLatency = m_metrics.networkLatency / m_metrics.frameCount;

        spdlog::info("Performance metrics:");
        spdlog::info("  Average frame time: {:.2f}ms", avgFrameTime * 1000.0);
        spdlog::info("  Average physics time: {:.2f}ms", avgPhysicsTime * 1000.0);
        spdlog::info("  Average render time: {:.2f}ms", avgRenderTime * 1000.0);
        spdlog::info("  Average network latency: {:.2f}ms", avgNetworkLatency * 1000.0);
        spdlog::info("  Total frames: {}", m_metrics.frameCount);
    }
}

bool GameEngine::checkMemoryRequirements() {
    // Implementation would check system memory availability
    return true;
}

bool GameEngine::checkThreadSupport() {
    // Implementation would verify thread support
    return true;
}
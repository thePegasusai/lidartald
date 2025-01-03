/**
 * @file game_core.hpp
 * @brief Core game engine system for TALD UNIA platform
 * 
 * Provides high-performance game engine functionality with integrated physics,
 * graphics rendering, and enhanced CRDT-based fleet synchronization.
 * 
 * @version 1.0.0
 * @license MIT
 */

#pragma once

// External dependencies
#include <glm/glm.hpp>           // v0.9.9.8
#include <spdlog/spdlog.h>       // v1.11.0

// Internal dependencies
#include "physics.hpp"
#include "renderer.hpp"
#include "state_manager.hpp"

// Standard library includes
#include <memory>
#include <atomic>
#include <chrono>
#include <string>

// Global constants
constexpr const char* ENGINE_VERSION = "1.0.0";
constexpr int TARGET_FRAME_RATE = 60;
constexpr int MAX_FLEET_SIZE = 32;
constexpr int CRDT_MERGE_INTERVAL_MS = 50;
constexpr int STATE_SYNC_TIMEOUT_MS = 100;

namespace tald {
namespace game {

/**
 * @class GameEngineConfig
 * @brief Enhanced configuration structure for game engine initialization with CRDT support
 */
class GameEngineConfig {
public:
    PhysicsConfig physicsConfig;
    RenderConfig renderConfig;
    GameStateConfig stateConfig;
    std::string fleetId;
    bool enableDebug;
    CRDTConfig crdtConfig;
    FleetNetworkConfig fleetConfig;
    VectorClockConfig vectorClockConfig;

    /**
     * @brief Default constructor initializing engine configuration
     */
    GameEngineConfig()
        : enableDebug(
#ifdef NDEBUG
            false
#else
            true
#endif
        ) {
        // Initialize fleet configuration
        fleetConfig.maxPeers = MAX_FLEET_SIZE;
        fleetConfig.syncInterval = CRDT_MERGE_INTERVAL_MS;
        fleetConfig.syncTimeout = STATE_SYNC_TIMEOUT_MS;

        // Initialize CRDT configuration
        crdtConfig.mergePolicy = boost::crdt::merge_policy::latest_wins;
        crdtConfig.pruneInterval = 300000; // 5 minutes
    }
};

/**
 * @class GameEngine
 * @brief Core game engine class integrating physics, rendering, and enhanced state management
 */
class GameEngine {
public:
    /**
     * @brief Initializes the game engine with provided configuration
     * @param config Engine configuration parameters
     * @throws std::runtime_error if initialization fails
     */
    explicit GameEngine(const GameEngineConfig& config)
        : m_deltaTime(0.0)
        , m_isRunning(false)
        , m_fleetSyncEnabled(false)
        , m_config(config) {
        
        spdlog::info("Initializing TALD UNIA Game Engine v{}", ENGINE_VERSION);
        
        // Initialize core components
        m_physics = std::make_unique<PhysicsEngine>(config.physicsConfig);
        m_renderer = std::make_unique<Renderer>(config.renderConfig);
        m_stateManager = std::make_unique<GameStateManager>(config.stateConfig);
        
        // Initialize vector clock for CRDT
        m_vectorClock = boost::crdt::vclock();
        m_mergePolicy = config.crdtConfig.mergePolicy;
    }

    // Disable copying
    GameEngine(const GameEngine&) = delete;
    GameEngine& operator=(const GameEngine&) = delete;

    /**
     * @brief Initializes all engine subsystems
     * @return Success status of initialization
     */
    bool initialize() {
        try {
            // Initialize physics system
            if (!m_physics->initialize()) {
                spdlog::error("Physics initialization failed");
                return false;
            }

            // Initialize renderer
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
            if (!m_config.fleetId.empty()) {
                m_fleetSyncEnabled = true;
                m_stateManager->enableFleetSync(m_config.fleetId);
            }

            m_isRunning = true;
            spdlog::info("Game engine initialization complete");
            return true;

        } catch (const std::exception& e) {
            spdlog::error("Engine initialization failed: {}", e.what());
            return false;
        }
    }

    /**
     * @brief Updates all engine components with CRDT synchronization
     */
    void update() {
        if (!m_isRunning) return;

        // Calculate frame timing
        auto currentTime = std::chrono::high_resolution_clock::now();
        m_deltaTime = std::chrono::duration<double>(
            currentTime - m_lastFrameTime).count();
        m_lastFrameTime = currentTime;

        try {
            // Update vector clock
            m_vectorClock.increment();

            // Update physics simulation
            m_physics->update(m_deltaTime);

            // Update game state
            m_stateManager->update(m_deltaTime);

            // Perform CRDT state reconciliation if fleet sync is enabled
            if (m_fleetSyncEnabled) {
                m_stateManager->syncState();
                m_stateManager->mergeCRDT(m_mergePolicy);
            }

            // Render frame
            m_renderer->render(m_stateManager->getCurrentState(), 
                             m_physics->getDebugData());

        } catch (const std::exception& e) {
            spdlog::error("Update cycle failed: {}", e.what());
        }
    }

    /**
     * @brief Performs clean shutdown of engine components
     */
    void shutdown() {
        spdlog::info("Shutting down game engine");
        
        m_isRunning = false;

        // Finalize CRDT state if fleet sync was enabled
        if (m_fleetSyncEnabled) {
            try {
                m_stateManager->finalizeState();
                m_stateManager->syncState();
            } catch (const std::exception& e) {
                spdlog::warn("Final state sync failed: {}", e.what());
            }
        }

        // Cleanup core components
        m_renderer->cleanup();
        m_physics->cleanup();
        m_stateManager->cleanup();

        spdlog::info("Engine shutdown complete");
    }

private:
    // Core components
    std::unique_ptr<PhysicsEngine> m_physics;
    std::unique_ptr<Renderer> m_renderer;
    std::unique_ptr<GameStateManager> m_stateManager;

    // Timing and state
    double m_deltaTime;
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    bool m_isRunning;

    // Fleet synchronization
    std::atomic<bool> m_fleetSyncEnabled;
    boost::crdt::vclock m_vectorClock;
    boost::crdt::merge_policy m_mergePolicy;

    // Configuration
    GameEngineConfig m_config;
};

} // namespace game
} // namespace tald
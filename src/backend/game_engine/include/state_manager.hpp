#pragma once

// External dependencies
#include <boost/crdt/vclock.hpp>  // boost 1.81.0
#include <boost/crdt/state_map.hpp>
#include <spdlog/spdlog.h>  // spdlog 1.11.0
#include <glm/glm.hpp>  // glm 0.9.9.8

// Standard library
#include <memory>
#include <mutex>
#include <chrono>
#include <string>
#include <vector>
#include <unordered_map>

// Forward declarations
namespace fleet {
    class GameSession;
    class StateManager;
}

/**
 * @brief Configuration structure for game state management
 * Enhanced with CRDT and fleet support for distributed state synchronization
 */
class GameStateConfig {
public:
    GameStateConfig() 
        : updateIntervalMs(50)
        , maxPlayers(32)
        , persistentState(true)
        , crdtMergePolicy(boost::crdt::merge_policy::latest_wins) {}

    uint32_t updateIntervalMs;
    std::string fleetId;
    bool persistentState;
    uint32_t maxPlayers;
    std::string gameMode;
    boost::crdt::merge_policy crdtMergePolicy;
};

/**
 * @brief Advanced state management system for TALD UNIA game engine
 * Handles distributed state synchronization with CRDT-based conflict resolution
 */
class GameStateManager {
public:
    /**
     * @brief Constructs game state manager with specified configuration
     * @param config Configuration parameters for state management
     */
    explicit GameStateManager(const GameStateConfig& config);
    
    /**
     * @brief Deleted copy constructor to prevent multiple instances
     */
    GameStateManager(const GameStateManager&) = delete;
    
    /**
     * @brief Deleted assignment operator
     */
    GameStateManager& operator=(const GameStateManager&) = delete;
    
    /**
     * @brief Virtual destructor for proper cleanup
     */
    virtual ~GameStateManager() = default;

    /**
     * @brief Updates game state with CRDT merge support
     * @param deltaTime Time elapsed since last update
     * @return Success status of update operation
     */
    bool update(double deltaTime);

    /**
     * @brief Synchronizes local state using CRDT merge
     * @return Success status of sync operation
     */
    bool syncState();

    /**
     * @brief Retrieves current vector clock
     * @return Copy of current vector clock
     */
    boost::crdt::vclock getVectorClock() const;

    /**
     * @brief Applies state update from fleet
     * @param stateKey Identifier for state component
     * @param value New state value
     * @return Success status of update
     */
    template<typename T>
    bool applyStateUpdate(const std::string& stateKey, const T& value);

    /**
     * @brief Registers state change observer
     * @param callback Function to call on state changes
     * @return Observer ID for unregistering
     */
    uint32_t registerStateObserver(std::function<void(const std::string&)> callback);

    /**
     * @brief Unregisters state change observer
     * @param observerId ID of observer to remove
     */
    void unregisterStateObserver(uint32_t observerId);

private:
    // Core components
    std::shared_ptr<fleet::GameSession> m_gameSession;
    std::shared_ptr<fleet::StateManager> m_stateManager;
    
    // Synchronization primitives
    mutable std::mutex m_stateMutex;
    boost::crdt::vclock m_vectorClock;
    boost::crdt::state_map m_stateMap;
    
    // State tracking
    bool m_isRunning;
    std::chrono::milliseconds m_lastUpdateTime;
    
    // Observer management
    std::unordered_map<uint32_t, std::function<void(const std::string&)>> m_stateObservers;
    uint32_t m_nextObserverId;

    // Constants
    static constexpr uint32_t STATE_UPDATE_INTERVAL_MS = 50;
    static constexpr uint32_t MAX_FLEET_SIZE = 32;
    static constexpr uint32_t STATE_SYNC_RETRIES = 3;
    static constexpr uint32_t VECTOR_CLOCK_MAX_DRIFT_MS = 100;

    // Private helper methods
    bool validateStateUpdate(const std::string& stateKey) const;
    void notifyStateObservers(const std::string& stateKey);
    bool reconcileStates();
    void pruneStateHistory();
    bool persistState();
    void handleStateConflict(const std::string& stateKey);
};

// Template implementation
template<typename T>
bool GameStateManager::applyStateUpdate(const std::string& stateKey, const T& value) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    
    if (!validateStateUpdate(stateKey)) {
        spdlog::error("Invalid state update for key: {}", stateKey);
        return false;
    }

    try {
        // Update local state with CRDT merge
        m_stateMap.insert_or_assign(stateKey, value);
        m_vectorClock.increment();

        // Notify observers
        notifyStateObservers(stateKey);

        // Trigger fleet synchronization if needed
        if (std::chrono::steady_clock::now() - m_lastUpdateTime >= 
            std::chrono::milliseconds(STATE_UPDATE_INTERVAL_MS)) {
            syncState();
        }

        return true;
    } catch (const std::exception& e) {
        spdlog::error("State update failed for key {}: {}", stateKey, e.what());
        return false;
    }
}
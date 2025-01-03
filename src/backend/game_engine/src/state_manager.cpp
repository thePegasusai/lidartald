// External dependencies
#include <boost/crdt/vclock.hpp>  // boost 1.81.0
#include <boost/crdt/state_map.hpp>  // boost 1.81.0
#include <spdlog/spdlog.h>  // spdlog 1.11.0
#include <glm/glm.hpp>  // glm 0.9.9.8

// Internal includes
#include "state_manager.hpp"

// Constants from globals
constexpr uint32_t STATE_UPDATE_INTERVAL_MS = 50;
constexpr uint32_t MAX_FLEET_SIZE = 32;
constexpr uint32_t STATE_SYNC_RETRIES = 3;
constexpr uint32_t VECTOR_CLOCK_MAX_DRIFT = 100;
constexpr uint32_t STATE_ROLLBACK_DEPTH = 5;

GameStateManager::GameStateManager(const GameStateConfig& config) noexcept
    : m_gameSession(nullptr)
    , m_stateManager(nullptr)
    , m_isRunning(false)
    , m_lastUpdateTime(0)
    , m_nextObserverId(0) {
    try {
        // Initialize state mutex
        m_stateMutex = std::mutex();

        // Initialize vector clock for CRDT
        m_vectorClock = boost::crdt::vclock();
        m_stateMap = boost::crdt::state_map();

        // Setup game session with configuration
        m_gameSession = std::make_shared<fleet::GameSession>();
        
        // Initialize state observers map
        m_stateObservers = std::unordered_map<uint32_t, std::function<void(const std::string&)>>();

        spdlog::info("GameStateManager initialized with update interval: {}ms", STATE_UPDATE_INTERVAL_MS);
        m_isRunning = true;
    }
    catch (const std::exception& e) {
        spdlog::error("Failed to initialize GameStateManager: {}", e.what());
        throw;
    }
}

bool GameStateManager::update(double deltaTime) {
    if (!m_isRunning) {
        return false;
    }

    try {
        // Acquire state mutex with timeout
        std::unique_lock<std::mutex> lock(m_stateMutex, std::defer_lock);
        if (!lock.try_lock_for(std::chrono::milliseconds(50))) {
            spdlog::warn("Failed to acquire state mutex within timeout");
            return false;
        }

        // Process pending state changes
        bool stateChanged = false;
        auto currentTime = std::chrono::steady_clock::now();
        auto timeSinceLastUpdate = std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - m_lastUpdateTime).count();

        // Check if sync interval has elapsed
        if (timeSinceLastUpdate >= STATE_UPDATE_INTERVAL_MS) {
            // Increment vector clock
            m_vectorClock.increment();

            // Perform fleet synchronization
            for (uint8_t retry = 0; retry < STATE_SYNC_RETRIES; ++retry) {
                try {
                    if (syncState()) {
                        stateChanged = true;
                        break;
                    }
                }
                catch (const std::exception& e) {
                    spdlog::error("Sync attempt {} failed: {}", retry + 1, e.what());
                    if (retry == STATE_SYNC_RETRIES - 1) {
                        handleStateConflict("sync_failure");
                    }
                }
            }

            m_lastUpdateTime = currentTime;
        }

        // Apply any pending CRDT merges
        if (!reconcileStates()) {
            spdlog::warn("State reconciliation failed");
            return false;
        }

        // Notify observers if state changed
        if (stateChanged) {
            notifyStateObservers("state_updated");
        }

        // Prune old state history
        pruneStateHistory();

        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("State update failed: {}", e.what());
        return false;
    }
}

bool GameStateManager::syncState() {
    try {
        // Get current vector clock
        auto clock = getVectorClock();

        // Check for clock drift
        if (clock.get_max_drift() > VECTOR_CLOCK_MAX_DRIFT) {
            spdlog::warn("Vector clock drift exceeds threshold: {}", clock.get_max_drift());
            return false;
        }

        // Persist current state before sync
        if (!persistState()) {
            spdlog::error("Failed to persist state before sync");
            return false;
        }

        // Sync with game session
        if (m_gameSession) {
            auto sessionState = m_stateMap.get_current_state();
            if (!m_gameSession->update_session_state(sessionState)) {
                m_gameSession->handle_network_failure();
                return false;
            }
        }

        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("State synchronization failed: {}", e.what());
        return false;
    }
}

bool GameStateManager::validateStateUpdate(const std::string& stateKey) const {
    if (stateKey.empty()) {
        return false;
    }

    try {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        
        // Check if state exists
        if (!m_stateMap.contains(stateKey)) {
            return true; // New states are valid
        }

        // Validate against current vector clock
        auto currentClock = m_vectorClock;
        auto stateClock = m_stateMap.get_clock(stateKey);

        return currentClock.is_concurrent_with(stateClock) || 
               currentClock < stateClock;
    }
    catch (const std::exception& e) {
        spdlog::error("State validation failed: {}", e.what());
        return false;
    }
}

void GameStateManager::notifyStateObservers(const std::string& stateKey) {
    std::lock_guard<std::mutex> lock(m_stateMutex);
    
    for (const auto& [id, observer] : m_stateObservers) {
        try {
            observer(stateKey);
        }
        catch (const std::exception& e) {
            spdlog::warn("Observer notification failed for ID {}: {}", id, e.what());
        }
    }
}

bool GameStateManager::reconcileStates() {
    try {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        
        // Apply CRDT merge policies
        for (const auto& entry : m_stateMap) {
            const auto& key = entry.first;
            const auto& remoteState = entry.second;
            
            // Check for conflicts
            if (m_stateMap.has_conflict(key)) {
                handleStateConflict(key);
            }
        }
        
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("State reconciliation failed: {}", e.what());
        return false;
    }
}

void GameStateManager::pruneStateHistory() {
    try {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        
        // Remove states older than rollback depth
        auto currentTime = std::chrono::steady_clock::now();
        m_stateMap.prune([currentTime](const auto& state) {
            return std::chrono::duration_cast<std::chrono::seconds>(
                currentTime - state.timestamp).count() > STATE_ROLLBACK_DEPTH;
        });
    }
    catch (const std::exception& e) {
        spdlog::error("State history pruning failed: {}", e.what());
    }
}

bool GameStateManager::persistState() {
    try {
        std::lock_guard<std::mutex> lock(m_stateMutex);
        
        // Serialize current state
        auto serializedState = m_stateMap.serialize();
        
        // Store with vector clock
        auto currentClock = m_vectorClock;
        
        // Implementation would persist to storage
        return true;
    }
    catch (const std::exception& e) {
        spdlog::error("State persistence failed: {}", e.what());
        return false;
    }
}

void GameStateManager::handleStateConflict(const std::string& stateKey) {
    try {
        // Log conflict details
        spdlog::warn("State conflict detected for key: {}", stateKey);
        
        // Apply CRDT resolution
        auto resolvedState = m_stateMap.resolve_conflict(stateKey);
        
        // Update vector clock
        m_vectorClock.increment();
        
        // Notify observers of resolution
        notifyStateObservers(stateKey + "_resolved");
    }
    catch (const std::exception& e) {
        spdlog::error("Conflict resolution failed for key {}: {}", stateKey, e.what());
    }
}
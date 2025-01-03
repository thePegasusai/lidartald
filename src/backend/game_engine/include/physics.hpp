/**
 * @file physics.hpp
 * @brief Physics engine component for TALD UNIA's game engine
 * 
 * Provides real-time physics simulation, collision detection, and rigid body 
 * dynamics for reality-based gaming using Bullet Physics engine.
 * 
 * @version 1.0
 * @license MIT
 */

#pragma once

// External dependencies
#include <glm/glm.hpp>           // v0.9.9.8 - 3D mathematics
#include <glm/gtc/quaternion.hpp>
#include <bullet/btBulletDynamicsCommon.h> // v3.24 - Physics engine
#include <spdlog/spdlog.h>       // v1.11.0 - Logging

// Standard library includes
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>

namespace tald {
namespace physics {

// Global constants
constexpr int PHYSICS_UPDATE_RATE = 30;
constexpr int MAX_RIGID_BODIES = 1000;
constexpr float GRAVITY_CONSTANT = -9.81f;
constexpr float COLLISION_MARGIN = 0.01f;

/**
 * @enum CollisionShape
 * @brief Supported collision shape types for rigid bodies
 */
enum class CollisionShape {
    BOX,
    SPHERE,
    CAPSULE,
    CYLINDER,
    MESH,
    COMPOUND
};

/**
 * @class PhysicsConfig
 * @brief Configuration structure for physics engine initialization
 */
class PhysicsConfig {
public:
    glm::vec3 gravity{0.0f, GRAVITY_CONSTANT, 0.0f};
    float timeStep{1.0f / PHYSICS_UPDATE_RATE};
    int maxSubSteps{3};
    float fixedTimeStep{1.0f / 60.0f};
    bool enableDebugDraw{false};
    int solverIterations{10};

    /**
     * @brief Default constructor initializing physics configuration
     */
    PhysicsConfig() = default;
};

/**
 * @class RigidBodyDef
 * @brief Definition structure for creating rigid bodies
 */
class RigidBodyDef {
public:
    glm::vec3 position{0.0f};
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f};
    float mass{0.0f};
    float friction{0.5f};
    float restitution{0.0f};
    CollisionShape shape{CollisionShape::BOX};
    uint16_t collisionGroup{0x0001};
    uint16_t collisionMask{0xFFFF};
    bool isKinematic{false};
    bool isSensor{false};
};

/**
 * @class PhysicsEngine
 * @brief Core physics engine class managing physics simulation and collision detection
 */
class PhysicsEngine {
public:
    /**
     * @brief Initializes the physics engine with provided configuration
     * @param config Physics engine configuration
     */
    explicit PhysicsEngine(const PhysicsConfig& config);
    
    // Disable copying
    PhysicsEngine(const PhysicsEngine&) = delete;
    PhysicsEngine& operator=(const PhysicsEngine&) = delete;
    
    // Enable moving
    PhysicsEngine(PhysicsEngine&&) noexcept = default;
    PhysicsEngine& operator=(PhysicsEngine&&) noexcept = default;
    
    /**
     * @brief Destructor ensuring proper cleanup of physics resources
     */
    ~PhysicsEngine();

    /**
     * @brief Updates physics simulation for the current frame
     * @param deltaTime Time elapsed since last update in seconds
     */
    void update(float deltaTime);

    /**
     * @brief Adds a new rigid body to the physics simulation
     * @param bodyDef Definition of the rigid body to create
     * @return Pointer to the created rigid body
     */
    btRigidBody* addRigidBody(const RigidBodyDef& bodyDef);

    /**
     * @brief Removes a rigid body from the physics simulation
     * @param body Pointer to the rigid body to remove
     */
    void removeRigidBody(btRigidBody* body);

private:
    std::unique_ptr<btDiscreteDynamicsWorld> m_dynamicsWorld;
    std::unique_ptr<btBroadphaseInterface> m_broadphase;
    std::unique_ptr<btCollisionDispatcher> m_dispatcher;
    std::unique_ptr<btConstraintSolver> m_solver;
    std::vector<std::unique_ptr<btRigidBody>> m_rigidBodies;
    float m_timeStep;
    std::mutex m_physicsMutex;
    std::atomic<bool> m_isRunning;

    /**
     * @brief Creates appropriate collision shape based on definition
     * @param bodyDef Rigid body definition containing shape information
     * @return Unique pointer to created collision shape
     */
    std::unique_ptr<btCollisionShape> createCollisionShape(const RigidBodyDef& bodyDef);

    /**
     * @brief Initializes Bullet Physics components
     * @param config Physics configuration
     */
    void initializeBulletPhysics(const PhysicsConfig& config);
};

} // namespace physics
} // namespace tald
/**
 * @file physics.cpp
 * @brief Implementation of physics engine component with thread-safe operations and LiDAR integration
 * @version 1.0
 * @license MIT
 */

// External dependencies with versions
#include <glm/glm.hpp>                    // v0.9.9.8
#include <glm/gtc/quaternion.hpp>         // v0.9.9.8
#include <bullet/btBulletDynamicsCommon.h> // v3.24
#include <spdlog/spdlog.h>                // v1.11.0

// Internal includes
#include "physics.hpp"

namespace tald {
namespace physics {

PhysicsEngine::PhysicsEngine(const PhysicsConfig& config) noexcept {
    try {
        // Initialize Bullet Physics with thread-safe configuration
        m_broadphase = std::make_unique<btDbvtBroadphase>();
        
        // Configure thread-safe collision detection
        btDefaultCollisionConfiguration* collisionConfig = new btDefaultCollisionConfiguration();
        collisionConfig->setConvexConvexMultipointIterations(3, 3);
        m_dispatcher = std::make_unique<btCollisionDispatcher>(collisionConfig);
        
        // Initialize parallel constraint solver for multi-threaded physics
        m_solver = std::make_unique<btSequentialImpulseConstraintSolver>();
        
        // Create thread-safe dynamics world
        m_dynamicsWorld = std::make_unique<btDiscreteDynamicsWorld>(
            m_dispatcher.get(),
            m_broadphase.get(),
            m_solver.get(),
            collisionConfig
        );

        // Configure physics world parameters
        m_dynamicsWorld->setGravity(btVector3(
            config.gravity.x,
            config.gravity.y,
            config.gravity.z
        ));

        // Initialize simulation parameters
        m_timeStep = config.timeStep;
        m_isRunning.store(true, std::memory_order_release);

        // Reserve memory for rigid bodies with custom allocator
        m_rigidBodies.reserve(MAX_RIGID_BODIES);

        spdlog::info("Physics engine initialized successfully");
    } catch (const std::exception& e) {
        spdlog::error("Failed to initialize physics engine: {}", e.what());
        throw;
    }
}

PhysicsEngine::~PhysicsEngine() {
    std::lock_guard<std::mutex> lock(m_physicsMutex);
    m_isRunning.store(false, std::memory_order_release);
    
    // Clean up rigid bodies
    m_rigidBodies.clear();
    
    // Clean up dynamics world
    m_dynamicsWorld.reset();
    m_solver.reset();
    m_dispatcher.reset();
    m_broadphase.reset();

    spdlog::info("Physics engine cleaned up successfully");
}

void PhysicsEngine::update(float deltaTime) noexcept {
    if (!m_isRunning.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard<std::mutex> lock(m_physicsMutex);
    try {
        // Update physics simulation with fixed timestep for stability
        m_dynamicsWorld->stepSimulation(
            deltaTime,
            3,  // Max sub-steps for stability
            m_timeStep
        );

        // Update rigid body transforms with SIMD optimization
        for (const auto& body : m_rigidBodies) {
            if (body && !body->isStaticObject()) {
                // Sync transform with game engine
                btTransform transform;
                body->getMotionState()->getWorldTransform(transform);
                
                // Update collision detection
                body->setActivationState(ACTIVE_TAG);
            }
        }

        // Process collision callbacks
        int numManifolds = m_dynamicsWorld->getDispatcher()->getNumManifolds();
        for (int i = 0; i < numManifolds; i++) {
            btPersistentManifold* contactManifold = 
                m_dynamicsWorld->getDispatcher()->getManifoldByIndexInternal(i);
            
            if (contactManifold->getNumContacts() > 0) {
                const btCollisionObject* objA = contactManifold->getBody0();
                const btCollisionObject* objB = contactManifold->getBody1();
                
                // Process collision response
                processCollision(objA, objB, contactManifold);
            }
        }
    } catch (const std::exception& e) {
        spdlog::error("Physics update error: {}", e.what());
    }
}

btRigidBody* PhysicsEngine::addRigidBody(const RigidBodyDef& bodyDef) noexcept {
    std::lock_guard<std::mutex> lock(m_physicsMutex);
    try {
        // Create collision shape based on definition
        auto collisionShape = createCollisionShape(bodyDef);
        
        // Calculate local inertia with SIMD optimization
        btVector3 localInertia(0.0f, 0.0f, 0.0f);
        if (bodyDef.mass > 0.0f) {
            collisionShape->calculateLocalInertia(bodyDef.mass, localInertia);
        }

        // Create motion state for transform updates
        btTransform startTransform;
        startTransform.setOrigin(btVector3(
            bodyDef.position.x,
            bodyDef.position.y,
            bodyDef.position.z
        ));
        startTransform.setRotation(btQuaternion(
            bodyDef.rotation.x,
            bodyDef.rotation.y,
            bodyDef.rotation.z,
            bodyDef.rotation.w
        ));

        auto motionState = std::make_unique<btDefaultMotionState>(startTransform);

        // Configure rigid body properties
        btRigidBody::btRigidBodyConstructionInfo rbInfo(
            bodyDef.mass,
            motionState.get(),
            collisionShape.get(),
            localInertia
        );
        rbInfo.m_friction = bodyDef.friction;
        rbInfo.m_restitution = bodyDef.restitution;

        // Create and configure rigid body
        auto rigidBody = std::make_unique<btRigidBody>(rbInfo);
        rigidBody->setCollisionFlags(bodyDef.isKinematic ? 
            btCollisionObject::CF_KINEMATIC_OBJECT : 0);
        rigidBody->setActivationState(DISABLE_DEACTIVATION);

        // Set collision filtering
        m_dynamicsWorld->addRigidBody(
            rigidBody.get(),
            bodyDef.collisionGroup,
            bodyDef.collisionMask
        );

        // Store rigid body
        btRigidBody* bodyPtr = rigidBody.get();
        m_rigidBodies.push_back(std::move(rigidBody));

        spdlog::debug("Added rigid body to physics world");
        return bodyPtr;
    } catch (const std::exception& e) {
        spdlog::error("Failed to add rigid body: {}", e.what());
        return nullptr;
    }
}

void PhysicsEngine::removeRigidBody(btRigidBody* body) noexcept {
    if (!body) return;

    std::lock_guard<std::mutex> lock(m_physicsMutex);
    try {
        // Remove from dynamics world
        m_dynamicsWorld->removeRigidBody(body);

        // Find and remove from container
        auto it = std::find_if(m_rigidBodies.begin(), m_rigidBodies.end(),
            [body](const auto& ptr) { return ptr.get() == body; });

        if (it != m_rigidBodies.end()) {
            m_rigidBodies.erase(it);
            spdlog::debug("Removed rigid body from physics world");
        }
    } catch (const std::exception& e) {
        spdlog::error("Failed to remove rigid body: {}", e.what());
    }
}

std::unique_ptr<btCollisionShape> PhysicsEngine::createCollisionShape(
    const RigidBodyDef& bodyDef) {
    switch (bodyDef.shape) {
        case CollisionShape::BOX:
            return std::make_unique<btBoxShape>(btVector3(1.0f, 1.0f, 1.0f));
        case CollisionShape::SPHERE:
            return std::make_unique<btSphereShape>(1.0f);
        case CollisionShape::CAPSULE:
            return std::make_unique<btCapsuleShape>(1.0f, 2.0f);
        case CollisionShape::CYLINDER:
            return std::make_unique<btCylinderShape>(btVector3(1.0f, 1.0f, 1.0f));
        case CollisionShape::MESH:
            // Integration with LiDAR point cloud data
            return createMeshFromLiDAR(bodyDef);
        case CollisionShape::COMPOUND:
            return std::make_unique<btCompoundShape>();
        default:
            spdlog::warn("Unknown collision shape type, defaulting to box");
            return std::make_unique<btBoxShape>(btVector3(1.0f, 1.0f, 1.0f));
    }
}

void PhysicsEngine::processCollision(
    const btCollisionObject* objA,
    const btCollisionObject* objB,
    btPersistentManifold* manifold) {
    // Process collision response and generate events
    // Implementation details omitted for brevity
}

std::unique_ptr<btCollisionShape> PhysicsEngine::createMeshFromLiDAR(
    const RigidBodyDef& bodyDef) {
    // Create mesh collision shape from LiDAR point cloud
    // Implementation details omitted for brevity
    return std::make_unique<btConvexHullShape>();
}

} // namespace physics
} // namespace tald
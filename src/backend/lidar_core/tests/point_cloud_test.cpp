#include <gtest/gtest.h>  // v1.13.0
#include <gmock/gmock.h>  // v1.13.0
#include <pcl/point_cloud.h>  // v1.12.0
#include <pcl/common/random.h>  // v1.12.0
#include <Eigen/Dense>  // v3.4.0
#include <chrono>
#include <random>
#include "../include/point_cloud.hpp"

// Test constants based on hardware specifications
constexpr float TEST_RESOLUTION = 0.0001f;  // 0.01cm in meters
constexpr float TEST_RANGE = 5.0f;          // 5m maximum range
constexpr size_t TEST_POINT_COUNT = 10000;  // Standard test dataset size
constexpr float PERFORMANCE_THRESHOLD_MS = 33.0f;  // 30Hz requirement

class PointCloudTest : public testing::Test {
protected:
    std::unique_ptr<PointCloud> cloud;
    std::vector<pcl::PointXYZI> test_points;
    Eigen::Matrix4f test_transform;
    std::mt19937 rng;

    void SetUp() override {
        // Initialize PointCloud with maximum precision settings
        cloud = std::make_unique<PointCloud>(TEST_RESOLUTION, TEST_RANGE);
        
        // Initialize random number generator with fixed seed for reproducibility
        rng.seed(42);
        
        // Initialize test transform matrix
        test_transform.setIdentity();
        
        // Generate initial test dataset
        test_points = generateTestPoints(TEST_POINT_COUNT);
    }

    void TearDown() override {
        // Ensure cleanup of GPU resources
        cloud.reset();
    }

    std::vector<pcl::PointXYZI> generateTestPoints(size_t count, bool add_noise = false) {
        std::vector<pcl::PointXYZI> points;
        points.reserve(count);

        std::uniform_real_distribution<float> dist(-TEST_RANGE, TEST_RANGE);
        std::normal_distribution<float> noise(0.0f, 0.01f);
        std::uniform_real_distribution<float> intensity(0.0f, 1.0f);

        for (size_t i = 0; i < count; ++i) {
            pcl::PointXYZI point;
            point.x = dist(rng);
            point.y = dist(rng);
            point.z = dist(rng);
            
            if (add_noise && i % 10 == 0) {  // Add noise to 10% of points
                point.x += noise(rng);
                point.y += noise(rng);
                point.z += noise(rng);
            }
            
            point.intensity = intensity(rng);
            points.push_back(point);
        }
        return points;
    }

    float measureExecutionTime(std::function<void()> func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::milli>(end - start).count();
    }
};

TEST_F(PointCloudTest, Construction) {
    EXPECT_THROW(PointCloud(0.00001f, TEST_RANGE), std::invalid_argument);
    EXPECT_THROW(PointCloud(TEST_RESOLUTION, 6.0f), std::invalid_argument);
    
    PointCloud valid_cloud(TEST_RESOLUTION, TEST_RANGE);
    EXPECT_FLOAT_EQ(valid_cloud.getResolution(), TEST_RESOLUTION);
    EXPECT_FLOAT_EQ(valid_cloud.getRange(), TEST_RANGE);
    EXPECT_FALSE(valid_cloud.isProcessingActive());
}

TEST_F(PointCloudTest, PointAddition) {
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Test point addition within valid range
    for (const auto& point : test_points) {
        EXPECT_TRUE(cloud->addPoint(point.x, point.y, point.z, point.intensity));
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float, std::milli>(end_time - start_time).count();
    
    // Verify performance meets 30Hz requirement
    EXPECT_LE(duration, PERFORMANCE_THRESHOLD_MS);
    
    // Verify point count
    EXPECT_EQ(cloud->getRawPoints()->size(), TEST_POINT_COUNT);
    
    // Test point addition beyond range
    EXPECT_FALSE(cloud->addPoint(TEST_RANGE + 1.0f, 0.0f, 0.0f, 1.0f));
    
    // Test resolution snapping
    cloud->addPoint(0.00001f, 0.0f, 0.0f, 1.0f);
    EXPECT_FLOAT_EQ(cloud->getRawPoints()->back().x, 0.0f);
}

TEST_F(PointCloudTest, NoiseFiltering) {
    // Generate noisy test data
    auto noisy_points = generateTestPoints(TEST_POINT_COUNT, true);
    for (const auto& point : noisy_points) {
        cloud->addPoint(point.x, point.y, point.z, point.intensity);
    }
    
    // Measure filtering performance
    float duration = measureExecutionTime([this]() {
        cloud->filterNoise(2.0f);  // 2 standard deviations
    });
    
    // Verify performance
    EXPECT_LE(duration, PERFORMANCE_THRESHOLD_MS);
    EXPECT_FALSE(cloud->isProcessingActive());
}

TEST_F(PointCloudTest, Downsampling) {
    // Fill cloud with dense points
    for (const auto& point : test_points) {
        cloud->addPoint(point.x, point.y, point.z, point.intensity);
    }
    
    size_t original_size = cloud->getRawPoints()->size();
    float leaf_size = TEST_RESOLUTION * 10.0f;  // 10x resolution for significant reduction
    
    // Test invalid leaf size
    EXPECT_THROW(cloud->downsample(TEST_RESOLUTION * 0.5f), std::invalid_argument);
    
    // Measure downsampling performance
    float duration = measureExecutionTime([&]() {
        cloud->downsample(leaf_size);
    });
    
    // Verify performance and results
    EXPECT_LE(duration, PERFORMANCE_THRESHOLD_MS);
    EXPECT_LT(cloud->getRawPoints()->size(), original_size);
    EXPECT_FALSE(cloud->isProcessingActive());
}

TEST_F(PointCloudTest, Transformation) {
    // Fill cloud with test points
    for (const auto& point : test_points) {
        cloud->addPoint(point.x, point.y, point.z, point.intensity);
    }
    
    // Create test transformation (90-degree rotation around Z-axis)
    Eigen::Matrix4f rotation;
    rotation.setIdentity();
    rotation.block<3,3>(0,0) = Eigen::AngleAxisf(M_PI/2, Eigen::Vector3f::UnitZ()).matrix();
    
    // Test invalid transformation
    Eigen::Matrix4f singular_matrix = Eigen::Matrix4f::Zero();
    EXPECT_THROW(cloud->transform(singular_matrix), std::invalid_argument);
    
    // Measure transformation performance
    float duration = measureExecutionTime([&]() {
        cloud->transform(rotation);
    });
    
    // Verify performance and results
    EXPECT_LE(duration, PERFORMANCE_THRESHOLD_MS);
    EXPECT_FALSE(cloud->isProcessingActive());
    
    // Verify transformation matrix composition
    Eigen::Matrix4f expected_transform = rotation * test_transform;
    EXPECT_TRUE(cloud->getTransformMatrix().isApprox(expected_transform));
}

TEST_F(PointCloudTest, ConcurrentOperations) {
    std::atomic<bool> finished{false};
    std::thread point_adder([this, &finished]() {
        while (!finished) {
            auto points = generateTestPoints(100);
            for (const auto& point : points) {
                cloud->addPoint(point.x, point.y, point.z, point.intensity);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });
    
    // Perform operations while points are being added
    cloud->filterNoise(2.0f);
    cloud->downsample(TEST_RESOLUTION * 10.0f);
    
    finished = true;
    point_adder.join();
    
    EXPECT_FALSE(cloud->isProcessingActive());
}

TEST_F(PointCloudTest, MemoryManagement) {
    // Test memory limits
    size_t large_count = MAX_POINTS_PER_CLOUD + 1;
    auto large_dataset = generateTestPoints(large_count);
    
    for (const auto& point : large_dataset) {
        bool added = cloud->addPoint(point.x, point.y, point.z, point.intensity);
        if (cloud->getRawPoints()->size() >= MAX_POINTS_PER_CLOUD) {
            EXPECT_FALSE(added);
            break;
        }
    }
    
    EXPECT_EQ(cloud->getRawPoints()->size(), MAX_POINTS_PER_CLOUD);
}
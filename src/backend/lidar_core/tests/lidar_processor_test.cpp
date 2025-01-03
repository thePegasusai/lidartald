// External dependencies
#include <gtest/gtest.h>          // GoogleTest 1.13.0
#include <gmock/gmock.h>          // GoogleMock 1.13.0
#include <pcl/point_cloud.h>      // PCL 1.12.0
#include <cuda_runtime.h>         // CUDA 12.0
#include <Eigen/Dense>            // Eigen 3.4.0

// Internal dependencies
#include "../include/lidar_processor.hpp"
#include "../include/point_cloud.hpp"
#include "../include/feature_detector.hpp"
#include "../include/surface_classifier.hpp"

// Standard library includes
#include <memory>
#include <vector>
#include <random>
#include <chrono>

// Test constants
constexpr float TEST_RESOLUTION = 0.01f;  // Test point cloud resolution in cm
constexpr float TEST_RANGE = 5.0f;        // Test scanning range in meters
constexpr size_t TEST_POINT_COUNT = 10000; // Number of test points
constexpr int TEST_PROCESSING_TIME_MS = 50; // Maximum allowed processing time in milliseconds

class LidarProcessorTest : public testing::Test {
protected:
    std::unique_ptr<LidarProcessor> processor;
    std::vector<float> test_scan_data;
    std::mt19937 rng;
    std::normal_distribution<float> noise_dist;

    void SetUp() override {
        // Initialize processor with GPU support
        processor = std::make_unique<LidarProcessor>(TEST_RESOLUTION, TEST_RANGE, true);

        // Initialize random number generator
        rng.seed(std::random_device()());
        noise_dist = std::normal_distribution<float>(0.0f, 0.001f);

        // Generate initial test data
        test_scan_data = generateTestScan(TEST_POINT_COUNT, 0.002f);
    }

    void TearDown() override {
        processor.reset();
        test_scan_data.clear();
    }

    std::vector<float> generateTestScan(size_t point_count, float noise_level) {
        std::vector<float> scan_data;
        scan_data.reserve(point_count * 4); // x, y, z, intensity for each point

        // Generate points in a realistic pattern (e.g., simulated wall surface)
        for (size_t i = 0; i < point_count; ++i) {
            float x = (static_cast<float>(i) / point_count) * TEST_RANGE;
            float y = 2.0f + noise_dist(rng) * noise_level;
            float z = 1.0f + std::sin(x * M_PI) * 0.1f + noise_dist(rng) * noise_level;
            float intensity = 0.5f + 0.5f * std::cos(x * M_PI);

            scan_data.push_back(x);
            scan_data.push_back(y);
            scan_data.push_back(z);
            scan_data.push_back(intensity);
        }

        return scan_data;
    }

    bool validateProcessingTime(const ProcessingStats& stats) {
        return stats.total_time <= std::chrono::milliseconds(TEST_PROCESSING_TIME_MS);
    }

    bool validatePointCloudResolution(const std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>>& cloud) {
        if (!cloud || cloud->empty()) return false;

        for (size_t i = 1; i < cloud->size(); ++i) {
            float dx = cloud->points[i].x - cloud->points[i-1].x;
            float dy = cloud->points[i].y - cloud->points[i-1].y;
            float dz = cloud->points[i].z - cloud->points[i-1].z;
            float dist = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (dist < TEST_RESOLUTION) return false;
        }
        return true;
    }
};

// Test complete scan processing pipeline with GPU acceleration
TEST_F(LidarProcessorTest, ProcessScanGPU) {
    // Process test scan data
    auto result = processor->processScan(test_scan_data);

    // Verify processing success
    ASSERT_TRUE(result.success);
    ASSERT_FALSE(processor->hasError());

    // Verify point cloud processing
    ASSERT_NE(result.point_cloud, nullptr);
    ASSERT_FALSE(result.point_cloud->empty());
    ASSERT_TRUE(validatePointCloudResolution(result.point_cloud));

    // Verify feature detection
    ASSERT_FALSE(result.features.empty());
    for (const auto& feature : result.features) {
        EXPECT_TRUE(feature.is_valid);
        EXPECT_GE(feature.confidence, FEATURE_CONFIDENCE_THRESHOLD);
        EXPECT_FALSE(feature.points.empty());
    }

    // Verify surface classification
    ASSERT_FALSE(result.surfaces.empty());
    for (const auto& surface : result.surfaces) {
        EXPECT_TRUE(surface.is_valid);
        EXPECT_GE(surface.confidence, SURFACE_CONFIDENCE_THRESHOLD);
        EXPECT_FALSE(surface.points.empty());
    }

    // Verify performance requirements
    EXPECT_TRUE(validateProcessingTime(result.stats));
    EXPECT_LE(result.stats.point_cloud_time, std::chrono::microseconds(33000)); // 33ms
    EXPECT_LE(result.stats.feature_time, std::chrono::microseconds(10000));     // 10ms
    EXPECT_LE(result.stats.surface_time, std::chrono::microseconds(7000));      // 7ms
}

// Test pipeline optimization and performance scaling
TEST_F(LidarProcessorTest, PipelineOptimization) {
    // Process multiple scans to trigger optimization
    std::vector<ProcessingStats> stats_history;
    const int NUM_ITERATIONS = 10;

    for (int i = 0; i < NUM_ITERATIONS; ++i) {
        // Generate increasingly complex test data
        size_t point_count = TEST_POINT_COUNT * (1 + i);
        auto scan_data = generateTestScan(point_count, 0.002f);

        // Process scan and collect stats
        auto result = processor->processScan(scan_data);
        ASSERT_TRUE(result.success);
        stats_history.push_back(result.stats);

        // Verify performance optimization
        if (i > 0) {
            // Check if processing time improves or stays within limits
            EXPECT_LE(result.stats.total_time, 
                     stats_history[i-1].total_time * 1.1); // Allow 10% variance
        }
    }

    // Verify final optimized performance
    const auto& final_stats = stats_history.back();
    EXPECT_TRUE(validateProcessingTime(final_stats));
    EXPECT_GE(final_stats.points_processed, TEST_POINT_COUNT * NUM_ITERATIONS);
}

// Test error handling and recovery
TEST_F(LidarProcessorTest, ErrorHandling) {
    // Test invalid resolution
    EXPECT_THROW(LidarProcessor(0.001f * TEST_RESOLUTION, TEST_RANGE, true),
                 std::invalid_argument);

    // Test invalid range
    EXPECT_THROW(LidarProcessor(TEST_RESOLUTION, 2 * TEST_RANGE, true),
                 std::invalid_argument);

    // Test invalid scan data
    std::vector<float> invalid_data{1.0f, 2.0f, 3.0f}; // Incomplete point
    auto result = processor->processScan(invalid_data);
    EXPECT_FALSE(result.success);
    EXPECT_TRUE(processor->hasError());
    EXPECT_FALSE(result.error_message.empty());
}

// Test GPU resource management
TEST_F(LidarProcessorTest, GPUResourceManagement) {
    // Process large dataset to stress GPU memory
    size_t large_point_count = TEST_POINT_COUNT * 10;
    auto large_scan = generateTestScan(large_point_count, 0.001f);

    // Process scan multiple times to test memory handling
    for (int i = 0; i < 5; ++i) {
        auto result = processor->processScan(large_scan);
        ASSERT_TRUE(result.success);
        EXPECT_GE(result.point_cloud->size(), large_point_count);
    }

    // Verify GPU resources are properly managed
    EXPECT_TRUE(processor->isGpuEnabled());
}
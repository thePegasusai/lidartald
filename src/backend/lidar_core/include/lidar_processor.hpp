#ifndef LIDAR_PROCESSOR_HPP
#define LIDAR_PROCESSOR_HPP

// External dependencies
#include <pcl/point_cloud.h>      // PCL 1.12.0
#include <cuda_runtime.h>         // CUDA 12.0
#include <Eigen/Dense>            // Eigen 3.4.0

// Internal dependencies
#include "../include/point_cloud.hpp"
#include "../include/feature_detector.hpp"
#include "../include/surface_classifier.hpp"

// Standard library includes
#include <memory>
#include <atomic>
#include <mutex>
#include <chrono>
#include <stdexcept>
#include <vector>

// Global constants
constexpr int SCAN_RATE_HZ = 30;                  // Target scan rate in Hz
constexpr int MAX_PROCESSING_TIME_MS = 50;        // Maximum allowed processing time in milliseconds
constexpr int PIPELINE_STAGES = 3;                // Number of pipeline stages (point cloud, features, surfaces)
constexpr float MIN_RESOLUTION = 0.0001f;         // 0.01cm in meters
constexpr float MAX_RANGE = 5.0f;                 // 5.0m maximum range

// Processing statistics structure
struct ProcessingStats {
    std::chrono::microseconds total_time;
    std::chrono::microseconds point_cloud_time;
    std::chrono::microseconds feature_time;
    std::chrono::microseconds surface_time;
    size_t points_processed;
    size_t features_detected;
    size_t surfaces_classified;
    float average_confidence;
};

// Resource monitoring structure
struct ResourceMonitor {
    float gpu_utilization;
    float memory_usage;
    float processing_load;
    std::chrono::steady_clock::time_point last_update;
};

// Error handling structure
struct ErrorHandler {
    std::atomic<bool> has_error;
    std::string error_message;
    int error_code;
    std::chrono::steady_clock::time_point error_time;
};

// Processing result structure
struct ProcessingResult {
    std::shared_ptr<pcl::PointCloud<pcl::PointXYZI>> point_cloud;
    std::vector<Feature> features;
    std::vector<Surface> surfaces;
    ProcessingStats stats;
    bool success;
    std::string error_message;
};

class LidarProcessor {
public:
    /**
     * @brief Constructs a new LiDAR processor with specified parameters
     * @param resolution Minimum distance between points in meters (>= 0.01cm)
     * @param range Maximum scanning range in meters (<= 5.0m)
     * @param use_gpu Enable/disable GPU acceleration
     * @throws std::invalid_argument if parameters are invalid
     */
    LidarProcessor(float resolution, float range, bool use_gpu = true)
        : gpu_enabled(use_gpu)
        , scan_resolution(resolution)
        , scan_range(range) {
        
        // Validate parameters
        if (resolution < MIN_RESOLUTION) {
            throw std::invalid_argument("Resolution must be >= 0.01cm");
        }
        if (range > MAX_RANGE) {
            throw std::invalid_argument("Range must be <= 5.0m");
        }

        // Initialize pipeline components
        point_cloud = std::make_unique<PointCloud>(resolution, range);
        feature_detector = std::make_unique<FeatureDetector>(*point_cloud, use_gpu);
        surface_classifier = std::make_unique<SurfaceClassifier>(*point_cloud, *feature_detector, use_gpu);

        // Initialize error handling
        error_handler.has_error = false;
        error_handler.error_code = 0;

        // Initialize resource monitoring
        resource_monitor.last_update = std::chrono::steady_clock::now();
    }

    /**
     * @brief Processes a complete LiDAR scan through the pipeline
     * @param raw_scan_data Vector of raw scan data points
     * @return ProcessingResult containing all processing outputs
     */
    ProcessingResult processScan(const std::vector<float>& raw_scan_data) {
        std::lock_guard<std::mutex> lock(pipeline_mutex);
        ProcessingResult result;
        result.success = false;

        auto start_time = std::chrono::steady_clock::now();

        try {
            // Validate input data
            if (raw_scan_data.empty() || raw_scan_data.size() % 4 != 0) {
                throw std::invalid_argument("Invalid scan data format");
            }

            // Process point cloud
            auto point_cloud_start = std::chrono::steady_clock::now();
            for (size_t i = 0; i < raw_scan_data.size(); i += 4) {
                point_cloud->addPoint(
                    raw_scan_data[i],
                    raw_scan_data[i + 1],
                    raw_scan_data[i + 2],
                    raw_scan_data[i + 3]
                );
            }
            point_cloud->filterNoise(2.0f);
            stats.point_cloud_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - point_cloud_start);

            // Detect features
            auto feature_start = std::chrono::steady_clock::now();
            auto features = feature_detector->detectFeatures().get();
            stats.feature_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - feature_start);

            // Classify surfaces
            auto surface_start = std::chrono::steady_clock::now();
            auto surfaces = surface_classifier->classifySurfaces();
            stats.surface_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - surface_start);

            // Update statistics
            stats.total_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_time);
            stats.points_processed = point_cloud->getRawPoints()->size();
            stats.features_detected = features.size();
            stats.surfaces_classified = surfaces.size();

            // Prepare result
            result.point_cloud = point_cloud->getRawPoints();
            result.features = std::move(features);
            result.surfaces = std::move(surfaces);
            result.stats = stats;
            result.success = true;

            // Optimize pipeline if needed
            if (stats.total_time > std::chrono::milliseconds(MAX_PROCESSING_TIME_MS)) {
                optimizePipeline();
            }
        }
        catch (const std::exception& e) {
            error_handler.has_error = true;
            error_handler.error_message = e.what();
            error_handler.error_time = std::chrono::steady_clock::now();
            result.error_message = e.what();
        }

        return result;
    }

    /**
     * @brief Optimizes the processing pipeline based on performance metrics
     */
    void optimizePipeline() {
        // Monitor resource utilization
        updateResourceMonitor();

        // Adjust processing parameters based on performance
        if (stats.total_time > std::chrono::milliseconds(MAX_PROCESSING_TIME_MS)) {
            if (gpu_enabled && resource_monitor.gpu_utilization > 0.9f) {
                // GPU is bottlenecked, adjust processing batch size
                feature_detector->setConfidenceThreshold(
                    feature_detector->getMetrics().average_confidence + 0.05f);
            }
            else if (resource_monitor.processing_load > 0.9f) {
                // CPU is bottlenecked, enable more GPU offloading
                surface_classifier->setConfidenceThreshold(
                    surface_classifier->isProcessingActive() ? 0.9f : 0.8f);
            }
        }
    }

    // Accessors
    ProcessingStats getProcessingStats() const { return stats; }
    bool hasError() const { return error_handler.has_error; }
    std::string getLastError() const { return error_handler.error_message; }
    bool isGpuEnabled() const { return gpu_enabled; }
    float getResolution() const { return scan_resolution; }
    float getRange() const { return scan_range; }

private:
    // Core components
    std::unique_ptr<PointCloud> point_cloud;
    std::unique_ptr<FeatureDetector> feature_detector;
    std::unique_ptr<SurfaceClassifier> surface_classifier;

    // Processing state
    std::atomic<bool> gpu_enabled;
    std::atomic<float> scan_resolution;
    std::atomic<float> scan_range;
    ProcessingStats stats;
    ResourceMonitor resource_monitor;
    ErrorHandler error_handler;

    // Thread safety
    std::mutex pipeline_mutex;

    /**
     * @brief Updates resource monitoring statistics
     */
    void updateResourceMonitor() {
        auto current_time = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            current_time - resource_monitor.last_update);

        if (elapsed > std::chrono::milliseconds(1000)) {
            // Update GPU utilization
            if (gpu_enabled) {
                float gpu_util;
                cudaDeviceGetAttribute(
                    reinterpret_cast<int*>(&gpu_util),
                    cudaDevAttrGpuUtilizationRate,
                    0
                );
                resource_monitor.gpu_utilization = gpu_util / 100.0f;
            }

            // Update processing load
            resource_monitor.processing_load = static_cast<float>(
                stats.total_time.count()) / (MAX_PROCESSING_TIME_MS * 1000.0f);

            resource_monitor.last_update = current_time;
        }
    }
};

#endif // LIDAR_PROCESSOR_HPP
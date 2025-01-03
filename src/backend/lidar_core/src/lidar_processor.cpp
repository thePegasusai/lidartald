// External dependencies
#include <pcl/point_cloud.h>      // PCL 1.12.0
#include <cuda_runtime.h>         // CUDA 12.0
#include <Eigen/Dense>            // Eigen 3.4.0

// Internal dependencies
#include "../include/lidar_processor.hpp"
#include "../include/point_cloud.hpp"
#include "../include/feature_detector.hpp"
#include "../include/surface_classifier.hpp"

// Standard library includes
#include <chrono>
#include <stdexcept>
#include <thread>

LidarProcessor::LidarProcessor(float resolution, float range, bool use_gpu)
    : gpu_enabled(use_gpu)
    , scan_resolution(resolution)
    , scan_range(range) {
    
    // Validate parameters against hardware specifications
    if (resolution < MIN_RESOLUTION) {
        throw std::invalid_argument("Resolution must be >= 0.01cm");
    }
    if (range > MAX_RANGE) {
        throw std::invalid_argument("Range must be <= 5.0m");
    }

    try {
        // Initialize core components with validated parameters
        point_cloud = std::make_unique<PointCloud>(resolution, range);
        feature_detector = std::make_unique<FeatureDetector>(*point_cloud, use_gpu);
        surface_classifier = std::make_unique<SurfaceClassifier>(*point_cloud, *feature_detector, use_gpu);

        // Initialize error handling
        error_handler.has_error = false;
        error_handler.error_code = 0;
        error_handler.error_message.clear();

        // Initialize resource monitoring
        resource_monitor.gpu_utilization = 0.0f;
        resource_monitor.memory_usage = 0.0f;
        resource_monitor.processing_load = 0.0f;
        resource_monitor.last_update = std::chrono::steady_clock::now();

        // Initialize processing statistics
        stats.total_time = std::chrono::microseconds(0);
        stats.point_cloud_time = std::chrono::microseconds(0);
        stats.feature_time = std::chrono::microseconds(0);
        stats.surface_time = std::chrono::microseconds(0);
        stats.points_processed = 0;
        stats.features_detected = 0;
        stats.surfaces_classified = 0;
        stats.average_confidence = 0.0f;

    } catch (const std::exception& e) {
        error_handler.has_error = true;
        error_handler.error_message = "Initialization failed: " + std::string(e.what());
        throw;
    }
}

ProcessingResult LidarProcessor::processScan(const std::vector<float>& raw_scan_data) {
    std::lock_guard<std::mutex> lock(pipeline_mutex);
    ProcessingResult result;
    result.success = false;

    auto start_time = std::chrono::steady_clock::now();

    try {
        // Validate input data
        if (raw_scan_data.empty() || raw_scan_data.size() % 4 != 0) {
            throw std::invalid_argument("Invalid scan data format");
        }

        // Process point cloud (target: 33ms)
        auto point_cloud_start = std::chrono::steady_clock::now();
        
        for (size_t i = 0; i < raw_scan_data.size(); i += 4) {
            if (!point_cloud->addPoint(
                raw_scan_data[i],
                raw_scan_data[i + 1],
                raw_scan_data[i + 2],
                raw_scan_data[i + 3]
            )) {
                throw std::runtime_error("Failed to add point to cloud");
            }
        }

        point_cloud->filterNoise(2.0f);
        stats.point_cloud_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - point_cloud_start);

        // Detect features (target: 10ms)
        auto feature_start = std::chrono::steady_clock::now();
        auto features = feature_detector->detectFeatures().get();
        stats.feature_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - feature_start);

        // Classify surfaces (target: 7ms)
        auto surface_start = std::chrono::steady_clock::now();
        auto surfaces = surface_classifier->classifySurfaces();
        stats.surface_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - surface_start);

        // Update processing statistics
        stats.total_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - start_time);
        stats.points_processed = point_cloud->getRawPoints()->size();
        stats.features_detected = features.size();
        stats.surfaces_classified = surfaces.size();

        // Calculate average confidence
        float total_confidence = 0.0f;
        for (const auto& surface : surfaces) {
            total_confidence += surface.confidence;
        }
        stats.average_confidence = surfaces.empty() ? 0.0f : total_confidence / surfaces.size();

        // Prepare result
        result.point_cloud = point_cloud->getRawPoints();
        result.features = std::move(features);
        result.surfaces = std::move(surfaces);
        result.stats = stats;
        result.success = true;

        // Check processing time against requirements
        if (stats.total_time > std::chrono::milliseconds(MAX_PROCESSING_TIME_MS)) {
            optimizePipeline();
        }

    } catch (const std::exception& e) {
        error_handler.has_error = true;
        error_handler.error_message = e.what();
        error_handler.error_time = std::chrono::steady_clock::now();
        result.error_message = e.what();
    }

    return result;
}

void LidarProcessor::optimizePipeline() {
    // Update resource monitoring
    updateResourceMonitor();

    // Check if total processing time exceeds target
    if (stats.total_time > std::chrono::milliseconds(MAX_PROCESSING_TIME_MS)) {
        if (gpu_enabled) {
            // GPU optimization path
            if (resource_monitor.gpu_utilization > 0.9f) {
                // GPU is bottlenecked, adjust processing parameters
                feature_detector->setConfidenceThreshold(
                    feature_detector->getMetrics().average_confidence + 0.05f);
                surface_classifier->setConfidenceThreshold(
                    surface_classifier->isProcessingActive() ? 0.9f : 0.85f);
            }
        }
        
        // Check individual stage timings
        if (stats.point_cloud_time > std::chrono::milliseconds(33)) {
            // Point cloud processing optimization
            point_cloud->downsample(scan_resolution * 2.0f);
        }
        
        if (stats.feature_time > std::chrono::milliseconds(10)) {
            // Feature detection optimization
            feature_detector->setConfidenceThreshold(
                std::min(0.95f, feature_detector->getMetrics().average_confidence + 0.1f));
        }
        
        if (stats.surface_time > std::chrono::milliseconds(7)) {
            // Surface classification optimization
            surface_classifier->setConfidenceThreshold(
                std::min(0.95f, surface_classifier->isProcessingActive() ? 0.9f : 0.85f));
        }
    }
}

void LidarProcessor::updateResourceMonitor() {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        current_time - resource_monitor.last_update);

    if (elapsed > std::chrono::milliseconds(1000)) {
        if (gpu_enabled) {
            // Update GPU utilization
            float gpu_util;
            cudaDeviceGetAttribute(
                reinterpret_cast<int*>(&gpu_util),
                cudaDevAttrGpuUtilizationRate,
                0
            );
            resource_monitor.gpu_utilization = gpu_util / 100.0f;

            // Update GPU memory usage
            size_t free_memory, total_memory;
            cudaMemGetInfo(&free_memory, &total_memory);
            resource_monitor.memory_usage = 1.0f - (static_cast<float>(free_memory) / total_memory);
        }

        // Update processing load
        resource_monitor.processing_load = static_cast<float>(
            stats.total_time.count()) / (MAX_PROCESSING_TIME_MS * 1000.0f);

        resource_monitor.last_update = current_time;
    }
}

ProcessingStats LidarProcessor::getProcessingStats() const {
    return stats;
}
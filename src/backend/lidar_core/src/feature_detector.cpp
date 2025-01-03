// External dependencies
#include <pcl/point_cloud.h>          // PCL 1.12.0
#include <pcl/gpu/containers/device_array.h>  // PCL 1.12.0
#include <cuda_runtime.h>             // CUDA 12.0
#include <Eigen/Dense>                // Eigen 3.4.0

// Internal dependencies
#include "../include/feature_detector.hpp"
#include "../include/point_cloud.hpp"

// CUDA kernel declarations
namespace cuda {
    __global__ void detectPlanarFeaturesKernel(float* points, size_t num_points, Feature* features);
    __global__ void classifyFeaturesKernel(Feature* features, size_t num_features);
    __global__ void calculateConfidenceKernel(Feature* features, size_t num_features);
}

// Constants for CUDA processing
constexpr unsigned int CUDA_BLOCK_SIZE = 256;
constexpr unsigned int MAX_CUDA_STREAMS = 4;

std::future<std::vector<Feature>> FeatureDetector::detectFeatures(
    std::shared_ptr<PointCloud> point_cloud,
    cudaStream_t stream) noexcept {
    
    return std::async(std::launch::async, [this, point_cloud, stream]() {
        std::vector<Feature> features;
        if (!gpu_enabled || !point_cloud || point_cloud->getRawPoints()->empty()) {
            return features;
        }

        std::lock_guard<std::mutex> lock(feature_mutex);
        processing_active = true;
        auto start_time = std::chrono::steady_clock::now();

        try {
            // Initialize CUDA streams for parallel processing
            std::array<cudaStream_t, MAX_CUDA_STREAMS> streams;
            for (auto& s : streams) {
                cudaStreamCreate(&s);
            }

            // Allocate GPU memory from pool
            size_t point_count = point_cloud->getRawPoints()->size();
            void* gpu_points = memory_pool->allocate(point_count * sizeof(pcl::PointXYZI));
            void* gpu_features = memory_pool->allocate(MAX_FEATURES_PER_SCAN * sizeof(Feature));

            // Transfer point cloud to GPU asynchronously
            auto transfer_start = std::chrono::steady_clock::now();
            cudaMemcpyAsync(gpu_points, 
                          point_cloud->getRawPoints()->points.data(),
                          point_count * sizeof(pcl::PointXYZI),
                          cudaMemcpyHostToDevice,
                          streams[0]);

            // Launch feature detection kernels across multiple streams
            dim3 block(CUDA_BLOCK_SIZE);
            dim3 grid((point_count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

            for (size_t i = 0; i < MAX_CUDA_STREAMS; i++) {
                cuda::detectPlanarFeaturesKernel<<<grid, block, 0, streams[i]>>>(
                    static_cast<float*>(gpu_points),
                    point_count / MAX_CUDA_STREAMS,
                    static_cast<Feature*>(gpu_features) + (i * MAX_FEATURES_PER_SCAN / MAX_CUDA_STREAMS)
                );
            }

            // Synchronize streams before classification
            for (auto& s : streams) {
                cudaStreamSynchronize(s);
            }

            // Launch classification kernels
            for (size_t i = 0; i < MAX_CUDA_STREAMS; i++) {
                cuda::classifyFeaturesKernel<<<grid, block, 0, streams[i]>>>(
                    static_cast<Feature*>(gpu_features) + (i * MAX_FEATURES_PER_SCAN / MAX_CUDA_STREAMS),
                    MAX_FEATURES_PER_SCAN / MAX_CUDA_STREAMS
                );
            }

            // Calculate confidence scores
            cuda::calculateConfidenceKernel<<<grid, block, 0, stream>>>(
                static_cast<Feature*>(gpu_features),
                MAX_FEATURES_PER_SCAN
            );

            // Transfer results back to host
            features.resize(MAX_FEATURES_PER_SCAN);
            cudaMemcpyAsync(features.data(),
                          gpu_features,
                          MAX_FEATURES_PER_SCAN * sizeof(Feature),
                          cudaMemcpyDeviceToHost,
                          stream);

            // Clean up GPU resources
            memory_pool->deallocate(gpu_points);
            memory_pool->deallocate(gpu_features);
            for (auto& s : streams) {
                cudaStreamDestroy(s);
            }

            // Update performance metrics
            metrics.gpu_transfer_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - transfer_start);
            metrics.detection_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now() - start_time);
            metrics.features_detected = features.size();
            metrics.points_processed = point_count;

        } catch (const std::exception& e) {
            // Log error and return empty feature set
            processing_active = false;
            return std::vector<Feature>();
        }

        processing_active = false;
        return features;
    });
}

FeatureType FeatureDetector::classifyFeature(
    pcl::PointCloud<pcl::PointXYZI>& feature_points,
    cudaStream_t stream) noexcept {
    
    if (feature_points.empty() || feature_points.size() < MIN_FEATURE_POINTS) {
        return FeatureType::UNKNOWN;
    }

    try {
        // Allocate GPU memory for feature points
        void* gpu_feature_points = memory_pool->allocate(
            feature_points.size() * sizeof(pcl::PointXYZI));

        // Transfer feature points to GPU
        cudaMemcpyAsync(gpu_feature_points,
                       feature_points.points.data(),
                       feature_points.size() * sizeof(pcl::PointXYZI),
                       cudaMemcpyHostToDevice,
                       stream);

        // Extract geometric features
        Eigen::Vector3f normal, centroid;
        float planarity, linearity, sphericity;
        
        dim3 block(CUDA_BLOCK_SIZE);
        dim3 grid((feature_points.size() + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

        // Launch geometric analysis kernels
        // Note: Actual kernel implementations would be in separate CUDA files
        calculateGeometricFeatures<<<grid, block, 0, stream>>>(
            static_cast<float*>(gpu_feature_points),
            feature_points.size(),
            &normal,
            &centroid,
            &planarity,
            &linearity,
            &sphericity
        );

        // Classify based on geometric properties
        FeatureType type = FeatureType::UNKNOWN;
        float confidence = 0.0f;

        if (planarity > 0.8f) {
            if (std::abs(normal.dot(Eigen::Vector3f::UnitZ())) > 0.8f) {
                type = FeatureType::FLOOR;
                confidence = planarity;
            } else {
                type = FeatureType::WALL;
                confidence = planarity;
            }
        } else if (linearity > 0.8f) {
            type = FeatureType::EDGE;
            confidence = linearity;
        } else if (sphericity > 0.8f) {
            type = FeatureType::OBSTACLE;
            confidence = sphericity;
        }

        // Clean up GPU memory
        memory_pool->deallocate(gpu_feature_points);

        return (confidence >= confidence_threshold) ? type : FeatureType::UNKNOWN;

    } catch (const std::exception& e) {
        return FeatureType::UNKNOWN;
    }
}

// CUDA kernel implementations
namespace cuda {
    __global__ void detectPlanarFeaturesKernel(float* points, size_t num_points, Feature* features) {
        // Implementation of planar feature detection kernel
        // This would contain the actual CUDA kernel code for RANSAC-based
        // planar feature detection
    }

    __global__ void classifyFeaturesKernel(Feature* features, size_t num_features) {
        // Implementation of feature classification kernel
        // This would contain the actual CUDA kernel code for feature
        // classification using geometric properties
    }

    __global__ void calculateConfidenceKernel(Feature* features, size_t num_features) {
        // Implementation of confidence calculation kernel
        // This would contain the actual CUDA kernel code for calculating
        // confidence scores based on geometric properties
    }
}
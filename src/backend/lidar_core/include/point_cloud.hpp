#ifndef POINT_CLOUD_HPP
#define POINT_CLOUD_HPP

// External dependencies
#include <pcl/point_cloud.h>      // PCL 1.12.0
#include <pcl/point_types.h>      // PCL 1.12.0
#include <cuda_runtime.h>         // CUDA 12.0
#include <Eigen/Dense>            // Eigen 3.4.0

// Standard library includes
#include <vector>
#include <atomic>
#include <mutex>
#include <stdexcept>

// Global constants
constexpr unsigned int CUDA_BLOCK_SIZE = 256;
constexpr unsigned int MAX_POINTS_PER_CLOUD = 1000000;

// Minimum resolution and maximum range constants based on hardware specs
constexpr float MIN_RESOLUTION = 0.0001f;  // 0.01cm in meters
constexpr float MAX_RANGE = 5.0f;          // 5.0m maximum range

class PointCloud {
public:
    /**
     * @brief Constructs a new Point Cloud object with specified resolution and range
     * @param resolution Minimum distance between points in meters (>= 0.01cm)
     * @param range Maximum scanning range in meters (<= 5.0m)
     * @throws std::invalid_argument if parameters are out of valid ranges
     */
    PointCloud(float resolution, float range) 
        : resolution(resolution)
        , range(range)
        , gpu_enabled(false)
        , processing_active(false) {
        
        if (resolution < MIN_RESOLUTION) {
            throw std::invalid_argument("Resolution must be >= 0.01cm");
        }
        if (range > MAX_RANGE) {
            throw std::invalid_argument("Range must be <= 5.0m");
        }

        // Initialize PCL point cloud
        raw_points = pcl::PointCloud<pcl::PointXYZI>::Ptr(
            new pcl::PointCloud<pcl::PointXYZI>());
        
        // Initialize transformation matrix to identity
        transform_matrix.setIdentity();

        // Initialize CUDA resources
        cudaError_t cuda_status = cudaStreamCreate(&cuda_stream);
        if (cuda_status == cudaSuccess) {
            gpu_enabled = true;
            // Allocate GPU memory for points and intensities
            cuda_status = cudaMalloc(&d_points, 
                MAX_POINTS_PER_CLOUD * 3 * sizeof(float));
            if (cuda_status == cudaSuccess) {
                cuda_status = cudaMalloc(&d_intensities, 
                    MAX_POINTS_PER_CLOUD * sizeof(float));
                if (cuda_status != cudaSuccess) {
                    cudaFree(d_points);
                    gpu_enabled = false;
                }
            } else {
                gpu_enabled = false;
            }
        }
    }

    /**
     * @brief Destructor to clean up CUDA resources
     */
    ~PointCloud() {
        if (gpu_enabled) {
            cudaFree(d_points);
            cudaFree(d_intensities);
            cudaStreamDestroy(cuda_stream);
        }
    }

    /**
     * @brief Adds a new point to the cloud
     * @param x X coordinate in meters
     * @param y Y coordinate in meters
     * @param z Z coordinate in meters
     * @param intensity Point intensity value
     * @return true if point was successfully added
     */
    bool addPoint(float x, float y, float z, float intensity) {
        std::lock_guard<std::mutex> lock(point_mutex);
        
        // Validate point coordinates
        if (std::abs(x) > range || std::abs(y) > range || std::abs(z) > range) {
            return false;
        }

        // Check point cloud capacity
        if (raw_points->size() >= MAX_POINTS_PER_CLOUD) {
            return false;
        }

        // Round coordinates to resolution grid
        x = std::round(x / resolution) * resolution;
        y = std::round(y / resolution) * resolution;
        z = std::round(z / resolution) * resolution;

        // Create and add point
        pcl::PointXYZI point;
        point.x = x;
        point.y = y;
        point.z = z;
        point.intensity = intensity;
        
        raw_points->push_back(point);
        intensities.push_back(intensity);

        // Queue point for GPU transfer if enabled
        if (gpu_enabled) {
            size_t point_idx = raw_points->size() - 1;
            cudaMemcpyAsync(&d_points[point_idx * 3], &x, 
                3 * sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
            cudaMemcpyAsync(&d_intensities[point_idx], &intensity, 
                sizeof(float), cudaMemcpyHostToDevice, cuda_stream);
        }

        return true;
    }

    /**
     * @brief Performs statistical noise filtering using GPU acceleration
     * @param stddev_mult Standard deviation multiplier for outlier detection
     */
    void filterNoise(float stddev_mult) {
        if (!gpu_enabled || raw_points->empty()) {
            return;
        }

        processing_active = true;

        // GPU kernel launches for noise filtering would be implemented here
        // This is a placeholder for the actual CUDA kernel calls
        
        processing_active = false;
    }

    /**
     * @brief Downsamples the point cloud using voxel grid filtering
     * @param leaf_size Size of voxel grid cells
     * @throws std::invalid_argument if leaf_size is smaller than resolution
     */
    void downsample(float leaf_size) {
        if (leaf_size < resolution) {
            throw std::invalid_argument("Leaf size must be >= resolution");
        }

        if (!gpu_enabled || raw_points->empty()) {
            return;
        }

        processing_active = true;

        // GPU kernel launches for downsampling would be implemented here
        // This is a placeholder for the actual CUDA kernel calls

        processing_active = false;
    }

    /**
     * @brief Applies geometric transformation to the point cloud
     * @param transform 4x4 transformation matrix
     * @throws std::invalid_argument if transform matrix is singular
     */
    void transform(const Eigen::Matrix4f& transform) {
        if (std::abs(transform.determinant()) < 1e-6) {
            throw std::invalid_argument("Transform matrix is singular");
        }

        if (!gpu_enabled || raw_points->empty()) {
            return;
        }

        processing_active = true;

        // GPU kernel launches for transformation would be implemented here
        // This is a placeholder for the actual CUDA kernel calls

        transform_matrix = transform * transform_matrix;
        
        processing_active = false;
    }

    // Public accessors
    pcl::PointCloud<pcl::PointXYZI>::Ptr getRawPoints() const { return raw_points; }
    Eigen::Matrix4f getTransformMatrix() const { return transform_matrix; }
    float getResolution() const { return resolution; }
    float getRange() const { return range; }
    bool isProcessingActive() const { return processing_active; }

private:
    // Point cloud data
    pcl::PointCloud<pcl::PointXYZI>::Ptr raw_points;
    Eigen::Matrix4f transform_matrix;
    std::vector<float> intensities;

    // Configuration parameters
    float resolution;
    float range;

    // GPU processing resources
    bool gpu_enabled;
    cudaStream_t cuda_stream;
    float* d_points;
    float* d_intensities;

    // Thread safety
    std::atomic<bool> processing_active;
    std::mutex point_mutex;
};

#endif // POINT_CLOUD_HPP
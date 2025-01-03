// External dependencies
#include <pcl/point_cloud.h>      // PCL 1.12.0
#include <pcl/point_types.h>      // PCL 1.12.0
#include <cuda_runtime.h>         // CUDA 12.0
#include <cuda_profiler_api.h>    // CUDA 12.0
#include <Eigen/Dense>            // Eigen 3.4.0

// Internal includes
#include "../include/point_cloud.hpp"

// Standard library includes
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <cmath>

// CUDA kernel declarations
__global__ void statisticalNoiseFilterKernel(float* points, float* intensities, 
    bool* outliers, int num_points, float stddev_mult);

__global__ void voxelGridDownsampleKernel(float* points, float* intensities,
    int* voxel_indices, float leaf_size, int num_points);

__global__ void transformPointsKernel(float* points, float* matrix, 
    int num_points);

// Global constants for GPU processing
constexpr unsigned int PINNED_MEMORY_SIZE = 67108864; // 64MB
constexpr unsigned int MAX_CUDA_STREAMS = 4;

// Implementation of PointCloud constructor
PointCloud::PointCloud(float resolution, float range) 
    : resolution(resolution)
    , range(range)
    , gpu_enabled(false)
    , processing_active(false) {
    
    // Validate input parameters
    if (resolution < MIN_RESOLUTION) {
        throw std::invalid_argument("Resolution must be >= 0.01cm");
    }
    if (range > MAX_RANGE) {
        throw std::invalid_argument("Range must be <= 5.0m");
    }

    // Initialize PCL point cloud
    raw_points = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>());
    
    // Initialize transformation matrix
    transform_matrix.setIdentity();

    // Initialize CUDA resources with error checking
    cudaError_t cuda_status;

    // Create CUDA streams for async operations
    for (int i = 0; i < MAX_CUDA_STREAMS; ++i) {
        cudaStream_t stream;
        cuda_status = cudaStreamCreate(&stream);
        if (cuda_status != cudaSuccess) {
            // Clean up previously created streams
            for (int j = 0; j < i; ++j) {
                cudaStreamDestroy(cuda_streams[j]);
            }
            throw std::runtime_error("Failed to create CUDA stream");
        }
        cuda_streams[i] = stream;
    }

    // Allocate pinned memory for efficient GPU transfers
    cuda_status = cudaMallocHost(&h_pinned_buffer, PINNED_MEMORY_SIZE);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate pinned memory");
    }

    // Allocate GPU memory for point cloud data
    cuda_status = cudaMalloc(&d_points, MAX_POINTS_PER_CLOUD * 3 * sizeof(float));
    if (cuda_status == cudaSuccess) {
        cuda_status = cudaMalloc(&d_intensities, MAX_POINTS_PER_CLOUD * sizeof(float));
        if (cuda_status == cudaSuccess) {
            gpu_enabled = true;
        } else {
            cudaFree(d_points);
        }
    }

    if (!gpu_enabled) {
        throw std::runtime_error("Failed to initialize GPU resources");
    }

    // Initialize CUDA profiler
    cudaProfilerStart();
}

bool PointCloud::addPoint(float x, float y, float z, float intensity) {
    std::lock_guard<std::mutex> lock(point_mutex);
    
    // Validate point coordinates against range
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

    // Add point to PCL cloud
    pcl::PointXYZI point;
    point.x = x;
    point.y = y;
    point.z = z;
    point.intensity = intensity;
    
    raw_points->push_back(point);

    // Queue point for GPU transfer using pinned memory
    if (gpu_enabled) {
        size_t point_idx = raw_points->size() - 1;
        size_t offset = point_idx * 3;
        
        // Copy to pinned memory buffer
        float* pinned_ptr = reinterpret_cast<float*>(h_pinned_buffer);
        pinned_ptr[offset] = x;
        pinned_ptr[offset + 1] = y;
        pinned_ptr[offset + 2] = z;
        pinned_ptr[MAX_POINTS_PER_CLOUD * 3 + point_idx] = intensity;

        // Async transfer to GPU using round-robin stream selection
        cudaStream_t& stream = cuda_streams[point_idx % MAX_CUDA_STREAMS];
        cudaMemcpyAsync(&d_points[offset], &pinned_ptr[offset],
            3 * sizeof(float), cudaMemcpyHostToDevice, stream);
        cudaMemcpyAsync(&d_intensities[point_idx], &pinned_ptr[MAX_POINTS_PER_CLOUD * 3 + point_idx],
            sizeof(float), cudaMemcpyHostToDevice, stream);
    }

    return true;
}

void PointCloud::filterNoise(float stddev_mult) {
    if (!gpu_enabled || raw_points->empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(point_mutex);
    processing_active = true;

    int num_points = raw_points->size();
    
    // Allocate device memory for outlier flags
    bool* d_outliers;
    cudaMalloc(&d_outliers, num_points * sizeof(bool));

    // Calculate grid dimensions
    dim3 block(CUDA_BLOCK_SIZE);
    dim3 grid((num_points + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    // Launch statistical noise filter kernel
    cudaStream_t& stream = cuda_streams[0];
    statisticalNoiseFilterKernel<<<grid, block, 0, stream>>>(
        d_points, d_intensities, d_outliers, num_points, stddev_mult);

    // Allocate host memory for filtered points
    std::vector<bool> outliers(num_points);
    cudaMemcpyAsync(outliers.data(), d_outliers, 
        num_points * sizeof(bool), cudaMemcpyDeviceToHost, stream);

    // Wait for GPU operations to complete
    cudaStreamSynchronize(stream);

    // Remove outlier points
    auto new_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < num_points; ++i) {
        if (!outliers[i]) {
            new_cloud->push_back(raw_points->points[i]);
        }
    }

    // Update point cloud
    raw_points = new_cloud;

    // Clean up
    cudaFree(d_outliers);
    processing_active = false;
}

void PointCloud::downsample(float leaf_size) {
    if (leaf_size < resolution) {
        throw std::invalid_argument("Leaf size must be >= resolution");
    }

    if (!gpu_enabled || raw_points->empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(point_mutex);
    processing_active = true;

    int num_points = raw_points->size();

    // Allocate device memory for voxel grid
    int* d_voxel_indices;
    cudaMalloc(&d_voxel_indices, num_points * sizeof(int));

    // Calculate grid dimensions
    dim3 block(CUDA_BLOCK_SIZE);
    dim3 grid((num_points + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    // Launch voxel grid downsample kernel
    cudaStream_t& stream = cuda_streams[1];
    voxelGridDownsampleKernel<<<grid, block, 0, stream>>>(
        d_points, d_intensities, d_voxel_indices, leaf_size, num_points);

    // Allocate host memory for voxel indices
    std::vector<int> voxel_indices(num_points);
    cudaMemcpyAsync(voxel_indices.data(), d_voxel_indices,
        num_points * sizeof(int), cudaMemcpyDeviceToHost, stream);

    // Wait for GPU operations to complete
    cudaStreamSynchronize(stream);

    // Create downsampled point cloud
    auto downsampled_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(
        new pcl::PointCloud<pcl::PointXYZI>);
    
    // Use voxel indices to compute centroids
    std::unordered_map<int, std::vector<pcl::PointXYZI>> voxel_map;
    for (int i = 0; i < num_points; ++i) {
        voxel_map[voxel_indices[i]].push_back(raw_points->points[i]);
    }

    // Compute centroids for each voxel
    for (const auto& voxel : voxel_map) {
        pcl::PointXYZI centroid;
        centroid.x = 0;
        centroid.y = 0;
        centroid.z = 0;
        centroid.intensity = 0;

        for (const auto& point : voxel.second) {
            centroid.x += point.x;
            centroid.y += point.y;
            centroid.z += point.z;
            centroid.intensity += point.intensity;
        }

        float inv_size = 1.0f / voxel.second.size();
        centroid.x *= inv_size;
        centroid.y *= inv_size;
        centroid.z *= inv_size;
        centroid.intensity *= inv_size;

        downsampled_cloud->push_back(centroid);
    }

    // Update point cloud
    raw_points = downsampled_cloud;

    // Clean up
    cudaFree(d_voxel_indices);
    processing_active = false;
}

void PointCloud::transform(const Eigen::Matrix4f& transform) {
    if (std::abs(transform.determinant()) < 1e-6) {
        throw std::invalid_argument("Transform matrix is singular");
    }

    if (!gpu_enabled || raw_points->empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(point_mutex);
    processing_active = true;

    int num_points = raw_points->size();

    // Copy transformation matrix to device
    float* d_matrix;
    cudaMalloc(&d_matrix, 16 * sizeof(float));
    cudaMemcpyAsync(d_matrix, transform.data(),
        16 * sizeof(float), cudaMemcpyHostToDevice, cuda_streams[2]);

    // Calculate grid dimensions
    dim3 block(CUDA_BLOCK_SIZE);
    dim3 grid((num_points + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE);

    // Launch transform kernel
    transformPointsKernel<<<grid, block, 0, cuda_streams[2]>>>(
        d_points, d_matrix, num_points);

    // Copy transformed points back to host
    cudaMemcpyAsync(h_pinned_buffer, d_points,
        num_points * 3 * sizeof(float), cudaMemcpyDeviceToHost, cuda_streams[2]);

    // Wait for GPU operations to complete
    cudaStreamSynchronize(cuda_streams[2]);

    // Update PCL point cloud with transformed points
    float* transformed_points = reinterpret_cast<float*>(h_pinned_buffer);
    for (int i = 0; i < num_points; ++i) {
        raw_points->points[i].x = transformed_points[i * 3];
        raw_points->points[i].y = transformed_points[i * 3 + 1];
        raw_points->points[i].z = transformed_points[i * 3 + 2];
    }

    // Update transformation matrix
    transform_matrix = transform * transform_matrix;

    // Clean up
    cudaFree(d_matrix);
    processing_active = false;
}

// Destructor implementation
PointCloud::~PointCloud() {
    if (gpu_enabled) {
        // Wait for all operations to complete
        for (int i = 0; i < MAX_CUDA_STREAMS; ++i) {
            cudaStreamSynchronize(cuda_streams[i]);
            cudaStreamDestroy(cuda_streams[i]);
        }

        // Free GPU resources
        cudaFree(d_points);
        cudaFree(d_intensities);
        cudaFreeHost(h_pinned_buffer);
    }

    // Stop CUDA profiler
    cudaProfilerStop();
}
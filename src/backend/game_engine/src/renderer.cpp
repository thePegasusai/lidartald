#include "renderer.hpp"

// External dependencies with versions
#include <vulkan/vulkan.h>      // v1.3.0
#include <glm/glm.hpp>          // v0.9.9.8
#include <SPIRV/SPIRV.h>        // v1.6

// Standard library includes
#include <chrono>
#include <algorithm>
#include <stdexcept>
#include <array>
#include <thread>

Renderer::Renderer(const RenderConfig& config) noexcept : config(config) {
    try {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createPointCloudPipeline();
        createFramebuffers();
        createCommandPool();
        createPointCloudResources();
        createSyncObjects();
        createPerformanceQueryPool();

        // Initialize GPU memory pool if enabled
        if (config.useGPUMemoryPool) {
            VkDeviceSize poolSize = config.maxPointCloudSize * sizeof(float) * 4;
            VkBufferCreateInfo bufferInfo{};
            bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bufferInfo.size = poolSize;
            bufferInfo.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | 
                             VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            if (vkCreateBuffer(device, &bufferInfo, nullptr, &pointCloudVertexBuffer) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create point cloud vertex buffer");
            }

            VkMemoryRequirements memRequirements;
            vkGetBufferMemoryRequirements(device, pointCloudVertexBuffer, &memRequirements);

            VkMemoryAllocateInfo allocInfo{};
            allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
            allocInfo.allocationSize = memRequirements.size;
            allocInfo.memoryTypeIndex = findMemoryType(
                memRequirements.memoryTypeBits,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            );

            if (vkAllocateMemory(device, &allocInfo, nullptr, &pointCloudMemoryPool) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate point cloud memory pool");
            }

            vkBindBufferMemory(device, pointCloudVertexBuffer, pointCloudMemoryPool, 0);
        }
    } catch (const std::exception& e) {
        cleanup();
        throw;
    }
}

void Renderer::render(const GameState& gameState, const PointCloud& pointCloud) noexcept {
    try {
        // Begin performance query
        vkCmdBeginQuery(commandBuffers[currentFrame], performanceQueryPool, 0, 0);

        // Wait for previous frame
        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // Acquire next swapchain image
        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX,
            imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);

        if (result == VK_ERROR_OUT_OF_DATE_KHR) {
            recreateSwapChain();
            return;
        } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {
            throw std::runtime_error("Failed to acquire swapchain image");
        }

        // Update uniform buffers
        updateUniformBuffer(currentFrame, gameState);

        // Thread-safe point cloud update
        {
            std::lock_guard<std::mutex> lock(renderMutex);
            updatePointCloudBuffer(pointCloud);
        }

        // Reset and begin command buffer
        vkResetCommandBuffer(commandBuffers[currentFrame], 0);
        
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        if (vkBeginCommandBuffer(commandBuffers[currentFrame], &beginInfo) != VK_SUCCESS) {
            throw std::runtime_error("Failed to begin recording command buffer");
        }

        // Begin render pass
        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffers[imageIndex];
        renderPassInfo.renderArea.offset = {0, 0};
        renderPassInfo.renderArea.extent = swapchainExtent;

        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = {{0.0f, 0.0f, 0.0f, 1.0f}};
        clearValues[1].depthStencil = {1.0f, 0};
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffers[currentFrame], &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);

        // Bind point cloud pipeline and draw
        vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, pointCloudPipeline);
        VkBuffer vertexBuffers[] = {pointCloudVertexBuffer};
        VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(commandBuffers[currentFrame], 0, 1, vertexBuffers, offsets);
        vkCmdBindDescriptorSets(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        vkCmdDraw(commandBuffers[currentFrame], pointCloud.getPointCount(), 1, 0, 0);

        // Bind game object pipeline and draw
        vkCmdBindPipeline(commandBuffers[currentFrame], VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        // Draw game objects here...

        vkCmdEndRenderPass(commandBuffers[currentFrame]);

        // End performance query
        vkCmdEndQuery(commandBuffers[currentFrame], performanceQueryPool, 0);

        if (vkEndCommandBuffer(commandBuffers[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to record command buffer");
        }

        // Submit command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

        VkSemaphore waitSemaphores[] = {imageAvailableSemaphores[currentFrame]};
        VkPipelineStageFlags waitStages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];

        VkSemaphore signalSemaphores[] = {renderFinishedSemaphores[currentFrame]};
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);

        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to submit draw command buffer");
        }

        // Present rendered image
        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;

        VkSwapchainKHR swapChains[] = {swapchain};
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;

        result = vkQueuePresentKHR(presentQueue, &presentInfo);

        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR) {
            recreateSwapChain();
        } else if (result != VK_SUCCESS) {
            throw std::runtime_error("Failed to present swapchain image");
        }

        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Frame pacing for VSync
        if (config.vsync) {
            std::this_thread::sleep_until(lastFrameTime + 
                std::chrono::microseconds(1000000 / TARGET_FRAME_RATE));
        }
        lastFrameTime = std::chrono::high_resolution_clock::now();

    } catch (const std::exception& e) {
        // Log error and attempt recovery
        std::cerr << "Render error: " << e.what() << std::endl;
        recreateSwapChain();
    }
}

void Renderer::updatePointCloud(const PointCloud& pointCloud) {
    try {
        std::lock_guard<std::mutex> lock(renderMutex);
        
        const auto& points = pointCloud.getRawPoints();
        if (!points || points->empty()) {
            return;
        }

        VkDeviceSize bufferSize = points->size() * sizeof(float) * 4;
        
        // Create staging buffer
        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        createBuffer(bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            stagingBuffer, stagingBufferMemory);

        // Map memory and copy point cloud data
        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, points->points.data(), bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);

        // Copy to device local buffer
        copyBuffer(stagingBuffer, pointCloudVertexBuffer, bufferSize);

        // Cleanup staging buffer
        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);

    } catch (const std::exception& e) {
        std::cerr << "Point cloud update error: " << e.what() << std::endl;
    }
}

void Renderer::cleanup() {
    vkDeviceWaitIdle(device);

    // Cleanup synchronization objects
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
        vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
        vkDestroyFence(device, inFlightFences[i], nullptr);
    }

    // Cleanup point cloud resources
    if (pointCloudVertexBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, pointCloudVertexBuffer, nullptr);
    }
    if (pointCloudMemoryPool != VK_NULL_HANDLE) {
        vkFreeMemory(device, pointCloudMemoryPool, nullptr);
    }

    // Cleanup pipeline resources
    vkDestroyPipeline(device, graphicsPipeline, nullptr);
    vkDestroyPipeline(device, pointCloudPipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyRenderPass(device, renderPass, nullptr);

    // Cleanup swapchain
    for (auto framebuffer : framebuffers) {
        vkDestroyFramebuffer(device, framebuffer, nullptr);
    }
    for (auto imageView : swapchainImageViews) {
        vkDestroyImageView(device, imageView, nullptr);
    }
    vkDestroySwapchainKHR(device, swapchain, nullptr);

    // Cleanup device
    vkDestroyDevice(device, nullptr);

    // Cleanup instance
    vkDestroyInstance(instance, nullptr);
}

Renderer::~Renderer() {
    cleanup();
}
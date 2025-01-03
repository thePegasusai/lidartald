#ifndef RENDERER_HPP
#define RENDERER_HPP

// External dependencies
#include <vulkan/vulkan.h>      // v1.3.0
#include <glm/glm.hpp>          // v0.9.9.8
#include <SPIRV/SPIRV.h>        // v1.6

// Internal dependencies
#include "point_cloud.hpp"

// Standard library includes
#include <vector>
#include <memory>
#include <stdexcept>
#include <array>

// Global constants
constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
constexpr uint32_t TARGET_FRAME_RATE = 60;
constexpr uint32_t VULKAN_VERSION = VK_API_VERSION_1_3;
constexpr uint32_t MAX_POINT_CLOUD_SIZE = 1000000;
constexpr uint32_t POINT_CLOUD_UPDATE_RATE = 30;

/**
 * @brief Configuration structure for renderer initialization
 */
class RenderConfig {
public:
    uint32_t width{1920};
    uint32_t height{1080};
    bool vsync{true};
    bool enableValidation{
#ifdef NDEBUG
        false
#else
        true
#endif
    };
    VkFormat colorFormat{VK_FORMAT_B8G8R8A8_SRGB};
    VkFormat depthFormat{VK_FORMAT_D32_SFLOAT};
    uint32_t maxPointCloudSize{MAX_POINT_CLOUD_SIZE};
    uint32_t pointCloudUpdateRate{POINT_CLOUD_UPDATE_RATE};
    bool useGPUMemoryPool{true};
};

/**
 * @brief Core rendering class with point cloud visualization pipeline
 */
class Renderer {
public:
    /**
     * @brief Initializes the Vulkan renderer
     * @param config Rendering configuration parameters
     * @throws std::runtime_error if Vulkan initialization fails
     */
    explicit Renderer(const RenderConfig& config);

    // Disable copying
    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    /**
     * @brief Renders a frame with point cloud visualization
     * @param gameState Current game state for rendering
     * @param pointCloud Point cloud data to visualize
     * @throws std::runtime_error if rendering fails
     */
    void render(const struct GameState& gameState, const PointCloud& pointCloud);

    /**
     * @brief Updates point cloud data in GPU memory
     * @param pointCloud New point cloud data
     * @throws std::runtime_error if update fails
     */
    void updatePointCloud(const PointCloud& pointCloud);

    /**
     * @brief Cleans up Vulkan resources
     */
    void cleanup();

    /**
     * @brief Destructor ensuring proper cleanup
     */
    ~Renderer();

private:
    // Vulkan instance and device
    VkInstance instance{VK_NULL_HANDLE};
    VkDevice device{VK_NULL_HANDLE};
    VkPhysicalDevice physicalDevice{VK_NULL_HANDLE};
    VkQueue graphicsQueue{VK_NULL_HANDLE};
    VkQueue presentQueue{VK_NULL_HANDLE};

    // Swapchain resources
    VkSwapchainKHR swapchain{VK_NULL_HANDLE};
    std::vector<VkImage> swapchainImages;
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainImageFormat;
    VkExtent2D swapchainExtent;

    // Render pass and pipeline
    VkRenderPass renderPass{VK_NULL_HANDLE};
    std::vector<VkFramebuffer> framebuffers;
    VkPipelineLayout pipelineLayout{VK_NULL_HANDLE};
    VkPipeline graphicsPipeline{VK_NULL_HANDLE};
    VkPipeline pointCloudPipeline{VK_NULL_HANDLE};

    // Point cloud resources
    VkBuffer pointCloudVertexBuffer{VK_NULL_HANDLE};
    VkDeviceMemory pointCloudMemoryPool{VK_NULL_HANDLE};
    VkDescriptorPool descriptorPool{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> descriptorSets;

    // Synchronization primitives
    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    uint32_t currentFrame{0};

    // Performance monitoring
    VkQueryPool performanceQueryPool{VK_NULL_HANDLE};
    
    // Configuration
    RenderConfig config;

    // Initialization functions
    void createInstance();
    void setupDebugMessenger();
    void createSurface();
    void pickPhysicalDevice();
    void createLogicalDevice();
    void createSwapChain();
    void createImageViews();
    void createRenderPass();
    void createDescriptorSetLayout();
    void createGraphicsPipeline();
    void createPointCloudPipeline();
    void createFramebuffers();
    void createCommandPool();
    void createPointCloudResources();
    void createSyncObjects();
    void createPerformanceQueryPool();

    // Helper functions
    bool checkValidationLayerSupport();
    std::vector<const char*> getRequiredExtensions();
    void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo);
    bool isDeviceSuitable(VkPhysicalDevice device);
    QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device);
    bool checkDeviceExtensionSupport(VkPhysicalDevice device);
    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device);
    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats);
    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes);
    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities);
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    void createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);
    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size);
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void transitionImageLayout(VkImage image, VkFormat format, VkImageLayout oldLayout, VkImageLayout newLayout);
    void copyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height);
};

#endif // RENDERER_HPP
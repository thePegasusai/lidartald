import { Router, Request, Response } from 'express'; // express@4.18.x
import { z } from 'zod'; // zod@3.x
import { createRateLimiter } from '../../middleware/rate-limiter';
import { LidarProcessor } from '../../../../lidar_core/include/lidar_processor';

// Constants from technical specifications
const SCAN_RATE_LIMIT = 30; // 30Hz scan rate
const MAX_SCAN_SIZE_MB = 10;
const DEFAULT_RESOLUTION = 0.01; // 0.01cm resolution
const MAX_RANGE_METERS = 5.0; // 5.0m maximum range
const PROCESSING_TIMEOUT_MS = 50; // 50ms total latency budget
const GPU_MEMORY_LIMIT_MB = 2048;

// Request validation schemas
const ScanRequestSchema = z.object({
  raw_data: z.instanceof(Buffer).refine(
    (data) => data.length <= MAX_SCAN_SIZE_MB * 1024 * 1024,
    'Scan data exceeds maximum size limit'
  ),
  resolution: z.number()
    .min(DEFAULT_RESOLUTION)
    .default(DEFAULT_RESOLUTION),
  range: z.number()
    .max(MAX_RANGE_METERS)
    .default(MAX_RANGE_METERS),
  use_gpu: z.boolean().default(true),
  compression_type: z.enum(['none', 'lz4', 'zstd']).default('none'),
  encryption_type: z.enum(['aes256', 'none']).default('aes256')
});

// Initialize router with rate limiting
const router = Router();
const rateLimiter = createRateLimiter({
  windowMs: 60000,
  maxRequests: SCAN_RATE_LIMIT,
  keyPrefix: 'lidar',
  endpointPattern: 'process|metrics',
  enableFallback: true,
  redisConfig: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    tls: process.env.NODE_ENV === 'production'
  },
  metrics: {
    enabled: true,
    prefix: 'lidar_api'
  }
});

// Initialize LiDAR processor
const lidarProcessor = new LidarProcessor(
  DEFAULT_RESOLUTION,
  MAX_RANGE_METERS,
  true // Enable GPU by default
);

/**
 * Process LiDAR scan data with GPU acceleration
 * Rate limit: 30 requests per minute
 */
router.post('/process', rateLimiter, async (req: Request, res: Response) => {
  try {
    // Validate request body
    const validatedData = ScanRequestSchema.parse(req.body);

    // Initialize performance monitoring
    const startTime = process.hrtime();

    // Decrypt scan data if encrypted
    let scanData = validatedData.raw_data;
    if (validatedData.encryption_type === 'aes256') {
      scanData = await decryptScanData(scanData);
    }

    // Decompress scan data if compressed
    if (validatedData.compression_type !== 'none') {
      scanData = await decompressScanData(scanData, validatedData.compression_type);
    }

    // Process scan through LiDAR pipeline
    const result = await lidarProcessor.processScan({
      raw_data: scanData,
      resolution: validatedData.resolution,
      range: validatedData.range,
      use_gpu: validatedData.use_gpu
    });

    // Calculate processing time
    const [seconds, nanoseconds] = process.hrtime(startTime);
    const processingTime = seconds * 1000 + nanoseconds / 1000000;

    // Validate processing time against latency budget
    if (processingTime > PROCESSING_TIMEOUT_MS) {
      req.app.get('metrics')?.incrementCounter('lidar_processing_timeout', {
        processing_time: processingTime
      });
    }

    // Format successful response
    const response = {
      success: true,
      processing_time_ms: processingTime,
      point_cloud: {
        size: result.point_cloud.size(),
        resolution: validatedData.resolution,
        range: validatedData.range
      },
      features: {
        count: result.features.length,
        types: countFeatureTypes(result.features)
      },
      surfaces: {
        count: result.surfaces.length,
        types: countSurfaceTypes(result.surfaces)
      },
      stats: result.stats
    };

    // Set security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');

    // Send response
    return res.status(200).json(response);

  } catch (error) {
    handleProcessingError(error, req, res);
  }
});

/**
 * Get LiDAR processing metrics and statistics
 * Rate limit: 30 requests per minute
 */
router.get('/metrics', rateLimiter, async (req: Request, res: Response) => {
  try {
    const stats = lidarProcessor.getProcessingStats();
    
    // Format metrics response
    const response = {
      success: true,
      timestamp: new Date().toISOString(),
      processing_stats: {
        total_time_us: stats.total_time.count(),
        point_cloud_time_us: stats.point_cloud_time.count(),
        feature_time_us: stats.feature_time.count(),
        surface_time_us: stats.surface_time.count(),
        points_processed: stats.points_processed,
        features_detected: stats.features_detected,
        surfaces_classified: stats.surfaces_classified,
        average_confidence: stats.average_confidence
      },
      resource_usage: {
        gpu_enabled: lidarProcessor.isGpuEnabled(),
        resolution: lidarProcessor.getResolution(),
        range: lidarProcessor.getRange()
      }
    };

    // Set cache control headers
    res.setHeader('Cache-Control', 'private, max-age=5');
    
    // Set security headers
    res.setHeader('X-Content-Type-Options', 'nosniff');
    res.setHeader('X-Frame-Options', 'DENY');
    res.setHeader('X-XSS-Protection', '1; mode=block');

    return res.status(200).json(response);

  } catch (error) {
    handleProcessingError(error, req, res);
  }
});

// Helper functions
async function decryptScanData(data: Buffer): Promise<Buffer> {
  // Implementation of AES-256-GCM decryption
  return data; // Placeholder
}

async function decompressScanData(data: Buffer, type: string): Promise<Buffer> {
  // Implementation of LZ4/ZSTD decompression
  return data; // Placeholder
}

function countFeatureTypes(features: any[]): Record<string, number> {
  return features.reduce((acc, feature) => {
    acc[feature.type] = (acc[feature.type] || 0) + 1;
    return acc;
  }, {});
}

function countSurfaceTypes(surfaces: any[]): Record<string, number> {
  return surfaces.reduce((acc, surface) => {
    acc[surface.type] = (acc[surface.type] || 0) + 1;
    return acc;
  }, {});
}

function handleProcessingError(error: any, req: Request, res: Response): void {
  const errorResponse = {
    success: false,
    error: error.message,
    code: error.code || 500,
    timestamp: new Date().toISOString()
  };

  // Log error
  req.app.get('logger')?.error({
    message: error.message,
    stack: error.stack,
    endpoint: req.path,
    method: req.method
  });

  // Track error metrics
  req.app.get('metrics')?.incrementCounter('lidar_processing_errors', {
    error_type: error.name,
    endpoint: req.path
  });

  // Set security headers
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('X-Frame-Options', 'DENY');
  res.setHeader('X-XSS-Protection', '1; mode=block');

  res.status(error.code || 500).json(errorResponse);
}

export default router;
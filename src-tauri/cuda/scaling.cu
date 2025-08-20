#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Bicubic interpolation coefficients
__device__ float bicubic_weight(float x) {
    x = fabsf(x);
    if (x <= 1.0f) {
        return 1.5f * x * x * x - 2.5f * x * x + 1.0f;
    } else if (x <= 2.0f) {
        return -0.5f * x * x * x + 2.5f * x * x - 4.0f * x + 2.0f;
    }
    return 0.0f;
}

// Lanczos filter for high-quality scaling
__device__ float lanczos_weight(float x, int radius) {
    if (x == 0.0f) return 1.0f;
    if (fabsf(x) >= radius) return 0.0f;
    
    float pi_x = CUDART_PI_F * x;
    float pi_x_over_radius = pi_x / radius;
    
    return (sinf(pi_x) / pi_x) * (sinf(pi_x_over_radius) / pi_x_over_radius);
}

// Advanced bicubic scaling kernel
__global__ void bicubic_scale_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    // Map destination pixel to source coordinates
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    int src_x_int = (int)floorf(src_x);
    int src_y_int = (int)floorf(src_y);
    
    float dx = src_x - src_x_int;
    float dy = src_y - src_y_int;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        // 4x4 bicubic convolution
        for (int j = -1; j <= 2; j++) {
            for (int i = -1; i <= 2; i++) {
                int sample_x = src_x_int + i;
                int sample_y = src_y_int + j;
                
                // Handle border conditions with clamping
                sample_x = max(0, min(src_width - 1, sample_x));
                sample_y = max(0, min(src_height - 1, sample_y));
                
                int sample_idx = (sample_y * src_width + sample_x) * channels + c;
                float pixel_value = input[sample_idx];
                
                float weight_x = bicubic_weight(dx - i);
                float weight_y = bicubic_weight(dy - j);
                float weight = weight_x * weight_y;
                
                sum += pixel_value * weight;
            }
        }
        
        int output_idx = (dst_y * dst_width + dst_x) * channels + c;
        output[output_idx] = (uint8_t)max(0.0f, min(255.0f, roundf(sum)));
    }
}

// Lanczos scaling kernel for best quality
__global__ void lanczos_scale_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    int radius
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        // Calculate support region
        int x_start = max(0, (int)ceilf(src_x - radius));
        int x_end = min(src_width - 1, (int)floorf(src_x + radius));
        int y_start = max(0, (int)ceilf(src_y - radius));
        int y_end = min(src_height - 1, (int)floorf(src_y + radius));
        
        for (int sample_y = y_start; sample_y <= y_end; sample_y++) {
            for (int sample_x = x_start; sample_x <= x_end; sample_x++) {
                float weight_x = lanczos_weight(src_x - sample_x, radius);
                float weight_y = lanczos_weight(src_y - sample_y, radius);
                float weight = weight_x * weight_y;
                
                if (weight != 0.0f) {
                    int sample_idx = (sample_y * src_width + sample_x) * channels + c;
                    sum += input[sample_idx] * weight;
                    weight_sum += weight;
                }
            }
        }
        
        int output_idx = (dst_y * dst_width + dst_x) * channels + c;
        if (weight_sum > 0.0f) {
            output[output_idx] = (uint8_t)max(0.0f, min(255.0f, roundf(sum / weight_sum)));
        } else {
            // Fallback to nearest neighbor
            int nearest_x = max(0, min(src_width - 1, (int)roundf(src_x)));
            int nearest_y = max(0, min(src_height - 1, (int)roundf(src_y)));
            int nearest_idx = (nearest_y * src_width + nearest_x) * channels + c;
            output[output_idx] = input[nearest_idx];
        }
    }
}

// Advanced area averaging for downscaling
__global__ void area_average_downscale_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    
    // Calculate source region for this destination pixel
    float src_x_start = dst_x * scale_x;
    float src_y_start = dst_y * scale_y;
    float src_x_end = (dst_x + 1) * scale_x;
    float src_y_end = (dst_y + 1) * scale_y;
    
    int x_start = (int)floorf(src_x_start);
    int y_start = (int)floorf(src_y_start);
    int x_end = (int)ceilf(src_x_end);
    int y_end = (int)ceilf(src_y_end);
    
    // Clamp to image bounds
    x_start = max(0, x_start);
    y_start = max(0, y_start);
    x_end = min(src_width, x_end);
    y_end = min(src_height, y_end);
    
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        float total_weight = 0.0f;
        
        for (int sample_y = y_start; sample_y < y_end; sample_y++) {
            for (int sample_x = x_start; sample_x < x_end; sample_x++) {
                // Calculate overlap weight
                float x_overlap = min((float)(sample_x + 1), src_x_end) - max((float)sample_x, src_x_start);
                float y_overlap = min((float)(sample_y + 1), src_y_end) - max((float)sample_y, src_y_start);
                float weight = x_overlap * y_overlap;
                
                if (weight > 0.0f) {
                    int sample_idx = (sample_y * src_width + sample_x) * channels + c;
                    sum += input[sample_idx] * weight;
                    total_weight += weight;
                }
            }
        }
        
        int output_idx = (dst_y * dst_width + dst_x) * channels + c;
        if (total_weight > 0.0f) {
            output[output_idx] = (uint8_t)roundf(sum / total_weight);
        } else {
            output[output_idx] = 0;
        }
    }
}

// Smart scaling that chooses best algorithm based on scale factor
__global__ void smart_scale_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    float sharpness
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    float avg_scale = (scale_x + scale_y) * 0.5f;
    
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    for (int c = 0; c < channels; c++) {
        float result;
        
        if (avg_scale > 2.0f) {
            // Heavy downscaling - use area averaging
            float src_x_start = dst_x * scale_x;
            float src_y_start = dst_y * scale_y;
            float src_x_end = (dst_x + 1) * scale_x;
            float src_y_end = (dst_y + 1) * scale_y;
            
            int x_start = max(0, (int)floorf(src_x_start));
            int y_start = max(0, (int)floorf(src_y_start));
            int x_end = min(src_width, (int)ceilf(src_x_end));
            int y_end = min(src_height, (int)ceilf(src_y_end));
            
            float sum = 0.0f;
            int count = 0;
            
            for (int sample_y = y_start; sample_y < y_end; sample_y++) {
                for (int sample_x = x_start; sample_x < x_end; sample_x++) {
                    int sample_idx = (sample_y * src_width + sample_x) * channels + c;
                    sum += input[sample_idx];
                    count++;
                }
            }
            
            result = (count > 0) ? sum / count : 0.0f;
            
        } else if (avg_scale > 1.0f) {
            // Moderate downscaling - use bicubic with anti-aliasing
            int src_x_int = (int)floorf(src_x);
            int src_y_int = (int)floorf(src_y);
            float dx = src_x - src_x_int;
            float dy = src_y - src_y_int;
            
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int j = -1; j <= 2; j++) {
                for (int i = -1; i <= 2; i++) {
                    int sample_x = max(0, min(src_width - 1, src_x_int + i));
                    int sample_y = max(0, min(src_height - 1, src_y_int + j));
                    
                    int sample_idx = (sample_y * src_width + sample_x) * channels + c;
                    float pixel_value = input[sample_idx];
                    
                    float weight_x = bicubic_weight(dx - i) / avg_scale;
                    float weight_y = bicubic_weight(dy - j) / avg_scale;
                    float weight = weight_x * weight_y;
                    
                    sum += pixel_value * weight;
                    weight_sum += weight;
                }
            }
            
            result = (weight_sum > 0.0f) ? sum / weight_sum : 0.0f;
            
        } else {
            // Upscaling or 1:1 - use bicubic with sharpening
            int src_x_int = (int)floorf(src_x);
            int src_y_int = (int)floorf(src_y);
            float dx = src_x - src_x_int;
            float dy = src_y - src_y_int;
            
            float sum = 0.0f;
            
            for (int j = -1; j <= 2; j++) {
                for (int i = -1; i <= 2; i++) {
                    int sample_x = max(0, min(src_width - 1, src_x_int + i));
                    int sample_y = max(0, min(src_height - 1, src_y_int + j));
                    
                    int sample_idx = (sample_y * src_width + sample_x) * channels + c;
                    float pixel_value = input[sample_idx];
                    
                    float weight_x = bicubic_weight(dx - i);
                    float weight_y = bicubic_weight(dy - j);
                    float weight = weight_x * weight_y;
                    
                    // Apply sharpening
                    if (i == 0 && j == 0) {
                        weight *= (1.0f + sharpness);
                    } else {
                        weight *= (1.0f - sharpness / 8.0f);
                    }
                    
                    sum += pixel_value * weight;
                }
            }
            
            result = sum;
        }
        
        int output_idx = (dst_y * dst_width + dst_x) * channels + c;
        output[output_idx] = (uint8_t)max(0.0f, min(255.0f, roundf(result)));
    }
}

// Aspect ratio preserving scaling with letterboxing/pillarboxing
__global__ void aspect_preserving_scale_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    uint8_t fill_r,
    uint8_t fill_g,
    uint8_t fill_b
) {
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (dst_x >= dst_width || dst_y >= dst_height) return;
    
    // Calculate aspect ratios
    float src_aspect = (float)src_width / src_height;
    float dst_aspect = (float)dst_width / dst_height;
    
    // Calculate scaled dimensions and offsets
    int scaled_width, scaled_height;
    int offset_x, offset_y;
    
    if (src_aspect > dst_aspect) {
        // Source is wider - fit to width, letterbox top/bottom
        scaled_width = dst_width;
        scaled_height = (int)roundf(dst_width / src_aspect);
        offset_x = 0;
        offset_y = (dst_height - scaled_height) / 2;
    } else {
        // Source is taller - fit to height, pillarbox left/right
        scaled_width = (int)roundf(dst_height * src_aspect);
        scaled_height = dst_height;
        offset_x = (dst_width - scaled_width) / 2;
        offset_y = 0;
    }
    
    int output_idx = (dst_y * dst_width + dst_x) * channels;
    
    // Check if we're in the scaled image area
    if (dst_x >= offset_x && dst_x < offset_x + scaled_width &&
        dst_y >= offset_y && dst_y < offset_y + scaled_height) {
        
        // Map to source coordinates
        float src_x = ((dst_x - offset_x) + 0.5f) * src_width / scaled_width - 0.5f;
        float src_y = ((dst_y - offset_y) + 0.5f) * src_height / scaled_height - 0.5f;
        
        // Bilinear interpolation
        int src_x0 = max(0, (int)floorf(src_x));
        int src_y0 = max(0, (int)floorf(src_y));
        int src_x1 = min(src_width - 1, src_x0 + 1);
        int src_y1 = min(src_height - 1, src_y0 + 1);
        
        float dx = src_x - src_x0;
        float dy = src_y - src_y0;
        
        for (int c = 0; c < channels; c++) {
            float tl = input[(src_y0 * src_width + src_x0) * channels + c];
            float tr = input[(src_y0 * src_width + src_x1) * channels + c];
            float bl = input[(src_y1 * src_width + src_x0) * channels + c];
            float br = input[(src_y1 * src_width + src_x1) * channels + c];
            
            float top = tl + dx * (tr - tl);
            float bottom = bl + dx * (br - bl);
            float result = top + dy * (bottom - top);
            
            output[output_idx + c] = (uint8_t)roundf(result);
        }
    } else {
        // Fill with background color
        if (channels >= 3) {
            output[output_idx] = fill_r;
            output[output_idx + 1] = fill_g;
            output[output_idx + 2] = fill_b;
            if (channels == 4) {
                output[output_idx + 3] = 255; // Alpha
            }
        } else {
            // Grayscale - use luminance of fill color
            uint8_t gray = (uint8_t)(0.299f * fill_r + 0.587f * fill_g + 0.114f * fill_b);
            output[output_idx] = gray;
        }
    }
}

// External C interface functions
extern "C" {
    
    cudaError_t launch_bicubic_scale(
        const uint8_t* input,
        uint8_t* output,
        int src_width,
        int src_height,
        int dst_width,
        int dst_height,
        int channels,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                      (dst_height + block_size.y - 1) / block_size.y);
        
        bicubic_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, src_width, src_height, dst_width, dst_height, channels
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_lanczos_scale(
        const uint8_t* input,
        uint8_t* output,
        int src_width,
        int src_height,
        int dst_width,
        int dst_height,
        int channels,
        int radius,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                      (dst_height + block_size.y - 1) / block_size.y);
        
        lanczos_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, src_width, src_height, dst_width, dst_height, channels, radius
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_area_average_downscale(
        const uint8_t* input,
        uint8_t* output,
        int src_width,
        int src_height,
        int dst_width,
        int dst_height,
        int channels,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                      (dst_height + block_size.y - 1) / block_size.y);
        
        area_average_downscale_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, src_width, src_height, dst_width, dst_height, channels
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_smart_scale(
        const uint8_t* input,
        uint8_t* output,
        int src_width,
        int src_height,
        int dst_width,
        int dst_height,
        int channels,
        float sharpness,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                      (dst_height + block_size.y - 1) / block_size.y);
        
        smart_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, src_width, src_height, dst_width, dst_height, channels, sharpness
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_aspect_preserving_scale(
        const uint8_t* input,
        uint8_t* output,
        int src_width,
        int src_height,
        int dst_width,
        int dst_height,
        int channels,
        uint8_t fill_r,
        uint8_t fill_g,
        uint8_t fill_b,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((dst_width + block_size.x - 1) / block_size.x,
                      (dst_height + block_size.y - 1) / block_size.y);
        
        aspect_preserving_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, src_width, src_height, dst_width, dst_height, channels,
            fill_r, fill_g, fill_b
        );
        
        return cudaGetLastError();
    }
}

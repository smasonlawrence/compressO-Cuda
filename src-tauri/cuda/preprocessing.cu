#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Advanced Gaussian blur kernel with separable convolution
__global__ void gaussian_blur_horizontal_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    int channels,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate Gaussian kernel radius (3 sigma rule)
    int radius = (int)ceilf(3.0f * sigma);
    if (radius > 15) radius = 15; // Limit kernel size
    
    // Pre-calculate Gaussian weights
    float weights[31]; // Max radius * 2 + 1
    float weight_sum = 0.0f;
    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
    
    for (int i = -radius; i <= radius; i++) {
        float weight = expf(-i * i * inv_2sigma2);
        weights[i + radius] = weight;
        weight_sum += weight;
    }
    
    // Normalize weights
    for (int i = 0; i < 2 * radius + 1; i++) {
        weights[i] /= weight_sum;
    }
    
    // Apply horizontal convolution
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int dx = -radius; dx <= radius; dx++) {
            int sample_x = min(width - 1, max(0, x + dx));
            int pixel_idx = (y * width + sample_x) * channels + c;
            sum += input[pixel_idx] * weights[dx + radius];
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (uint8_t)min(255.0f, max(0.0f, roundf(sum)));
    }
}

__global__ void gaussian_blur_vertical_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    int channels,
    float sigma
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Calculate Gaussian kernel radius
    int radius = (int)ceilf(3.0f * sigma);
    if (radius > 15) radius = 15;
    
    // Pre-calculate Gaussian weights
    float weights[31];
    float weight_sum = 0.0f;
    float inv_2sigma2 = 1.0f / (2.0f * sigma * sigma);
    
    for (int i = -radius; i <= radius; i++) {
        float weight = expf(-i * i * inv_2sigma2);
        weights[i + radius] = weight;
        weight_sum += weight;
    }
    
    for (int i = 0; i < 2 * radius + 1; i++) {
        weights[i] /= weight_sum;
    }
    
    // Apply vertical convolution
    for (int c = 0; c < channels; c++) {
        float sum = 0.0f;
        
        for (int dy = -radius; dy <= radius; dy++) {
            int sample_y = min(height - 1, max(0, y + dy));
            int pixel_idx = (sample_y * width + x) * channels + c;
            sum += input[pixel_idx] * weights[dy + radius];
        }
        
        int output_idx = (y * width + x) * channels + c;
        output[output_idx] = (uint8_t)min(255.0f, max(0.0f, roundf(sum)));
    }
}

// Advanced noise reduction using Non-Local Means algorithm (simplified)
__global__ void noise_reduction_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    int channels,
    float strength
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Parameters for non-local means
    int search_window = 11;  // Search window size
    int patch_size = 3;      // Patch size for comparison
    float h = strength * 10.0f; // Filtering strength
    float h2 = h * h;
    
    // Clamp search window to image bounds
    int search_half = search_window / 2;
    int patch_half = patch_size / 2;
    
    for (int c = 0; c < channels; c++) {
        float weighted_sum = 0.0f;
        float weight_sum = 0.0f;
        
        // Search in the neighborhood
        for (int sy = -search_half; sy <= search_half; sy++) {
            for (int sx = -search_half; sx <= search_half; sx++) {
                int nx = x + sx;
                int ny = y + sy;
                
                // Skip if outside image bounds
                if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
                
                // Calculate patch similarity
                float patch_diff = 0.0f;
                int valid_pixels = 0;
                
                for (int py = -patch_half; py <= patch_half; py++) {
                    for (int px = -patch_half; px <= patch_half; px++) {
                        int ref_x = min(width - 1, max(0, x + px));
                        int ref_y = min(height - 1, max(0, y + py));
                        int cmp_x = min(width - 1, max(0, nx + px));
                        int cmp_y = min(height - 1, max(0, ny + py));
                        
                        int ref_idx = (ref_y * width + ref_x) * channels + c;
                        int cmp_idx = (cmp_y * width + cmp_x) * channels + c;
                        
                        float diff = (float)input[ref_idx] - (float)input[cmp_idx];
                        patch_diff += diff * diff;
                        valid_pixels++;
                    }
                }
                
                if (valid_pixels > 0) {
                    patch_diff /= valid_pixels;
                    
                    // Calculate weight based on patch similarity
                    float weight = expf(-patch_diff / h2);
                    
                    int neighbor_idx = (ny * width + nx) * channels + c;
                    weighted_sum += input[neighbor_idx] * weight;
                    weight_sum += weight;
                }
            }
        }
        
        // Normalize and output
        int output_idx = (y * width + x) * channels + c;
        if (weight_sum > 0.0f) {
            output[output_idx] = (uint8_t)min(255.0f, max(0.0f, weighted_sum / weight_sum));
        } else {
            output[output_idx] = input[(y * width + x) * channels + c];
        }
    }
}

// Edge detection using Sobel operator
__global__ void edge_detection_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    float threshold
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Convert to grayscale first if needed (assuming RGB input)
    int input_idx = (y * width + x) * 3;
    float gray = 0.299f * input[input_idx] + 
                 0.587f * input[input_idx + 1] + 
                 0.114f * input[input_idx + 2];
    
    // Sobel kernels
    float sobel_x[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float sobel_y[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    
    float grad_x = 0.0f, grad_y = 0.0f;
    
    // Apply Sobel operators
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int sample_x = min(width - 1, max(0, x + dx));
            int sample_y = min(height - 1, max(0, y + dy));
            int sample_idx = (sample_y * width + sample_x) * 3;
            
            float sample_gray = 0.299f * input[sample_idx] + 
                               0.587f * input[sample_idx + 1] + 
                               0.114f * input[sample_idx + 2];
            
            int kernel_idx = (dy + 1) * 3 + (dx + 1);
            grad_x += sample_gray * sobel_x[kernel_idx];
            grad_y += sample_gray * sobel_y[kernel_idx];
        }
    }
    
    // Calculate gradient magnitude
    float magnitude = sqrtf(grad_x * grad_x + grad_y * grad_y);
    
    // Apply threshold
    uint8_t edge_value = (magnitude > threshold) ? 255 : 0;
    
    int output_idx = y * width + x;
    output[output_idx] = edge_value;
}

// Histogram computation kernel
__global__ void compute_histogram_kernel(
    const uint8_t* __restrict__ input,
    uint32_t* __restrict__ histogram,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Shared memory for local histogram
    __shared__ uint32_t local_hist[256 * 3]; // For RGB channels
    
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    for (int i = tid; i < 256 * channels; i += blockDim.x * blockDim.y) {
        local_hist[i] = 0;
    }
    __syncthreads();
    
    // Compute local histogram
    int pixel_idx = (y * width + x) * channels;
    for (int c = 0; c < channels; c++) {
        uint8_t value = input[pixel_idx + c];
        atomicAdd(&local_hist[c * 256 + value], 1);
    }
    __syncthreads();
    
    // Write back to global memory
    for (int i = tid; i < 256 * channels; i += blockDim.x * blockDim.y) {
        atomicAdd(&histogram[i], local_hist[i]);
    }
}

// Histogram equalization kernel
__global__ void histogram_equalization_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    const uint32_t* __restrict__ histogram,
    int width,
    int height,
    int channels
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int total_pixels = width * height;
    
    for (int c = 0; c < channels; c++) {
        int pixel_idx = (y * width + x) * channels + c;
        uint8_t value = input[pixel_idx];
        
        // Calculate cumulative distribution
        uint32_t cumulative = 0;
        for (int i = 0; i <= value; i++) {
            cumulative += histogram[c * 256 + i];
        }
        
        // Apply histogram equalization formula
        float normalized = (float)cumulative / total_pixels;
        uint8_t equalized = (uint8_t)(normalized * 255.0f);
        
        output[pixel_idx] = equalized;
    }
}

// External C interface functions
extern "C" {
    
    cudaError_t launch_gaussian_blur(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        int channels,
        float sigma,
        cudaStream_t stream
    ) {
        // Allocate temporary buffer for separable convolution
        uint8_t* temp_buffer;
        size_t temp_size = width * height * channels * sizeof(uint8_t);
        cudaError_t malloc_result = cudaMalloc(&temp_buffer, temp_size);
        if (malloc_result != cudaSuccess) return malloc_result;
        
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        // First pass: horizontal blur
        gaussian_blur_horizontal_kernel<<<grid_size, block_size, 0, stream>>>(
            input, temp_buffer, width, height, channels, sigma
        );
        
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            cudaFree(temp_buffer);
            return error;
        }
        
        // Second pass: vertical blur
        gaussian_blur_vertical_kernel<<<grid_size, block_size, 0, stream>>>(
            temp_buffer, output, width, height, channels, sigma
        );
        
        error = cudaGetLastError();
        cudaFree(temp_buffer);
        return error;
    }
    
    cudaError_t launch_noise_reduction(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        int channels,
        float strength,
        cudaStream_t stream
    ) {
        dim3 block_size(8, 8); // Smaller blocks for complex kernel
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        noise_reduction_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, width, height, channels, strength
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_edge_detection(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        float threshold,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        edge_detection_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, width, height, threshold
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_compute_histogram(
        const uint8_t* input,
        uint32_t* histogram,
        int width,
        int height,
        int channels,
        cudaStream_t stream
    ) {
        // Clear histogram first
        cudaMemsetAsync(histogram, 0, 256 * channels * sizeof(uint32_t), stream);
        
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        compute_histogram_kernel<<<grid_size, block_size, 0, stream>>>(
            input, histogram, width, height, channels
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_histogram_equalization(
        const uint8_t* input,
        uint8_t* output,
        const uint32_t* histogram,
        int width,
        int height,
        int channels,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        histogram_equalization_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, histogram, width, height, channels
        );
        
        return cudaGetLastError();
    }
}

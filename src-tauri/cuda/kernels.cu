#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Fast RGB to YUV420 conversion kernel
__global__ void rgb_to_yuv420_kernel(
    const uint8_t* __restrict__ rgb_input,
    uint8_t* __restrict__ y_output,
    uint8_t* __restrict__ u_output,
    uint8_t* __restrict__ v_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int rgb_idx = (y * width + x) * 3;
    int y_idx = y * width + x;
    
    // Load RGB values
    uint8_t r = rgb_input[rgb_idx];
    uint8_t g = rgb_input[rgb_idx + 1];
    uint8_t b = rgb_input[rgb_idx + 2];
    
    // Convert to YUV using BT.709 coefficients (HD standard)
    // Y = 0.2126*R + 0.7152*G + 0.0722*B
    int Y = (306 * r + 601 * g + 117 * b + 512) >> 10;
    y_output[y_idx] = min(255, max(0, Y));
    
    // For U and V, only process every 2x2 block (YUV420 subsampling)
    if ((x & 1) == 0 && (y & 1) == 0) {
        // Calculate chroma for 2x2 block
        int uv_x = x >> 1;
        int uv_y = y >> 1;
        int uv_idx = uv_y * (width >> 1) + uv_x;
        
        // U = -0.1146*R - 0.3854*G + 0.5*B + 128
        // V = 0.5*R - 0.4542*G - 0.0458*B + 128
        int U = ((-118 * r - 395 * g + 512 * b + 128) >> 10) + 128;
        int V = ((512 * r - 465 * g - 47 * b + 128) >> 10) + 128;
        
        u_output[uv_idx] = min(255, max(0, U));
        v_output[uv_idx] = min(255, max(0, V));
    }
}

// YUV420 to RGB conversion kernel
__global__ void yuv420_to_rgb_kernel(
    const uint8_t* __restrict__ y_input,
    const uint8_t* __restrict__ u_input,
    const uint8_t* __restrict__ v_input,
    uint8_t* __restrict__ rgb_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int y_idx = y * width + x;
    int uv_idx = (y >> 1) * (width >> 1) + (x >> 1);
    int rgb_idx = y_idx * 3;
    
    // Load YUV values
    int Y = y_input[y_idx];
    int U = u_input[uv_idx] - 128;
    int V = v_input[uv_idx] - 128;
    
    // Convert to RGB using BT.709 coefficients
    int R = Y + ((1434 * V + 512) >> 10);
    int G = Y - ((352 * U + 731 * V + 512) >> 10);
    int B = Y + ((1814 * U + 512) >> 10);
    
    rgb_output[rgb_idx] = min(255, max(0, R));
    rgb_output[rgb_idx + 1] = min(255, max(0, G));
    rgb_output[rgb_idx + 2] = min(255, max(0, B));
}

// High-quality bilinear scaling kernel
__global__ void bilinear_scale_kernel(
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
    
    float src_x_f = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y_f = (dst_y + 0.5f) * scale_y - 0.5f;
    
    int src_x0 = max(0, (int)floorf(src_x_f));
    int src_y0 = max(0, (int)floorf(src_y_f));
    int src_x1 = min(src_width - 1, src_x0 + 1);
    int src_y1 = min(src_height - 1, src_y0 + 1);
    
    float dx = src_x_f - src_x0;
    float dy = src_y_f - src_y0;
    
    for (int c = 0; c < channels; c++) {
        float tl = input[(src_y0 * src_width + src_x0) * channels + c];
        float tr = input[(src_y0 * src_width + src_x1) * channels + c];
        float bl = input[(src_y1 * src_width + src_x0) * channels + c];
        float br = input[(src_y1 * src_width + src_x1) * channels + c];
        
        float top = tl + dx * (tr - tl);
        float bottom = bl + dx * (br - bl);
        float result = top + dy * (bottom - top);
        
        output[(dst_y * dst_width + dst_x) * channels + c] = 
            (uint8_t)min(255.0f, max(0.0f, roundf(result)));
    }
}

// Frame preprocessing with brightness, contrast, gamma, and denoising
__global__ void preprocess_frame_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    int channels,
    float brightness,
    float contrast,
    float gamma,
    bool denoise
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int global_idx = y * width + x;
    
    for (int c = 0; c < channels; c++) {
        int pixel_idx = global_idx * channels + c;
        uint8_t pixel = input[pixel_idx];
        
        if (denoise) {
            // Simple 3x3 gaussian-like filter for noise reduction
            float sum = 0.0f;
            float weight_sum = 0.0f;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    int nx = min(width - 1, max(0, x + dx));
                    int ny = min(height - 1, max(0, y + dy));
                    int neighbor_idx = (ny * width + nx) * channels + c;
                    
                    float weight = (dx == 0 && dy == 0) ? 4.0f : 
                                  (abs(dx) + abs(dy) == 1) ? 2.0f : 1.0f;
                    
                    sum += input[neighbor_idx] * weight;
                    weight_sum += weight;
                }
            }
            pixel = (uint8_t)(sum / weight_sum);
        }
        
        // Apply brightness, contrast, and gamma correction
        float normalized = pixel / 255.0f;
        
        // Contrast and brightness adjustment
        normalized = (normalized - 0.5f) * contrast + 0.5f + brightness;
        
        // Gamma correction
        normalized = powf(max(0.0f, min(1.0f, normalized)), 1.0f / gamma);
        
        output[pixel_idx] = (uint8_t)(normalized * 255.0f);
    }
}

// External C interface functions
extern "C" {
    
    cudaError_t launch_rgb_to_yuv420(
        const uint8_t* rgb_input,
        uint8_t* y_output,
        uint8_t* u_output,
        uint8_t* v_output,
        int width,
        int height,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        rgb_to_yuv420_kernel<<<grid_size, block_size, 0, stream>>>(
            rgb_input, y_output, u_output, v_output, width, height
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_yuv420_to_rgb(
        const uint8_t* y_input,
        const uint8_t* u_input,
        const uint8_t* v_input,
        uint8_t* rgb_output,
        int width,
        int height,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        yuv420_to_rgb_kernel<<<grid_size, block_size, 0, stream>>>(
            y_input, u_input, v_input, rgb_output, width, height
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_bilinear_scale(
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
        
        bilinear_scale_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, src_width, src_height, dst_width, dst_height, channels
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_preprocess_frame(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        int channels,
        float brightness,
        float contrast,
        float gamma,
        bool denoise,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        preprocess_frame_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, width, height, channels, brightness, contrast, gamma, denoise
        );
        
        return cudaGetLastError();
    }

    // Device management functions
    int get_cuda_device_count(void) {
        int count = 0;
        cudaGetDeviceCount(&count);
        return count;
    }
    
    cudaError_t get_cuda_device_info(int device_id, CudaDeviceInfo* info) {
        cudaDeviceProp props;
        cudaError_t result = cudaGetDeviceProperties(&props, device_id);
        
        if (result == cudaSuccess) {
            info->device_id = device_id;
            info->major = props.major;
            info->minor = props.minor;
            info->multiprocessor_count = props.multiProcessorCount;
            info->max_threads_per_block = props.maxThreadsPerBlock;
            info->warp_size = props.warpSize;
            
            // Copy device name safely
            strncpy(info->name, props.name, sizeof(info->name) - 1);
            info->name[sizeof(info->name) - 1] = '\0';
            
            // Get memory info
            size_t free_mem, total_mem;
            if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
                info->free_memory = free_mem;
                info->total_memory = total_mem;
            } else {
                info->free_memory = 0;
                info->total_memory = 0;
            }
        }
        
        return result;
    }
    
    cudaError_t set_cuda_device(int device_id) {
        return cudaSetDevice(device_id);
    }
    
    cudaError_t cuda_device_synchronize(void) {
        return cudaDeviceSynchronize();
    }
    
    // Memory management functions
    cudaError_t cuda_malloc_device(void** ptr, size_t size) {
        return cudaMalloc(ptr, size);
    }
    
    cudaError_t cuda_free_device(void* ptr) {
        return cudaFree(ptr);
    }
    
    cudaError_t cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
        return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    }
    
    cudaError_t cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
        return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    }
    
    cudaError_t cuda_memcpy_device_to_device(void* dst, const void* src, size_t size) {
        return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
    }
    
    // Stream management functions
    cudaError_t cuda_stream_create(cudaStream_t* stream) {
        return cudaStreamCreate(stream);
    }
    
    cudaError_t cuda_stream_destroy(cudaStream_t stream) {
        return cudaStreamDestroy(stream);
    }
    
    cudaError_t cuda_stream_synchronize(cudaStream_t stream) {
        return cudaStreamSynchronize(stream);
    }
}

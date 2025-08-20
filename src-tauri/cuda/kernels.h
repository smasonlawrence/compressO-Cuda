#ifndef COMPRESSO_CUDA_KERNELS_H
#define COMPRESSO_CUDA_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Device information structure
typedef struct {
    int device_id;
    size_t total_memory;
    size_t free_memory;
    int major;
    int minor;
    char name[256];
    int multiprocessor_count;
    int max_threads_per_block;
    int warp_size;
} CudaDeviceInfo;

// Color space conversion kernels
cudaError_t launch_rgb_to_yuv420(
    const uint8_t* rgb_input,
    uint8_t* y_output,
    uint8_t* u_output,
    uint8_t* v_output,
    int width,
    int height,
    cudaStream_t stream
);

cudaError_t launch_yuv420_to_rgb(
    const uint8_t* y_input,
    const uint8_t* u_input,
    const uint8_t* v_input,
    uint8_t* rgb_output,
    int width,
    int height,
    cudaStream_t stream
);

// Scaling and filtering kernels
cudaError_t launch_bilinear_scale(
    const uint8_t* input,
    uint8_t* output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    cudaStream_t stream
);

cudaError_t launch_bicubic_scale(
    const uint8_t* input,
    uint8_t* output,
    int src_width,
    int src_height,
    int dst_width,
    int dst_height,
    int channels,
    cudaStream_t stream
);

// Frame preprocessing kernels
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
);

cudaError_t launch_gaussian_blur(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    float sigma,
    cudaStream_t stream
);

// Motion estimation kernels
cudaError_t launch_motion_estimation(
    const uint8_t* current_frame,
    const uint8_t* reference_frame,
    int2* motion_vectors,
    int width,
    int height,
    int block_size,
    int search_range,
    cudaStream_t stream
);

// Memory management utilities
cudaError_t launch_async_memcpy(
    const void* src,
    void* dst,
    size_t size,
    cudaStream_t stream
);

// Advanced filtering kernels
cudaError_t launch_edge_detection(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    float threshold,
    cudaStream_t stream
);

cudaError_t launch_noise_reduction(
    const uint8_t* input,
    uint8_t* output,
    int width,
    int height,
    int channels,
    float strength,
    cudaStream_t stream
);

// Histogram and statistics kernels
cudaError_t launch_compute_histogram(
    const uint8_t* input,
    uint32_t* histogram,
    int width,
    int height,
    int channels,
    cudaStream_t stream
);

cudaError_t launch_histogram_equalization(
    const uint8_t* input,
    uint8_t* output,
    const uint32_t* histogram,
    int width,
    int height,
    int channels,
    cudaStream_t stream
);

// Device management functions
int get_cuda_device_count(void);
cudaError_t get_cuda_device_info(int device_id, CudaDeviceInfo* info);
cudaError_t set_cuda_device(int device_id);
cudaError_t cuda_device_synchronize(void);

// Memory management functions
cudaError_t cuda_malloc_device(void** ptr, size_t size);
cudaError_t cuda_free_device(void* ptr);
cudaError_t cuda_memcpy_host_to_device(void* dst, const void* src, size_t size);
cudaError_t cuda_memcpy_device_to_host(void* dst, const void* src, size_t size);
cudaError_t cuda_memcpy_device_to_device(void* dst, const void* src, size_t size);

// Stream management functions
cudaError_t cuda_stream_create(cudaStream_t* stream);
cudaError_t cuda_stream_destroy(cudaStream_t stream);
cudaError_t cuda_stream_synchronize(cudaStream_t stream);

// Error handling utilities
const char* cuda_get_error_string(cudaError_t error);
cudaError_t cuda_get_last_error(void);

// Performance monitoring
cudaError_t cuda_event_create(cudaEvent_t* event);
cudaError_t cuda_event_destroy(cudaEvent_t event);
cudaError_t cuda_event_record(cudaEvent_t event, cudaStream_t stream);
cudaError_t cuda_event_elapsed_time(float* ms, cudaEvent_t start, cudaEvent_t end);

#ifdef __cplusplus
}
#endif

#endif // COMPRESSO_CUDA_KERNELS_H

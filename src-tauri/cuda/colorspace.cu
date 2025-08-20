#include "kernels.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_constants.h>

// Color space conversion matrices (BT.709)
__constant__ float RGB_TO_YUV_BT709[9] = {
    0.2126f, 0.7152f, 0.0722f,    // Y
   -0.1146f,-0.3854f, 0.5000f,    // U
    0.5000f,-0.4542f,-0.0458f     // V
};

__constant__ float YUV_TO_RGB_BT709[9] = {
    1.0000f, 0.0000f, 1.5748f,    // R
    1.0000f,-0.1873f,-0.4681f,    // G
    1.0000f, 1.8556f, 0.0000f     // B
};

// BT.2020 matrices for HDR content
__constant__ float RGB_TO_YUV_BT2020[9] = {
    0.2627f, 0.6780f, 0.0593f,
   -0.1396f,-0.3604f, 0.5000f,
    0.5000f,-0.4598f,-0.0402f
};

__constant__ float YUV_TO_RGB_BT2020[9] = {
    1.0000f, 0.0000f, 1.7166f,
    1.0000f,-0.1910f,-0.6663f,
    1.0000f, 2.1910f, 0.0000f
};

// Advanced RGB to YUV420 with proper chroma subsampling
__global__ void rgb_to_yuv420_advanced_kernel(
    const uint8_t* __restrict__ rgb_input,
    uint8_t* __restrict__ y_output,
    uint8_t* __restrict__ u_output,
    uint8_t* __restrict__ v_output,
    int width,
    int height,
    bool use_bt2020
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Choose color matrix
    const float* matrix = use_bt2020 ? RGB_TO_YUV_BT2020 : RGB_TO_YUV_BT709;
    
    int rgb_idx = (y * width + x) * 3;
    int y_idx = y * width + x;
    
    // Load RGB values and normalize to [0,1]
    float r = rgb_input[rgb_idx] / 255.0f;
    float g = rgb_input[rgb_idx + 1] / 255.0f;
    float b = rgb_input[rgb_idx + 2] / 255.0f;
    
    // Apply gamma correction (sRGB to linear)
    r = (r <= 0.04045f) ? r / 12.92f : powf((r + 0.055f) / 1.055f, 2.4f);
    g = (g <= 0.04045f) ? g / 12.92f : powf((g + 0.055f) / 1.055f, 2.4f);
    b = (b <= 0.04045f) ? b / 12.92f : powf((b + 0.055f) / 1.055f, 2.4f);
    
    // Convert to YUV
    float Y = matrix[0] * r + matrix[1] * g + matrix[2] * b;
    float U = matrix[3] * r + matrix[4] * g + matrix[5] * b + 0.5f;
    float V = matrix[6] * r + matrix[7] * g + matrix[8] * b + 0.5f;
    
    // Apply inverse gamma (linear to sRGB)
    Y = (Y <= 0.0031308f) ? Y * 12.92f : 1.055f * powf(Y, 1.0f/2.4f) - 0.055f;
    
    // Store Y component
    y_output[y_idx] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, Y * 255.0f + 0.5f)));
    
    // For chroma subsampling (4:2:0), process every 2x2 block
    if ((x & 1) == 0 && (y & 1) == 0) {
        // Average 2x2 block for better chroma quality
        float sum_u = 0.0f, sum_v = 0.0f;
        int count = 0;
        
        for (int dy = 0; dy < 2 && (y + dy) < height; dy++) {
            for (int dx = 0; dx < 2 && (x + dx) < width; dx++) {
                int sample_idx = ((y + dy) * width + (x + dx)) * 3;
                float sr = rgb_input[sample_idx] / 255.0f;
                float sg = rgb_input[sample_idx + 1] / 255.0f;
                float sb = rgb_input[sample_idx + 2] / 255.0f;
                
                // Gamma correction
                sr = (sr <= 0.04045f) ? sr / 12.92f : powf((sr + 0.055f) / 1.055f, 2.4f);
                sg = (sg <= 0.04045f) ? sg / 12.92f : powf((sg + 0.055f) / 1.055f, 2.4f);
                sb = (sb <= 0.04045f) ? sb / 12.92f : powf((sb + 0.055f) / 1.055f, 2.4f);
                
                sum_u += matrix[3] * sr + matrix[4] * sg + matrix[5] * sb + 0.5f;
                sum_v += matrix[6] * sr + matrix[7] * sg + matrix[8] * sb + 0.5f;
                count++;
            }
        }
        
        if (count > 0) {
            int uv_x = x >> 1;
            int uv_y = y >> 1;
            int uv_idx = uv_y * (width >> 1) + uv_x;
            
            u_output[uv_idx] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, (sum_u / count) * 255.0f + 0.5f)));
            v_output[uv_idx] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, (sum_v / count) * 255.0f + 0.5f)));
        }
    }
}

// Advanced YUV420 to RGB with chroma upsampling
__global__ void yuv420_to_rgb_advanced_kernel(
    const uint8_t* __restrict__ y_input,
    const uint8_t* __restrict__ u_input,
    const uint8_t* __restrict__ v_input,
    uint8_t* __restrict__ rgb_output,
    int width,
    int height,
    bool use_bt2020
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const float* matrix = use_bt2020 ? YUV_TO_RGB_BT2020 : YUV_TO_RGB_BT709;
    
    int y_idx = y * width + x;
    int rgb_idx = y_idx * 3;
    
    // Get Y component
    float Y_val = y_input[y_idx] / 255.0f;
    
    // Bilinear interpolation for chroma upsampling
    float u_x = (x * 0.5f) - 0.25f;
    float u_y = (y * 0.5f) - 0.25f;
    
    int u_x0 = (int)floorf(u_x);
    int u_y0 = (int)floorf(u_y);
    int u_x1 = u_x0 + 1;
    int u_y1 = u_y0 + 1;
    
    float fx = u_x - u_x0;
    float fy = u_y - u_y0;
    
    int chroma_width = width >> 1;
    int chroma_height = height >> 1;
    
    // Clamp coordinates
    u_x0 = fmaxf(0, fminf(chroma_width - 1, u_x0));
    u_y0 = fmaxf(0, fminf(chroma_height - 1, u_y0));
    u_x1 = fmaxf(0, fminf(chroma_width - 1, u_x1));
    u_y1 = fmaxf(0, fminf(chroma_height - 1, u_y1));
    
    // Bilinear interpolation for U
    float u00 = u_input[u_y0 * chroma_width + u_x0] / 255.0f - 0.5f;
    float u01 = u_input[u_y0 * chroma_width + u_x1] / 255.0f - 0.5f;
    float u10 = u_input[u_y1 * chroma_width + u_x0] / 255.0f - 0.5f;
    float u11 = u_input[u_y1 * chroma_width + u_x1] / 255.0f - 0.5f;
    
    float U_val = u00 * (1-fx) * (1-fy) + u01 * fx * (1-fy) + 
                  u10 * (1-fx) * fy + u11 * fx * fy;
    
    // Bilinear interpolation for V
    float v00 = v_input[u_y0 * chroma_width + u_x0] / 255.0f - 0.5f;
    float v01 = v_input[u_y0 * chroma_width + u_x1] / 255.0f - 0.5f;
    float v10 = v_input[u_y1 * chroma_width + u_x0] / 255.0f - 0.5f;
    float v11 = v_input[u_y1 * chroma_width + u_x1] / 255.0f - 0.5f;
    
    float V_val = v00 * (1-fx) * (1-fy) + v01 * fx * (1-fy) + 
                  v10 * (1-fx) * fy + v11 * fx * fy;
    
    // Convert YUV to RGB
    float R = matrix[0] * Y_val + matrix[1] * U_val + matrix[2] * V_val;
    float G = matrix[3] * Y_val + matrix[4] * U_val + matrix[5] * V_val;
    float B = matrix[6] * Y_val + matrix[7] * U_val + matrix[8] * V_val;
    
    // Apply gamma correction (linear to sRGB)
    R = fmaxf(0.0f, fminf(1.0f, R));
    G = fmaxf(0.0f, fminf(1.0f, G));
    B = fmaxf(0.0f, fminf(1.0f, B));
    
    R = (R <= 0.0031308f) ? R * 12.92f : 1.055f * powf(R, 1.0f/2.4f) - 0.055f;
    G = (G <= 0.0031308f) ? G * 12.92f : 1.055f * powf(G, 1.0f/2.4f) - 0.055f;
    B = (B <= 0.0031308f) ? B * 12.92f : 1.055f * powf(B, 1.0f/2.4f) - 0.055f;
    
    rgb_output[rgb_idx] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, R * 255.0f + 0.5f)));
    rgb_output[rgb_idx + 1] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, G * 255.0f + 0.5f)));
    rgb_output[rgb_idx + 2] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, B * 255.0f + 0.5f)));
}

// HSV color space conversion
__global__ void rgb_to_hsv_kernel(
    const uint8_t* __restrict__ rgb_input,
    float* __restrict__ hsv_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = (y * width + x) * 3;
    
    float r = rgb_input[pixel_idx] / 255.0f;
    float g = rgb_input[pixel_idx + 1] / 255.0f;
    float b = rgb_input[pixel_idx + 2] / 255.0f;
    
    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;
    
    // Hue calculation
    float h = 0.0f;
    if (delta > 0.0f) {
        if (max_val == r) {
            h = 60.0f * (fmodf((g - b) / delta, 6.0f));
        } else if (max_val == g) {
            h = 60.0f * ((b - r) / delta + 2.0f);
        } else {
            h = 60.0f * ((r - g) / delta + 4.0f);
        }
    }
    if (h < 0.0f) h += 360.0f;
    
    // Saturation calculation
    float s = (max_val > 0.0f) ? delta / max_val : 0.0f;
    
    // Value calculation
    float v = max_val;
    
    hsv_output[pixel_idx] = h;
    hsv_output[pixel_idx + 1] = s;
    hsv_output[pixel_idx + 2] = v;
}

// HSV to RGB conversion
__global__ void hsv_to_rgb_kernel(
    const float* __restrict__ hsv_input,
    uint8_t* __restrict__ rgb_output,
    int width,
    int height
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = (y * width + x) * 3;
    
    float h = hsv_input[pixel_idx];
    float s = hsv_input[pixel_idx + 1];
    float v = hsv_input[pixel_idx + 2];
    
    float c = v * s;
    float x_val = c * (1.0f - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;
    
    float r, g, b;
    
    if (h < 60.0f) {
        r = c; g = x_val; b = 0.0f;
    } else if (h < 120.0f) {
        r = x_val; g = c; b = 0.0f;
    } else if (h < 180.0f) {
        r = 0.0f; g = c; b = x_val;
    } else if (h < 240.0f) {
        r = 0.0f; g = x_val; b = c;
    } else if (h < 300.0f) {
        r = x_val; g = 0.0f; b = c;
    } else {
        r = c; g = 0.0f; b = x_val;
    }
    
    rgb_output[pixel_idx] = (uint8_t)((r + m) * 255.0f + 0.5f);
    rgb_output[pixel_idx + 1] = (uint8_t)((g + m) * 255.0f + 0.5f);
    rgb_output[pixel_idx + 2] = (uint8_t)((b + m) * 255.0f + 0.5f);
}

// Color enhancement kernel
__global__ void color_enhancement_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    float saturation_boost,
    float vibrance_boost,
    float warmth_adjustment
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = (y * width + x) * 3;
    
    float r = input[pixel_idx] / 255.0f;
    float g = input[pixel_idx + 1] / 255.0f;
    float b = input[pixel_idx + 2] / 255.0f;
    
    // Convert to HSV for saturation adjustment
    float max_val = fmaxf(r, fmaxf(g, b));
    float min_val = fminf(r, fminf(g, b));
    float delta = max_val - min_val;
    
    float sat = (max_val > 0.0f) ? delta / max_val : 0.0f;
    
    // Apply vibrance (selective saturation)
    float vibrance_factor = 1.0f + vibrance_boost * (1.0f - sat);
    sat *= vibrance_factor;
    
    // Apply general saturation boost
    sat *= (1.0f + saturation_boost);
    sat = fminf(1.0f, sat);
    
    // Warmth adjustment (blue-yellow balance)
    float warmth_factor = 1.0f + warmth_adjustment;
    if (warmth_adjustment > 0.0f) {
        // Warmer: boost red/yellow, reduce blue
        r *= warmth_factor;
        g *= (1.0f + warmth_adjustment * 0.5f);
        b *= (2.0f - warmth_factor);
    } else {
        // Cooler: boost blue, reduce red/yellow
        r *= (2.0f + warmth_factor);
        g *= (1.0f - warmth_adjustment * 0.5f);
        b *= (2.0f - warmth_factor);
    }
    
    // Reconstruct RGB with new saturation
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    r = gray + (r - gray) * sat;
    g = gray + (g - gray) * sat;
    b = gray + (b - gray) * sat;
    
    // Clamp and output
    output[pixel_idx] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, r * 255.0f + 0.5f)));
    output[pixel_idx + 1] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, g * 255.0f + 0.5f)));
    output[pixel_idx + 2] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, b * 255.0f + 0.5f)));
}

// White balance correction kernel
__global__ void white_balance_kernel(
    const uint8_t* __restrict__ input,
    uint8_t* __restrict__ output,
    int width,
    int height,
    float temp_adjustment,  // -1.0 to 1.0 (cool to warm)
    float tint_adjustment   // -1.0 to 1.0 (green to magenta)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int pixel_idx = (y * width + x) * 3;
    
    float r = input[pixel_idx] / 255.0f;
    float g = input[pixel_idx + 1] / 255.0f;
    float b = input[pixel_idx + 2] / 255.0f;
    
    // Temperature adjustment (blue-yellow axis)
    float temp_factor = 1.0f + temp_adjustment * 0.3f;
    if (temp_adjustment > 0.0f) {
        // Warmer
        r *= (1.0f + temp_adjustment * 0.2f);
        g *= (1.0f + temp_adjustment * 0.1f);
        b *= (1.0f - temp_adjustment * 0.3f);
    } else {
        // Cooler
        r *= (1.0f + temp_adjustment * 0.3f);
        g *= (1.0f + temp_adjustment * 0.1f);
        b *= (1.0f - temp_adjustment * 0.2f);
    }
    
    // Tint adjustment (green-magenta axis)
    if (tint_adjustment > 0.0f) {
        // More magenta
        r *= (1.0f + tint_adjustment * 0.1f);
        g *= (1.0f - tint_adjustment * 0.2f);
        b *= (1.0f + tint_adjustment * 0.1f);
    } else {
        // More green
        r *= (1.0f + tint_adjustment * 0.1f);
        g *= (1.0f - tint_adjustment * 0.2f);
        b *= (1.0f + tint_adjustment * 0.1f);
    }
    
    // Clamp and output
    output[pixel_idx] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, r * 255.0f + 0.5f)));
    output[pixel_idx + 1] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, g * 255.0f + 0.5f)));
    output[pixel_idx + 2] = (uint8_t)(fminf(255.0f, fmaxf(0.0f, b * 255.0f + 0.5f)));
}

// External C interface functions
extern "C" {
    
    cudaError_t launch_rgb_to_yuv420_advanced(
        const uint8_t* rgb_input,
        uint8_t* y_output,
        uint8_t* u_output,
        uint8_t* v_output,
        int width,
        int height,
        bool use_bt2020,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        rgb_to_yuv420_advanced_kernel<<<grid_size, block_size, 0, stream>>>(
            rgb_input, y_output, u_output, v_output, width, height, use_bt2020
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_yuv420_to_rgb_advanced(
        const uint8_t* y_input,
        const uint8_t* u_input,
        const uint8_t* v_input,
        uint8_t* rgb_output,
        int width,
        int height,
        bool use_bt2020,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        yuv420_to_rgb_advanced_kernel<<<grid_size, block_size, 0, stream>>>(
            y_input, u_input, v_input, rgb_output, width, height, use_bt2020
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_color_enhancement(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        float saturation_boost,
        float vibrance_boost,
        float warmth_adjustment,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        color_enhancement_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, width, height, saturation_boost, vibrance_boost, warmth_adjustment
        );
        
        return cudaGetLastError();
    }
    
    cudaError_t launch_white_balance(
        const uint8_t* input,
        uint8_t* output,
        int width,
        int height,
        float temp_adjustment,
        float tint_adjustment,
        cudaStream_t stream
    ) {
        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x,
                      (height + block_size.y - 1) / block_size.y);
        
        white_balance_kernel<<<grid_size, block_size, 0, stream>>>(
            input, output, width, height, temp_adjustment, tint_adjustment
        );
        
        return cudaGetLastError();
    }
}

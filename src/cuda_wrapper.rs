use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::ffi::{c_void, CStr};
use std::ptr;
use tracing::{debug, error, info, warn};

// Include the generated CUDA bindings only if CUDA feature is enabled
#[cfg(feature = "cuda")]
include!(concat!(env!("OUT_DIR"), "/cuda_bindings.rs"));

// Mock types for when CUDA is not available
#[cfg(not(feature = "cuda"))]
mod mock_cuda {
    pub type cudaError_t = i32;
    pub type cudaStream_t = *mut std::ffi::c_void;
    pub const cudaSuccess: cudaError_t = 0;
    
    #[repr(C)]
    pub struct CudaDeviceInfo {
        pub device_id: i32,
        pub total_memory: usize,
        pub free_memory: usize,
        pub major: i32,
        pub minor: i32,
        pub name: [i8; 256],
        pub multiprocessor_count: i32,
        pub max_threads_per_block: i32,
        pub warp_size: i32,
    }
}

#[cfg(not(feature = "cuda"))]
use mock_cuda::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CudaDeviceInfo {
    pub device_id: i32,
    pub name: String,
    pub total_memory: usize,
    pub free_memory: usize,
    pub compute_capability: (i32, i32),
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
}

#[derive(Debug)]
pub struct CudaContext {
    device_id: i32,
    streams: Vec<cudaStream_t>,
    device_info: CudaDeviceInfo,
}

#[derive(Debug)]
pub struct CudaBuffer {
    ptr: *mut c_void,
    size: usize,
    device_id: i32,
}

unsafe impl Send for CudaBuffer {}
unsafe impl Sync for CudaBuffer {}

impl CudaContext {
    pub fn new(device_id: Option<i32>) -> Result<Self> {
        if !is_cuda_available() {
            return Err(anyhow!("CUDA not available"));
        }

        let device_count = Self::get_device_count()?;
        if device_count == 0 {
            return Err(anyhow!("No CUDA devices found"));
        }

        let device_id = device_id.unwrap_or(0);
        if device_id >= device_count {
            return Err(anyhow!("Invalid device ID: {}", device_id));
        }

        #[cfg(feature = "cuda")]
        unsafe {
            let result = set_cuda_device(device_id);
            if result != cudaSuccess {
                return Err(anyhow!("Failed to set CUDA device: error code {}", result));
            }
        }

        let device_info = Self::get_device_info(device_id)?;
        info!("Initialized CUDA context on device: {}", device_info.name);
        debug!("Device info: {:?}", device_info);

        Ok(Self {
            device_id,
            streams: Vec::new(),
            device_info,
        })
    }

    pub fn get_device_count() -> Result<i32> {
        #[cfg(feature = "cuda")]
        {
            let count = unsafe { get_cuda_device_count() };
            Ok(count)
        }
        #[cfg(not(feature = "cuda"))]
        Ok(0)
    }

    pub fn get_device_info(device_id: i32) -> Result<CudaDeviceInfo> {
        #[cfg(feature = "cuda")]
        unsafe {
            let mut info: mock_cuda::CudaDeviceInfo = std::mem::zeroed();
            let result = get_cuda_device_info(device_id, &mut info);
            if result != cudaSuccess {
                return Err(anyhow!("Failed to get device properties: error code {}", result));
            }

            let name = CStr::from_ptr(info.name.as_ptr())
                .to_string_lossy()
                .to_string();

            Ok(CudaDeviceInfo {
                device_id,
                name,
                total_memory: info.total_memory,
                free_memory: info.free_memory,
                compute_capability: (info.major, info.minor),
                multiprocessor_count: info.multiprocessor_count,
                max_threads_per_block: info.max_threads_per_block,
                warp_size: info.warp_size,
            })
        }
        #[cfg(not(feature = "cuda"))]
        Err(anyhow!("CUDA not available"))
    }

    pub fn create_stream(&mut self) -> Result<cudaStream_t> {
        #[cfg(feature = "cuda")]
        unsafe {
            let mut stream: cudaStream_t = ptr::null_mut();
            let result = cuda_stream_create(&mut stream);
            if result != cudaSuccess {
                return Err(anyhow!("Failed to create CUDA stream: error code {}", result));
            }
            self.streams.push(stream);
            Ok(stream)
        }
        #[cfg(not(feature = "cuda"))]
        Err(anyhow!("CUDA not available"))
    }

    pub fn synchronize(&self) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let result = cuda_device_synchronize();
            if result != cudaSuccess {
                return Err(anyhow!("Failed to synchronize device: error code {}", result));
            }
        }
        Ok(())
    }

    pub fn allocate_buffer(&self, size: usize) -> Result<CudaBuffer> {
        CudaBuffer::new(size, self.device_id)
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }

    // Color space conversion methods
    pub fn rgb_to_yuv420(
        &self,
        rgb_input: &CudaBuffer,
        y_output: &mut CudaBuffer,
        u_output: &mut CudaBuffer,
        v_output: &mut CudaBuffer,
        width: i32,
        height: i32,
        stream: cudaStream_t,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let result = launch_rgb_to_yuv420(
                rgb_input.ptr as *const u8,
                y_output.ptr as *mut u8,
                u_output.ptr as *mut u8,
                v_output.ptr as *mut u8,
                width,
                height,
                stream,
            );
            
            if result != cudaSuccess {
                return Err(anyhow!("RGB to YUV420 conversion failed: error code {}", result));
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err(anyhow!("CUDA not available"));
        
        Ok(())
    }

    pub fn yuv420_to_rgb(
        &self,
        y_input: &CudaBuffer,
        u_input: &CudaBuffer,
        v_input: &CudaBuffer,
        rgb_output: &mut CudaBuffer,
        width: i32,
        height: i32,
        stream: cudaStream_t,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let result = launch_yuv420_to_rgb(
                y_input.ptr as *const u8,
                u_input.ptr as *const u8,
                v_input.ptr as *const u8,
                rgb_output.ptr as *mut u8,
                width,
                height,
                stream,
            );
            
            if result != cudaSuccess {
                return Err(anyhow!("YUV420 to RGB conversion failed: error code {}", result));
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err(anyhow!("CUDA not available"));
        
        Ok(())
    }

    // Scaling methods
    pub fn bilinear_scale(
        &self,
        input: &CudaBuffer,
        output: &mut CudaBuffer,
        src_width: i32,
        src_height: i32,
        dst_width: i32,
        dst_height: i32,
        channels: i32,
        stream: cudaStream_t,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let result = launch_bilinear_scale(
                input.ptr as *const u8,
                output.ptr as *mut u8,
                src_width,
                src_height,
                dst_width,
                dst_height,
                channels,
                stream,
            );
            
            if result != cudaSuccess {
                return Err(anyhow!("Bilinear scaling failed: error code {}", result));
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err(anyhow!("CUDA not available"));
        
        Ok(())
    }

    // Frame preprocessing methods
    pub fn preprocess_frame(
        &self,
        input: &CudaBuffer,
        output: &mut CudaBuffer,
        width: i32,
        height: i32,
        channels: i32,
        brightness: f32,
        contrast: f32,
        gamma: f32,
        denoise: bool,
        stream: cudaStream_t,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        unsafe {
            let result = launch_preprocess_frame(
                input.ptr as *const u8,
                output.ptr as *mut u8,
                width,
                height,
                channels,
                brightness,
                contrast,
                gamma,
                denoise,
                stream,
            );
            
            if result != cudaSuccess {
                return Err(anyhow!("Frame preprocessing failed: error code {}", result));
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err(anyhow!("CUDA not available"));
        
        Ok(())
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        {
            for &stream in &self.streams {
                unsafe {
                    let _ = cuda_stream_destroy(stream);
                }
            }
        }
    }
}

impl CudaBuffer {
    pub fn new(size: usize, device_id: i32) -> Result<Self> {
        #[cfg(feature = "cuda")]
        unsafe {
            let mut ptr: *mut c_void = ptr::null_mut();
            let result = cuda_malloc_device(&mut ptr, size);
            if result != cudaSuccess {
                return Err(anyhow!("Failed to allocate CUDA memory: error code {}", result));
            }

            Ok(Self {
                ptr,
                size,
                device_id,
            })
        }
        #[cfg(not(feature = "cuda"))]
        Err(anyhow!("CUDA not available"))
    }

    pub fn copy_from_host(&mut self, data: &[u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Data size exceeds buffer capacity"));
        }

        #[cfg(feature = "cuda")]
        unsafe {
            let result = cuda_memcpy_host_to_device(
                self.ptr,
                data.as_ptr() as *const c_void,
                data.len(),
            );
            
            if result != cudaSuccess {
                return Err(anyhow!("Failed to copy data to device: error code {}", result));
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err(anyhow!("CUDA not available"));
        
        Ok(())
    }

    pub fn copy_to_host(&self, data: &mut [u8]) -> Result<()> {
        if data.len() > self.size {
            return Err(anyhow!("Buffer size exceeds data capacity"));
        }

        #[cfg(feature = "cuda")]
        unsafe {
            let result = cuda_memcpy_device_to_host(
                data.as_mut_ptr() as *mut c_void,
                self.ptr,
                data.len(),
            );
            
            if result != cudaSuccess {
                return Err(anyhow!("Failed to copy data from device: error code {}", result));
            }
        }
        #[cfg(not(feature = "cuda"))]
        return Err(anyhow!("CUDA not available"));
        
        Ok(())
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn ptr(&self) -> *mut c_void {
        self.ptr
    }
}

impl Drop for CudaBuffer {
    fn drop(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            let _ = cuda_free_device(self.ptr);
        }
    }
}

// High-level video processor
pub struct CudaVideoProcessor {
    context: CudaContext,
    streams: Vec<cudaStream_t>,
}

impl CudaVideoProcessor {
    pub fn new(device_id: Option<i32>) -> Result<Self> {
        let context = CudaContext::new(device_id)?;
        Ok(Self {
            context,
            streams: Vec::new(),
        })
    }

    pub fn initialize_streams(&mut self, num_streams: usize) -> Result<()> {
        self.streams.clear();
        for _ in 0..num_streams {
            let stream = self.context.create_stream()?;
            self.streams.push(stream);
        }
        Ok(())
    }

    pub fn process_frame_pipeline(
        &mut self,
        input_data: &[u8],
        width: u32,
        height: u32,
        target_width: u32,
        target_height: u32,
        preprocessing_params: &PreprocessingParams,
    ) -> Result<Vec<u8>> {
        let channels = 3; // RGB
        let input_size = (width * height * channels) as usize;
        let output_size = (target_width * target_height * channels) as usize;

        // Allocate buffers
        let mut input_buffer = self.context.allocate_buffer(input_size)?;
        let mut scaled_buffer = self.context.allocate_buffer(output_size)?;
        let mut processed_buffer = self.context.allocate_buffer(output_size)?;

        // Get a stream for this operation
        let stream = if self.streams.is_empty() {
            self.context.create_stream()?
        } else {
            self.streams[0]
        };

        // Copy input data to GPU
        input_buffer.copy_from_host(input_data)?;

        // Scale the frame if needed
        if width != target_width || height != target_height {
            self.context.bilinear_scale(
                &input_buffer,
                &mut scaled_buffer,
                width as i32,
                height as i32,
                target_width as i32,
                target_height as i32,
                channels as i32,
                stream,
            )?;
        } else {
            // No scaling needed, just copy
            #[cfg(feature = "cuda")]
            unsafe {
                let _ = cuda_memcpy_device_to_device(
                    scaled_buffer.ptr(),
                    input_buffer.ptr(),
                    input_size.min(output_size),
                );
            }
        }

        // Apply preprocessing
        self.context.preprocess_frame(
            &scaled_buffer,
            &mut processed_buffer,
            target_width as i32,
            target_height as i32,
            channels as i32,
            preprocessing_params.brightness,
            preprocessing_params.contrast,
            preprocessing_params.gamma,
            preprocessing_params.denoise,
            stream,
        )?;

        // Copy result back to host
        let mut output_data = vec![0u8; output_size];
        processed_buffer.copy_to_host(&mut output_data)?;

        // Synchronize to ensure completion
        self.context.synchronize()?;

        Ok(output_data)
    }

    pub fn device_info(&self) -> &CudaDeviceInfo {
        self.context.device_info()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingParams {
    pub brightness: f32,
    pub contrast: f32,
    pub gamma: f32,
    pub denoise: bool,
}

impl Default for PreprocessingParams {
    fn default() -> Self {
        Self {
            brightness: 0.0,
            contrast: 1.0,
            gamma: 1.0,
            denoise: false,
        }
    }
}

// Utility functions
pub fn is_cuda_available() -> bool {
    #[cfg(feature = "cuda")]
    {
        match CudaContext::get_device_count() {
            Ok(count) => count > 0,
            Err(_) => false,
        }
    }
    #[cfg(not(feature = "cuda"))]
    false
}

pub fn get_best_cuda_device() -> Result<i32> {
    if !is_cuda_available() {
        return Err(anyhow!("CUDA not available"));
    }

    let device_count = CudaContext::get_device_count()?;
    let mut best_device = 0;
    let mut best_memory = 0;

    for i in 0..device_count {
        if let Ok(info) = CudaContext::get_device_info(i) {
            if info.total_memory > best_memory {
                best_memory = info.total_memory;
                best_device = i;
            }
        }
    }

    Ok(best_device)
}

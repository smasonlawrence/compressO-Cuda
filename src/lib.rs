//! CompressO-Cuda: CUDA-accelerated video compression library
//! 
//! This library provides high-performance video compression capabilities
//! using NVIDIA CUDA acceleration for preprocessing and NVENC for encoding.

pub mod cuda_wrapper;
pub mod compression;

// Re-export main types for easier access
pub use compression::{
    CompressionSettings, 
    CompressionProgress, 
    CompressionQuality,
    VideoCodec,
    EncodingPreset,
    H264Profile,
    H264Level,
    ContainerFormat,
    CudaCompressionEngine,
};

pub use cuda_wrapper::{
    CudaContext,
    CudaBuffer,
    CudaVideoProcessor,
    CudaDeviceInfo,
    PreprocessingParams,
    is_cuda_available,
    get_best_cuda_device,
};

/// Library version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Check if the library was compiled with CUDA support
pub const fn has_cuda_support() -> bool {
    cfg!(feature = "cuda")
}

/// Initialize the library with optional logging configuration
pub fn init() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging if not already done
    if std::env::var("RUST_LOG").is_err() {
        std::env::set_var("RUST_LOG", "info");
    }

    // Check system requirements
    if has_cuda_support() && is_cuda_available() {
        if let Ok(device_count) = CudaContext::get_device_count() {
            tracing::info!("CompressO-Cuda initialized with {} CUDA device(s)", device_count);
        }
    } else {
        tracing::warn!("CompressO-Cuda initialized in CPU-only mode");
    }

    Ok(())
}

/// Get system capabilities and information
pub fn get_system_capabilities() -> SystemCapabilities {
    let cuda_available = is_cuda_available();
    let mut cuda_devices = Vec::new();
    
    if cuda_available {
        if let Ok(device_count) = CudaContext::get_device_count() {
            for i in 0..device_count {
                if let Ok(info) = CudaContext::get_device_info(i) {
                    cuda_devices.push(info);
                }
            }
        }
    }

    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    SystemCapabilities {
        cuda_available,
        cuda_devices,
        ffmpeg_available,
        version: VERSION.to_string(),
        has_cuda_support: has_cuda_support(),
    }
}

/// System capabilities information
#[derive(Debug, Clone)]
pub struct SystemCapabilities {
    pub cuda_available: bool,
    pub cuda_devices: Vec<CudaDeviceInfo>,
    pub ffmpeg_available: bool,
    pub version: String,
    pub has_cuda_support: bool,
}

/// Presets for common compression scenarios
pub mod presets {
    use super::*;
    
    /// High quality preset for archival purposes
    pub fn high_quality() -> CompressionSettings {
        CompressionSettings {
            quality: CompressionQuality::High,
            codec: if is_cuda_available() { VideoCodec::H265_NVENC } else { VideoCodec::H265 },
            preset: if is_cuda_available() { EncodingPreset::P6 } else { EncodingPreset::Slow },
            profile: H264Profile::High,
            two_pass: true,
            ..Default::default()
        }
    }
    
    /// Balanced preset for general use
    pub fn balanced() -> CompressionSettings {
        CompressionSettings {
            quality: CompressionQuality::Medium,
            codec: if is_cuda_available() { VideoCodec::H264_NVENC } else { VideoCodec::H264 },
            preset: if is_cuda_available() { EncodingPreset::P4 } else { EncodingPreset::Fast },
            profile: H264Profile::High,
            ..Default::default()
        }
    }
    
    /// Fast preset for quick compression
    pub fn fast() -> CompressionSettings {
        CompressionSettings {
            quality: CompressionQuality::Medium,
            codec: if is_cuda_available() { VideoCodec::H264_NVENC } else { VideoCodec::H264 },
            preset: if is_cuda_available() { EncodingPreset::P1 } else { EncodingPreset::VeryFast },
            profile: H264Profile::Main,
            ..Default::default()
        }
    }
    
    /// Small file size preset
    pub fn small_size() -> CompressionSettings {
        CompressionSettings {
            quality: CompressionQuality::Low,
            codec: if is_cuda_available() { VideoCodec::H265_NVENC } else { VideoCodec::H265 },
            preset: if is_cuda_available() { EncodingPreset::P7 } else { EncodingPreset::VerySlow },
            profile: H264Profile::Main,
            two_pass: true,
            ..Default::default()
        }
    }
    
    /// Streaming preset for web/social media
    pub fn streaming() -> CompressionSettings {
        CompressionSettings {
            quality: CompressionQuality::Medium,
            codec: if is_cuda_available() { VideoCodec::H264_NVENC } else { VideoCodec::H264 },
            preset: if is_cuda_available() { EncodingPreset::P3 } else { EncodingPreset::Fast },
            profile: H264Profile::High,
            level: H264Level::L4_1,
            container_format: ContainerFormat::MP4,
            ..Default::default()
        }
    }
}

/// Utility functions for common operations
pub mod utils {
    use super::*;
    use std::path::Path;
    
    /// Calculate optimal resolution for target bitrate
    pub fn calculate_optimal_resolution(
        original_width: u32,
        original_height: u32,
        target_bitrate_kbps: u32,
    ) -> (u32, u32) {
        // Rough calculation based on common bitrate/resolution ratios
        let original_pixels = original_width * original_height;
        let bitrate_per_pixel = target_bitrate_kbps as f32 / original_pixels as f32;
        
        // Target around 0.1-0.2 kbps per pixel for good quality
        let scale_factor = if bitrate_per_pixel < 0.05 {
            0.75 // Reduce resolution
        } else if bitrate_per_pixel < 0.1 {
            0.85
        } else {
            1.0 // Keep original
        };
        
        let new_width = ((original_width as f32 * scale_factor) as u32 / 2) * 2; // Even numbers
        let new_height = ((original_height as f32 * scale_factor) as u32 / 2) * 2;
        
        (new_width.max(320), new_height.max(240)) // Minimum resolution
    }
    
    /// Estimate compression time based on input parameters
    pub fn estimate_compression_time(
        duration_seconds: f32,
        width: u32,
        height: u32,
        codec: &VideoCodec,
        preset: &EncodingPreset,
    ) -> f32 {
        let pixels_per_second = width * height * (30.0 * duration_seconds) as u32; // Assume 30fps
        
        // Base processing rate (pixels per second on average hardware)
        let base_rate = match codec {
            VideoCodec::H264_NVENC | VideoCodec::H265_NVENC => 100_000_000.0, // GPU is much faster
            VideoCodec::H264 => 20_000_000.0,
            VideoCodec::H265 => 10_000_000.0,
            VideoCodec::AV1 => 5_000_000.0,
            VideoCodec::VP9 => 8_000_000.0,
        };
        
        // Preset multiplier
        let preset_multiplier = match preset {
            EncodingPreset::UltraFast | EncodingPreset::P1 => 0.3,
            EncodingPreset::SuperFast | EncodingPreset::P2 => 0.5,
            EncodingPreset::VeryFast | EncodingPreset::P3 => 0.7,
            EncodingPreset::Faster | EncodingPreset::P4 => 1.0,
            EncodingPreset::Fast | EncodingPreset::P5 => 1.3,
            EncodingPreset::Medium | EncodingPreset::P6 => 1.8,
            EncodingPreset::Slow | EncodingPreset::P7 => 2.5,
            EncodingPreset::Slower => 3.5,
            EncodingPreset::VerySlow => 5.0,
        };
        
        (pixels_per_second as f32 / base_rate) * preset_multiplier
    }
    
    /// Check if a file is a supported video format
    pub fn is_supported_video_format(path: &Path) -> bool {
        if let Some(extension) = path.extension().and_then(|ext| ext.to_str()) {
            let ext_lower = extension.to_lowercase();
            matches!(ext_lower.as_str(), 
                "mp4" | "avi" | "mkv" | "mov" | "wmv" | "flv" | "webm" | 
                "m4v" | "3gp" | "ts" | "mts" | "m2ts" | "vob" | "asf"
            )
        } else {
            false
        }
    }
    
    /// Get recommended settings for a specific use case
    pub fn get_recommended_settings(use_case: UseCase) -> CompressionSettings {
        match use_case {
            UseCase::Archive => presets::high_quality(),
            UseCase::Web => presets::streaming(),
            UseCase::Mobile => {
                let mut settings = presets::fast();
                settings.resolution = Some((1280, 720)); // 720p for mobile
                settings.bitrate = Some(1500); // Lower bitrate
                settings
            },
            UseCase::Storage => presets::small_size(),
            UseCase::Editing => {
                let mut settings = presets::balanced();
                settings.container_format = ContainerFormat::MOV; // Better for editing
                settings.quality = CompressionQuality::High;
                settings
            },
        }
    }
}

/// Common use cases for video compression
#[derive(Debug, Clone, Copy)]
pub enum UseCase {
    Archive,   // High quality, larger files
    Web,       // Balanced for streaming
    Mobile,    // Optimized for mobile devices
    Storage,   // Smallest file size
    Editing,   // Optimized for video editing workflow
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_system_capabilities() {
        let caps = get_system_capabilities();
        assert_eq!(caps.version, VERSION);
        assert_eq!(caps.has_cuda_support, has_cuda_support());
    }
    
    #[test]
    fn test_presets() {
        let balanced = presets::balanced();
        assert_eq!(balanced.quality, CompressionQuality::Medium);
        
        let high_quality = presets::high_quality();
        assert_eq!(high_quality.quality, CompressionQuality::High);
    }
    
    #[test]
    fn test_resolution_calculation() {
        let (width, height) = utils::calculate_optimal_resolution(1920, 1080, 2000);
        assert!(width <= 1920);
        assert!(height <= 1080);
        assert!(width >= 320);
        assert!(height >= 240);
    }
}

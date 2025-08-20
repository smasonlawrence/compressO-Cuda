use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use tokio::process::Command as AsyncCommand;
use tracing::{debug, error, info, warn};
use regex::Regex;

use crate::cuda_wrapper::{CudaVideoProcessor, PreprocessingParams, is_cuda_available};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionSettings {
    // Basic settings
    pub quality: CompressionQuality,
    pub resolution: Option<(u32, u32)>,
    pub bitrate: Option<u32>,
    pub framerate: Option<f32>,
    
    // Audio settings
    pub mute_audio: bool,
    pub audio_bitrate: Option<u32>,
    
    // Advanced settings
    pub codec: VideoCodec,
    pub preset: EncodingPreset,
    pub profile: H264Profile,
    pub level: H264Level,
    
    // CUDA-specific settings
    pub use_cuda: bool,
    pub cuda_device_id: Option<i32>,
    pub preprocessing: PreprocessingParams,
    
    // Output settings
    pub container_format: ContainerFormat,
    pub two_pass: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionQuality {
    Low,
    Medium,
    High,
    Custom(u32), // CRF value
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoCodec {
    H264,
    H265,
    AV1,
    VP9,
    // CUDA-accelerated variants
    H264_NVENC,
    H265_NVENC,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncodingPreset {
    UltraFast,
    SuperFast,
    VeryFast,
    Faster,
    Fast,
    Medium,
    Slow,
    Slower,
    VerySlow,
    // NVENC presets
    P1, P2, P3, P4, P5, P6, P7,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum H264Profile {
    Baseline,
    Main,
    High,
    High10,
    High422,
    High444,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum H264Level {
    L3_0, L3_1, L4_0, L4_1, L4_2, L5_0, L5_1, L5_2,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContainerFormat {
    MP4,
    AVI,
    MKV,
    WebM,
    MOV,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionProgress {
    pub frame: u32,
    pub total_frames: u32,
    pub fps: f32,
    pub bitrate: f32,
    pub size: u64,
    pub speed: f32,
    pub time_elapsed: f32,
    pub eta: f32,
    pub percentage: f32,
}

pub struct CudaCompressionEngine {
    settings: CompressionSettings,
    cuda_processor: Option<CudaVideoProcessor>,
    ffmpeg_path: PathBuf,
    temp_dir: PathBuf,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            quality: CompressionQuality::Medium,
            resolution: None,
            bitrate: None,
            framerate: None,
            mute_audio: false,
            audio_bitrate: Some(128),
            codec: if is_cuda_available() { 
                VideoCodec::H264_NVENC 
            } else { 
                VideoCodec::H264 
            },
            preset: EncodingPreset::Fast,
            profile: H264Profile::High,
            level: H264Level::L4_1,
            use_cuda: is_cuda_available(),
            cuda_device_id: None,
            preprocessing: PreprocessingParams::default(),
            container_format: ContainerFormat::MP4,
            two_pass: false,
        }
    }
}

impl CudaCompressionEngine {
    pub fn new(settings: CompressionSettings) -> Result<Self> {
        let ffmpeg_path = Self::find_ffmpeg_binary()?;
        let temp_dir = std::env::temp_dir().join("compresso_cuda");
        std::fs::create_dir_all(&temp_dir)?;

        let cuda_processor = if settings.use_cuda && is_cuda_available() {
            match CudaVideoProcessor::new(settings.cuda_device_id) {
                Ok(mut processor) => {
                    // Initialize multiple streams for parallel processing
                    if let Err(e) = processor.initialize_streams(4) {
                        warn!("Failed to initialize CUDA streams: {}", e);
                    }
                    info!("CUDA acceleration enabled on device: {:?}", 
                          processor.device_info().name);
                    Some(processor)
                }
                Err(e) => {
                    warn!("Failed to initialize CUDA processor: {}, falling back to CPU", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(Self {
            settings,
            cuda_processor,
            ffmpeg_path,
            temp_dir,
        })
    }

    pub async fn compress_video<F>(
        &mut self,
        input_path: &Path,
        output_path: &Path,
        progress_callback: F,
    ) -> Result<()>
    where
        F: Fn(CompressionProgress) + Send + Sync + 'static,
    {
        info!("Starting video compression: {} -> {}", 
              input_path.display(), output_path.display());

        // Analyze input video
        let video_info = self.analyze_video(input_path).await?;
        debug!("Input video info: {:?}", video_info);

        // Prepare compression pipeline
        let compression_args = self.build_ffmpeg_args(&video_info, input_path, output_path)?;
        
        // Execute compression with progress monitoring
        self.execute_compression(compression_args, progress_callback, &video_info).await?;

        info!("Video compression completed successfully");
        Ok(())
    }

    async fn analyze_video(&self, input_path: &Path) -> Result<VideoInfo> {
        let mut cmd = AsyncCommand::new(&self.ffmpeg_path);
        cmd.args([
            "-i", input_path.to_str().unwrap(),
            "-f", "null", "-"
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

        let output = cmd.output().await?;
        let stderr = String::from_utf8_lossy(&output.stderr);
        
        self.parse_video_info(&stderr)
    }

    fn parse_video_info(&self, ffmpeg_output: &str) -> Result<VideoInfo> {
        let mut info = VideoInfo::default();
        
        // Parse duration
        if let Some(caps) = Regex::new(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})")
            .unwrap().captures(ffmpeg_output) {
            let hours: f32 = caps[1].parse().unwrap_or(0.0);
            let minutes: f32 = caps[2].parse().unwrap_or(0.0);
            let seconds: f32 = caps[3].parse().unwrap_or(0.0);
            info.duration = hours * 3600.0 + minutes * 60.0 + seconds;
        }
        
        // Parse video stream info
        for line in ffmpeg_output.lines() {
            if line.contains("Video:") {
                // Parse resolution
                if let Some(caps) = Regex::new(r"(\d+)x(\d+)")
                    .unwrap().captures(line) {
                    info.width = caps[1].parse().unwrap_or(1920);
                    info.height = caps[2].parse().unwrap_or(1080);
                }
                
                // Parse framerate
                if let Some(caps) = Regex::new(r"(\d+(?:\.\d+)?) fps")
                    .unwrap().captures(line) {
                    info.fps = caps[1].parse().unwrap_or(30.0);
                }
                
                // Parse codec
                if line.contains("h264") {
                    info.codec = "h264".to_string();
                } else if line.contains("hevc") || line.contains("h265") {
                    info.codec = "hevc".to_string();
                } else if line.contains("av1") {
                    info.codec = "av1".to_string();
                } else if line.contains("vp9") {
                    info.codec = "vp9".to_string();
                }
            }
        }
        
        info.total_frames = (info.duration * info.fps) as u32;
        Ok(info)
    }

    fn build_ffmpeg_args(
        &self,
        video_info: &VideoInfo,
        input_path: &Path,
        output_path: &Path,
    ) -> Result<Vec<String>> {
        let mut args = Vec::new();

        // Input settings
        args.extend([
            "-y".to_string(), // Overwrite output
            "-i".to_string(),
            input_path.to_str().unwrap().to_string(),
        ]);

        // CUDA hardware acceleration setup
        if self.cuda_processor.is_some() {
            match self.settings.codec {
                VideoCodec::H264_NVENC | VideoCodec::H265_NVENC => {
                    args.extend([
                        "-hwaccel".to_string(),
                        "cuda".to_string(),
                        "-hwaccel_output_format".to_string(),
                        "cuda".to_string(),
                    ]);
                    
                    if let Some(device_id) = self.settings.cuda_device_id {
                        args.extend([
                            "-hwaccel_device".to_string(),
                            device_id.to_string(),
                        ]);
                    }
                }
                _ => {}
            }
        }

        // Video codec settings
        match self.settings.codec {
            VideoCodec::H264 => {
                args.extend(["-c:v".to_string(), "libx264".to_string()]);
                self.add_x264_settings(&mut args);
            }
            VideoCodec::H264_NVENC => {
                args.extend(["-c:v".to_string(), "h264_nvenc".to_string()]);
                self.add_nvenc_settings(&mut args);
            }
            VideoCodec::H265 => {
                args.extend(["-c:v".to_string(), "libx265".to_string()]);
                self.add_x265_settings(&mut args);
            }
            VideoCodec::H265_NVENC => {
                args.extend(["-c:v".to_string(), "hevc_nvenc".to_string()]);
                self.add_nvenc_hevc_settings(&mut args);
            }
            VideoCodec::AV1 => {
                args.extend(["-c:v".to_string(), "libaom-av1".to_string()]);
                self.add_av1_settings(&mut args);
            }
            VideoCodec::VP9 => {
                args.extend(["-c:v".to_string(), "libvpx-vp9".to_string()]);
                self.add_vp9_settings(&mut args);
            }
        }

        // Quality settings
        match self.settings.quality {
            CompressionQuality::Low => {
                if self.is_nvenc_codec() {
                    args.extend(["-cq".to_string(), "35".to_string()]);
                } else {
                    args.extend(["-crf".to_string(), "28".to_string()]);
                }
            }
            CompressionQuality::Medium => {
                if self.is_nvenc_codec() {
                    args.extend(["-cq".to_string(), "25".to_string()]);
                } else {
                    args.extend(["-crf".to_string(), "23".to_string()]);
                }
            }
            CompressionQuality::High => {
                if self.is_nvenc_codec() {
                    args.extend(["-cq".to_string(), "18".to_string()]);
                } else {
                    args.extend(["-crf".to_string(), "18".to_string()]);
                }
            }
            CompressionQuality::Custom(crf) => {
                if self.is_nvenc_codec() {
                    args.extend(["-cq".to_string(), crf.to_string()]);
                } else {
                    args.extend(["-crf".to_string(), crf.to_string()]);
                }
            }
        }

        // Resolution settings with CUDA scaling
        if let Some((width, height)) = self.settings.resolution {
            if self.cuda_processor.is_some() && self.is_nvenc_codec() {
                // Use CUDA scaling for NVENC
                args.extend([
                    "-vf".to_string(),
                    format!("scale_cuda={}:{}", width, height),
                ]);
            } else {
                args.extend([
                    "-vf".to_string(),
                    format!("scale={}:{}", width, height),
                ]);
            }
        }

        // Framerate settings
        if let Some(fps) = self.settings.framerate {
            args.extend(["-r".to_string(), fps.to_string()]);
        }

        // Bitrate settings
        if let Some(bitrate) = self.settings.bitrate {
            args.extend([
                "-b:v".to_string(),
                format!("{}k", bitrate),
            ]);
        }

        // Audio settings
        if self.settings.mute_audio {
            args.push("-an".to_string());
        } else {
            args.extend(["-c:a".to_string(), "aac".to_string()]);
            if let Some(audio_bitrate) = self.settings.audio_bitrate {
                args.extend([
                    "-b:a".to_string(),
                    format!("{}k", audio_bitrate),
                ]);
            }
        }

        // Output format
        let extension = match self.settings.container_format {
            ContainerFormat::MP4 => "mp4",
            ContainerFormat::AVI => "avi",
            ContainerFormat::MKV => "mkv",
            ContainerFormat::WebM => "webm",
            ContainerFormat::MOV => "mov",
        };
        
        args.extend([
            "-f".to_string(),
            extension.to_string(),
        ]);

        // Progress reporting
        args.extend([
            "-progress".to_string(),
            "pipe:1".to_string(),
        ]);

        // Output file
        args.push(output_path.to_str().unwrap().to_string());

        Ok(args)
    }

    fn add_x264_settings(&self, args: &mut Vec<String>) {
        args.extend([
            "-preset".to_string(),
            self.preset_to_string(),
            "-profile:v".to_string(),
            self.profile_to_string(),
            "-level".to_string(),
            self.level_to_string(),
        ]);
    }

    fn add_nvenc_settings(&self, args: &mut Vec<String>) {
        args.extend([
            "-preset".to_string(),
            self.nvenc_preset_to_string(),
            "-profile:v".to_string(),
            self.profile_to_string(),
        ]);
        
        // NVENC-specific optimizations
        args.extend([
            "-rc".to_string(),
            "vbr".to_string(),
            "-surfaces".to_string(),
            "32".to_string(),
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-bf".to_string(),
            "3".to_string(),
            "-b_ref_mode".to_string(),
            "middle".to_string(),
        ]);
    }

    fn add_nvenc_hevc_settings(&self, args: &mut Vec<String>) {
        args.extend([
            "-preset".to_string(),
            self.nvenc_preset_to_string(),
            "-rc".to_string(),
            "vbr".to_string(),
            "-surfaces".to_string(),
            "64".to_string(),
            "-rc-lookahead".to_string(),
            "32".to_string(),
            "-tier".to_string(),
            "high".to_string(),
            "-spatial-aq".to_string(),
            "1".to_string(),
            "-temporal-aq".to_string(),
            "1".to_string(),
        ]);
    }

    fn add_x265_settings(&self, args: &mut Vec<String>) {
        args.extend([
            "-preset".to_string(),
            self.preset_to_string(),
            "-x265-params".to_string(),
            "aq-mode=3:aq-strength=1.0:deblock=1,1".to_string(),
        ]);
    }

    fn add_av1_settings(&self, args: &mut Vec<String>) {
        args.extend([
            "-cpu-used".to_string(),
            "6".to_string(),
            "-row-mt".to_string(),
            "1".to_string(),
        ]);
    }

    fn add_vp9_settings(&self, args: &mut Vec<String>) {
        args.extend([
            "-cpu-used".to_string(),
            "2".to_string(),
            "-row-mt".to_string(),
            "1".to_string(),
        ]);
    }

    async fn execute_compression<F>(
        &self,
        args: Vec<String>,
        progress_callback: F,
        video_info: &VideoInfo,
    ) -> Result<()>
    where
        F: Fn(CompressionProgress) + Send + Sync + 'static,
    {
        let mut cmd = AsyncCommand::new(&self.ffmpeg_path);
        cmd.args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .stdin(Stdio::null());

        debug!("Executing FFmpeg command: {:?} {:?}", self.ffmpeg_path, args);

        let mut child = cmd.spawn()?;
        
        // Monitor progress
        if let Some(stdout) = child.stdout.take() {
            let total_frames = video_info.total_frames;
            let progress_callback = std::sync::Arc::new(progress_callback);
            
            tokio::spawn(async move {
                use tokio::io::{AsyncBufReadExt, BufReader};
                
                let reader = BufReader::new(stdout);
                let mut lines = reader.lines();
                
                while let Ok(Some(line)) = lines.next_line().await {
                    if let Ok(progress) = Self::parse_progress_line(&line, total_frames) {
                        progress_callback(progress);
                    }
                }
            });
        }

        let output = child.wait_with_output().await?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            error!("FFmpeg error: {}", stderr);
            return Err(anyhow!("FFmpeg compression failed: {}", stderr));
        }

        Ok(())
    }

    fn parse_progress_line(line: &str, total_frames: u32) -> Result<CompressionProgress> {
        let mut progress = CompressionProgress {
            frame: 0,
            total_frames,
            fps: 0.0,
            bitrate: 0.0,
            size: 0,
            speed: 0.0,
            time_elapsed: 0.0,
            eta: 0.0,
            percentage: 0.0,
        };

        for part in line.split('=') {
            let parts: Vec<&str> = part.split('=').collect();
            if parts.len() == 2 {
                match parts[0] {
                    "frame" => progress.frame = parts[1].parse().unwrap_or(0),
                    "fps" => progress.fps = parts[1].parse().unwrap_or(0.0),
                    "bitrate" => {
                        let bitrate_str = parts[1].trim_end_matches("kbits/s");
                        progress.bitrate = bitrate_str.parse().unwrap_or(0.0);
                    }
                    "total_size" => progress.size = parts[1].parse().unwrap_or(0),
                    "speed" => {
                        let speed_str = parts[1].trim_end_matches('x');
                        progress.speed = speed_str.parse().unwrap_or(0.0);
                    }
                    _ => {}
                }
            }
        }

        // Calculate percentage and ETA
        if total_frames > 0 {
            progress.percentage = (progress.frame as f32 / total_frames as f32) * 100.0;
            
            if progress.fps > 0.0 {
                let remaining_frames = total_frames.saturating_sub(progress.frame);
                progress.eta = remaining_frames as f32 / progress.fps;
            }
        }

        Ok(progress)
    }

    fn find_ffmpeg_binary() -> Result<PathBuf> {
        let possible_paths = [
            "ffmpeg",
            "ffmpeg.exe",
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            "C:\\ffmpeg\\bin\\ffmpeg.exe",
            "C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        ];

        for path in &possible_paths {
            if let Ok(output) = Command::new(path).arg("-version").output() {
                if output.status.success() {
                    return Ok(PathBuf::from(path));
                }
            }
        }

        Err(anyhow!("FFmpeg binary not found. Please install FFmpeg and ensure it's in your PATH."))
    }

    fn is_nvenc_codec(&self) -> bool {
        matches!(self.settings.codec, VideoCodec::H264_NVENC | VideoCodec::H265_NVENC)
    }

    fn preset_to_string(&self) -> String {
        match self.settings.preset {
            EncodingPreset::UltraFast => "ultrafast",
            EncodingPreset::SuperFast => "superfast",
            EncodingPreset::VeryFast => "veryfast",
            EncodingPreset::Faster => "faster",
            EncodingPreset::Fast => "fast",
            EncodingPreset::Medium => "medium",
            EncodingPreset::Slow => "slow",
            EncodingPreset::Slower => "slower",
            EncodingPreset::VerySlow => "veryslow",
            _ => "fast",
        }.to_string()
    }

    fn nvenc_preset_to_string(&self) -> String {
        match self.settings.preset {
            EncodingPreset::P1 => "p1",
            EncodingPreset::P2 => "p2",
            EncodingPreset::P3 => "p3",
            EncodingPreset::P4 => "p4",
            EncodingPreset::P5 => "p5",
            EncodingPreset::P6 => "p6",
            EncodingPreset::P7 => "p7",
            _ => "p4",
        }.to_string()
    }

    fn profile_to_string(&self) -> String {
        match self.settings.profile {
            H264Profile::Baseline => "baseline",
            H264Profile::Main => "main",
            H264Profile::High => "high",
            H264Profile::High10 => "high10",
            H264Profile::High422 => "high422",
            H264Profile::High444 => "high444",
        }.to_string()
    }

    fn level_to_string(&self) -> String {
        match self.settings.level {
            H264Level::L3_0 => "3.0",
            H264Level::L3_1 => "3.1",
            H264Level::L4_0 => "4.0",
            H264Level::L4_1 => "4.1",
            H264Level::L4_2 => "4.2",
            H264Level::L5_0 => "5.0",
            H264Level::L5_1 => "5.1",
            H264Level::L5_2 => "5.2",
        }.to_string()
    }

    pub fn get_cuda_info(&self) -> Option<&crate::cuda_wrapper::CudaDeviceInfo> {
        self.cuda_processor.as_ref().map(|p| p.device_info())
    }

    pub fn supports_cuda(&self) -> bool {
        self.cuda_processor.is_some()
    }
}

#[derive(Debug, Default)]
struct VideoInfo {
    duration: f32,
    width: u32,
    height: u32,
    fps: f32,
    total_frames: u32,
    codec: String,
}

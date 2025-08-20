// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tauri::{Manager, State};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

mod cuda_wrapper;
mod compression;

use compression::{
    CompressionProgress, CompressionSettings, CudaCompressionEngine,
};
use cuda_wrapper::{get_best_cuda_device, is_cuda_available, CudaDeviceInfo, CudaContext};

// Application state
#[derive(Default)]
struct AppState {
    active_jobs: Arc<Mutex<HashMap<String, CompressionJobHandle>>>,
    compression_engine: Arc<Mutex<Option<CudaCompressionEngine>>>,
}

#[derive(Debug, Clone, Serialize)]
struct CompressionJobHandle {
    id: String,
    input_path: PathBuf,
    output_path: PathBuf,
    settings: CompressionSettings,
    status: JobStatus,
    progress: Option<CompressionProgress>,
    error: Option<String>,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
enum JobStatus {
    Pending,
    Running,
    Completed,
    Failed,
    Cancelled,
}

#[derive(Debug, Serialize)]
struct CudaInfo {
    available: bool,
    devices: Vec<CudaDeviceInfo>,
    best_device: Option<i32>,
    driver_version: Option<String>,
    runtime_version: Option<String>,
}

#[derive(Debug, Serialize)]
struct SystemInfo {
    cuda: CudaInfo,
    ffmpeg_available: bool,
    temp_dir: String,
    version: String,
}

// Tauri commands
#[tauri::command]
async fn get_system_info() -> Result<SystemInfo, String> {
    info!("Getting system information");

    let cuda_available = is_cuda_available();
    let mut devices = Vec::new();
    let best_device = if cuda_available {
        match get_best_cuda_device() {
            Ok(device_id) => {
                // Get info for all devices
                if let Ok(device_count) = CudaContext::get_device_count() {
                    for i in 0..device_count {
                        if let Ok(info) = CudaContext::get_device_info(i) {
                            devices.push(info);
                        }
                    }
                }
                Some(device_id)
            }
            Err(_) => None,
        }
    } else {
        None
    };

    let cuda_info = CudaInfo {
        available: cuda_available,
        devices,
        best_device,
        driver_version: None, // Could be implemented if needed
        runtime_version: None, // Could be implemented if needed
    };

    // Check FFmpeg availability
    let ffmpeg_available = std::process::Command::new("ffmpeg")
        .arg("-version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false);

    let temp_dir = std::env::temp_dir()
        .join("compresso_cuda")
        .to_string_lossy()
        .to_string();

    Ok(SystemInfo {
        cuda: cuda_info,
        ffmpeg_available,
        temp_dir,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[tauri::command]
async fn initialize_compression_engine(
    settings: CompressionSettings,
    state: State<'_, AppState>,
) -> Result<(), String> {
    info!("Initializing compression engine with settings: {:?}", settings);

    let engine = CudaCompressionEngine::new(settings)
        .map_err(|e| {
            error!("Failed to initialize compression engine: {}", e);
            format!("Failed to initialize compression engine: {}", e)
        })?;

    let mut engine_guard = state.compression_engine.lock().unwrap();
    *engine_guard = Some(engine);

    info!("Compression engine initialized successfully");
    Ok(())
}

#[tauri::command]
async fn start_compression(
    input_path: String,
    output_path: String,
    settings: CompressionSettings,
    state: State<'_, AppState>,
    app_handle: tauri::AppHandle,
) -> Result<String, String> {
    let job_id = Uuid::new_v4().to_string();
    let input_path = PathBuf::from(input_path);
    let output_path = PathBuf::from(output_path);

    info!(
        "Starting compression job {}: {} -> {}",
        job_id,
        input_path.display(),
        output_path.display()
    );

    // Validate input file exists
    if !input_path.exists() {
        return Err("Input file does not exist".to_string());
    }

    // Ensure output directory exists
    if let Some(parent) = output_path.parent() {
        if let Err(e) = std::fs::create_dir_all(parent) {
            return Err(format!("Failed to create output directory: {}", e));
        }
    }

    // Create job handle
    let job_handle = CompressionJobHandle {
        id: job_id.clone(),
        input_path: input_path.clone(),
        output_path: output_path.clone(),
        settings: settings.clone(),
        status: JobStatus::Pending,
        progress: None,
        error: None,
        created_at: chrono::Utc::now(),
        updated_at: chrono::Utc::now(),
    };

    // Add to active jobs
    {
        let mut jobs = state.active_jobs.lock().unwrap();
        jobs.insert(job_id.clone(), job_handle);
    }

    // Initialize engine if not already done
    {
        let mut engine_guard = state.compression_engine.lock().unwrap();
        if engine_guard.is_none() {
            match CudaCompressionEngine::new(settings.clone()) {
                Ok(engine) => *engine_guard = Some(engine),
                Err(e) => {
                    error!("Failed to initialize compression engine: {}", e);
                    return Err(format!("Failed to initialize compression engine: {}", e));
                }
            }
        }
    }

    // Start compression in background
    let job_id_clone = job_id.clone();
    let state_clone = state.inner().clone();
    let app_handle_clone = app_handle.clone();

    tokio::spawn(async move {
        let result = run_compression_job(
            job_id_clone.clone(),
            input_path,
            output_path,
            settings,
            state_clone,
            app_handle_clone,
        ).await;

        // Update job status
        let mut jobs = state_clone.active_jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&job_id_clone) {
            match result {
                Ok(_) => {
                    job.status = JobStatus::Completed;
                    info!("Compression job {} completed successfully", job_id_clone);
                }
                Err(e) => {
                    job.status = JobStatus::Failed;
                    job.error = Some(e.to_string());
                    error!("Compression job {} failed: {}", job_id_clone, e);
                }
            }
            job.updated_at = chrono::Utc::now();
        }
    });

    Ok(job_id)
}

async fn run_compression_job(
    job_id: String,
    input_path: PathBuf,
    output_path: PathBuf,
    settings: CompressionSettings,
    state: Arc<AppState>,
    app_handle: tauri::AppHandle,
) -> Result<()> {
    // Update job status to running
    {
        let mut jobs = state.active_jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&job_id) {
            job.status = JobStatus::Running;
            job.updated_at = chrono::Utc::now();
        }
    }

    // Get or create compression engine
    let mut engine = {
        let mut engine_guard = state.compression_engine.lock().unwrap();
        match engine_guard.take() {
            Some(engine) => engine,
            None => CudaCompressionEngine::new(settings.clone())?,
        }
    };

    // Set up progress callback
    let progress_callback = {
        let job_id = job_id.clone();
        let state = state.clone();
        let app_handle = app_handle.clone();
        
        move |progress: CompressionProgress| {
            // Update job progress
            {
                let mut jobs = state.active_jobs.lock().unwrap();
                if let Some(job) = jobs.get_mut(&job_id) {
                    job.progress = Some(progress.clone());
                    job.updated_at = chrono::Utc::now();
                }
            }

            // Emit progress event to frontend
            if let Err(e) = app_handle.emit_all("compression-progress", &progress) {
                warn!("Failed to emit progress event: {}", e);
            }
        }
    };

    // Run compression
    let result = engine.compress_video(&input_path, &output_path, progress_callback).await;

    // Put engine back
    {
        let mut engine_guard = state.compression_engine.lock().unwrap();
        *engine_guard = Some(engine);
    }

    result
}

#[tauri::command]
async fn cancel_compression(
    job_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    info!("Cancelling compression job: {}", job_id);

    let mut jobs = state.active_jobs.lock().unwrap();
    if let Some(job) = jobs.get_mut(&job_id) {
        if matches!(job.status, JobStatus::Running | JobStatus::Pending) {
            job.status = JobStatus::Cancelled;
            job.updated_at = chrono::Utc::now();
            info!("Compression job {} cancelled", job_id);
        } else {
            return Err("Job cannot be cancelled".to_string());
        }
    } else {
        return Err("Job not found".to_string());
    }

    Ok(())
}

#[tauri::command]
async fn get_compression_jobs(
    state: State<'_, AppState>,
) -> Result<Vec<CompressionJobHandle>, String> {
    let jobs = state.active_jobs.lock().unwrap();
    let mut job_list: Vec<CompressionJobHandle> = jobs.values().cloned().collect();
    
    // Sort by creation time (newest first)
    job_list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    
    Ok(job_list)
}

#[tauri::command]
async fn remove_compression_job(
    job_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let mut jobs = state.active_jobs.lock().unwrap();
    
    if let Some(job) = jobs.get(&job_id) {
        if matches!(job.status, JobStatus::Running) {
            return Err("Cannot remove running job".to_string());
        }
    }
    
    jobs.remove(&job_id);
    info!("Removed compression job: {}", job_id);
    Ok(())
}

#[tauri::command]
async fn clear_completed_jobs(
    state: State<'_, AppState>,
) -> Result<u32, String> {
    let mut jobs = state.active_jobs.lock().unwrap();
    let initial_count = jobs.len();
    
    jobs.retain(|_, job| !matches!(job.status, JobStatus::Completed | JobStatus::Failed | JobStatus::Cancelled));
    
    let removed_count = initial_count - jobs.len();
    info!("Cleared {} completed jobs", removed_count);
    Ok(removed_count as u32)
}

#[tauri::command]
async fn get_default_settings() -> Result<CompressionSettings, String> {
    Ok(CompressionSettings::default())
}

#[tauri::command]
async fn validate_file_path(path: String) -> Result<bool, String> {
    let path = PathBuf::from(path);
    Ok(path.exists() && path.is_file())
}

#[tauri::command]
async fn get_file_info(path: String) -> Result<FileInfo, String> {
    let path = PathBuf::from(path);
    
    if !path.exists() {
        return Err("File does not exist".to_string());
    }

    let metadata = std::fs::metadata(&path)
        .map_err(|e| format!("Failed to get file metadata: {}", e))?;

    let size = metadata.len();
    let modified = metadata.modified()
        .ok()
        .and_then(|time| chrono::DateTime::from(time).naive_utc().and_utc().into());

    let extension = path.extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or("")
        .to_string();

    let filename = path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_string();

    Ok(FileInfo {
        filename,
        size,
        extension,
        modified,
    })
}

#[derive(Debug, Serialize)]
struct FileInfo {
    filename: String,
    size: u64,
    extension: String,
    modified: Option<chrono::DateTime<chrono::Utc>>,
}

fn setup_logging() {
    use tracing_subscriber::{fmt, prelude::*, EnvFilter};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(fmt::layer().with_target(false))
        .with(filter)
        .init();
}

fn main() {
    setup_logging();
    
    info!("Starting CompressO-Cuda v{}", env!("CARGO_PKG_VERSION"));
    
    // Check system capabilities on startup
    if is_cuda_available() {
        if let Ok(device_count) = CudaContext::get_device_count() {
            info!("CUDA available with {} device(s)", device_count);
        }
    } else {
        warn!("CUDA not available, running in CPU-only mode");
    }

    tauri::Builder::default()
        .manage(AppState::default())
        .invoke_handler(tauri::generate_handler![
            get_system_info,
            initialize_compression_engine,
            start_compression,
            cancel_compression,
            get_compression_jobs,
            remove_compression_job,
            clear_completed_jobs,
            get_default_settings,
            validate_file_path,
            get_file_info,
        ])
        .setup(|app| {
            info!("Application setup completed");
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
